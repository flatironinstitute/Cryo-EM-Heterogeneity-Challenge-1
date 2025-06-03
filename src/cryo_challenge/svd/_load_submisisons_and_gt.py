import logging
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import Dict, List
from typing_extensions import Literal
import os
from natsort import natsorted
import numpy as np
import mrcfile
import glob
from pathlib import PurePath, Path

from ..preprocessing._downsampling import downsample_submission
from ..preprocessing._global_alignment import threshold_submission
from ..utils._whitening import WhiteningFilter
from ..config_validation._svd_analysis_validation import SVDInputConfig
from ..utils._fourier_statistics import compute_radially_averaged_powerspectrum_on_grid
from pydantic import DirectoryPath, FilePath


class DatasetForSVD(Dataset):
    submission_files: List[Path]

    def __init__(
        self,
        submission_files: List[FilePath],
    ):
        # check submission files are valid
        for file in submission_files:
            _is_valid_submission_file(file)
        self.submission_files = submission_files

    def __len__(self):
        return len(self.submission_files)

    def __getitem__(self, idx) -> Dict[str, Tensor]:
        submission = torch.load(
            self.submission_files[idx], mmap=False, weights_only=False
        )
        return submission


def load_submissions(svd_input_config: SVDInputConfig) -> DatasetForSVD:
    """
    Load the submissions from the specified directory and prepare them for SVD analysis.

    !!!Note
        This function will load submission files, prepared them according to `svd_input_config.normalize_params`. The normalized submissions will be saved in a temporary directory for later use. If the `continue_from_previous` flag is set to `True`, it will load the prepared submissions from the previous run instead of reprocessing them. The output is given as a `DatasetForSVD` object, which will load the processed submissions saved on disk in the form of a dictionary.
    **Arguments:**
        - `svd_input_config`: The configuration object containing the parameters for loading and preparing the submissions. See the `cryo_challenge.config_validation.SVDInputConfig` for details.
    **Returns:**
        - `DatasetForSVD`: A dataset object containing the prepared submissions. Elements in the datased are dictionaries with the following keys
        - `id`: The ID of the submission.
        - `volumes`: The volumes of the submission.
        - `whitening_filter`: The whitening filter used for the submission.
        - `power_spectrum_on_grid`: The power spectrum of the submission.
        - `populations`: The populations of the submission.
    """
    submission_files = _get_submission_files(
        svd_input_config.path_to_submissions, svd_input_config.excluded_submissions
    )
    if svd_input_config.continue_from_previous:
        logging.info("  Loading prepared submissions from previous run...")
        prepared_sub_files = _load_submissions_from_previous(
            submission_files=submission_files,
            path_to_output_dir=svd_input_config.output_params["path_to_output_dir"],
            normalize_params=svd_input_config.normalize_params,
            dtype=svd_input_config.dtype,
        )

    else:
        logging.info("  Loading submissions from scratch...")
        prepared_sub_files = _load_submissions_from_scrath(
            submission_files=submission_files,
            path_to_output_dir=svd_input_config.output_params["path_to_output_dir"],
            normalize_params=svd_input_config.normalize_params,
            dtype=svd_input_config.dtype,
        )
    return DatasetForSVD(prepared_sub_files)


def load_gt(svd_input_config: SVDInputConfig) -> DatasetForSVD:
    """
    Load the ground truth data from the specified directory and prepare it for SVD analysis.

    !!!Note
        This function will load the ground truth data, prepare it according to `svd_input_config.normalize_params` and `svd_input_config.gt_params`, and save the normalized data in a temporary directory for later use. If the `continue_from_previous` flag is set to `True`, it will load the prepared ground truth data from the previous run instead of reprocessing it. The output is given as a `DatasetForSVD` object, which will load the processed ground truth data saved on disk in the form of a dictionary.

    **Arguments:**
        - `svd_input_config`: The configuration object containing the parameters for loading and preparing the ground truth data. See the `cryo_challenge.config_validation.SVDInputConfig` for details.

    **Returns:**
        - `DatasetForSVD`: A dataset object containing the prepared ground truth data. To load the data, simply use index 0. The elements in the dataset are dictionaries with the following keys
        - `id`: The ID of the ground truth data.
        - `volumes`: The volumes of the ground truth data.
        - `whitening_filter`: The whitening filter used for the ground truth data.
        - `power_spectrum_on_grid`: The power spectrum of the ground truth data.
        - `populations`: The populations of the ground truth data.
    """
    tmp_dir = os.path.join(
        svd_input_config.output_params["path_to_output_dir"],
        "prepared_submissions_for_svd/",
    )
    fname_for_saving = os.path.join(tmp_dir, "ground_truth_svd.pt")

    if os.path.exists(fname_for_saving) and svd_input_config.continue_from_previous:
        logging.info(
            f"  Loading prepared ground truth from previous run: {fname_for_saving}"
        )
        gt_data = DatasetForSVD([fname_for_saving])

    else:
        logging.info("Loading ground truth from scratch...")
        gt_data = _load_gt_from_scratch(
            path_to_gt_volumes=svd_input_config.gt_params["path_to_gt_volumes"],
            path_to_output_dir=svd_input_config.output_params["path_to_output_dir"],
            skip_vols=svd_input_config.gt_params["skip_vols"],
            normalize_params=svd_input_config.normalize_params,
            dtype=svd_input_config.dtype,
        )
        logging.info("  ... done")
    return gt_data


def _is_valid_submission_file(filename: FilePath) -> None:
    submission_metadata = torch.load(filename, mmap=True, weights_only=False)
    if "id" not in submission_metadata:
        raise ValueError(f"Submission file {filename} does not contain 'id' key.")
    if "volumes" not in submission_metadata:
        raise ValueError(f"Submission file {filename} does not contain 'volumes' key.")
    if "populations" not in submission_metadata:
        raise ValueError(
            f"Submission file {filename} does not contain 'populations' key."
        )
    if "whitening_filter" not in submission_metadata:
        raise ValueError(
            f"Submission file {filename} does not contain 'whitening_filter' key."
        )
    if "power_spectrum_on_grid" not in submission_metadata:
        raise ValueError(
            f"Submission file {filename} does not contain 'power_spectrum_on_grid' key."
        )
    return


def _get_submission_files(
    path_to_submissions: DirectoryPath, excluded_submissions: List[str]
) -> List[FilePath]:
    submission_files = glob.glob(os.path.join(path_to_submissions, "*.pt"))
    submission_files = [PurePath(file) for file in submission_files]
    filtered_submission_files = []
    for file in submission_files:
        if "submission" in file.name:
            if file.name not in excluded_submissions:
                filtered_submission_files.append(file)
    return natsorted(filtered_submission_files)


def _load_submissions_from_previous(
    submission_files: List[FilePath],
    path_to_output_dir: DirectoryPath,
    normalize_params: dict = {
        "path_to_mask": None,
        "downsample_box_size": None,
        "threshold_percentile": None,
    },
    dtype: Literal["float32", "float64"] = "float32",
) -> List[FilePath]:
    prepared_sub_files = []
    not_prepared_sub_files = []

    tmp_dir = os.path.join(path_to_output_dir, "prepared_submissions_for_svd/")
    for file in submission_files:
        fname_for_saving = os.path.join(tmp_dir, file.name.replace(".pt", "_svd.pt"))
        if os.path.exists(fname_for_saving):
            prepared_sub_files.append(fname_for_saving)
        else:
            logging.info(f"    Submission {file} not found in {tmp_dir}.")
            not_prepared_sub_files.append(file)

    if len(not_prepared_sub_files) > 0:
        logging.info(
            f"    Preparing {len(not_prepared_sub_files)} missing submissions..."
        )
        prepared_sub_files += _load_submissions_from_scrath(
            submission_files=not_prepared_sub_files,
            path_to_output_dir=path_to_output_dir,
            normalize_params=normalize_params,
            dtype=dtype,
        )
        logging.info("    ... done")

    return prepared_sub_files


def _load_submissions_from_scrath(
    submission_files: List[FilePath],
    path_to_output_dir: DirectoryPath,
    normalize_params: dict = {
        "path_to_mask": None,
        "downsample_box_size": None,
        "threshold_percentile": None,
    },
    dtype: Literal["float32", "float64"] = "float32",
) -> List[FilePath]:
    box_size = torch.load(submission_files[0], weights_only=False, mmap=True)[
        "volumes"
    ].shape[-1]

    mask = (
        _validate_mask(
            normalize_params["path_to_mask"],
            box_size,
        )
        if normalize_params["path_to_mask"] is not None
        else None
    )

    tmp_dir = os.path.join(path_to_output_dir, "prepared_submissions_for_svd/")

    prepared_subs_files = []
    for file in tqdm(submission_files, desc="Loading submissions"):
        submission_data = _load_submission(
            file,
            normalize_params["downsample_box_size"],
            normalize_params["threshold_percentile"],
            mask,
            dtype,
        )
        logging.info(f"    Saving submission: {submission_data['id']}")
        fname_for_saving = os.path.join(tmp_dir, file.name.replace(".pt", "_svd.pt"))
        prepared_subs_files.append(fname_for_saving)
        torch.save(submission_data, fname_for_saving)

    return prepared_subs_files


def _load_gt_from_scratch(
    path_to_gt_volumes: str | Path,
    path_to_output_dir: str | Path,
    skip_vols: int = 1,
    normalize_params: dict = {
        "path_to_mask": None,
        "downsample_box_size": None,
        "threshold_percentile": None,
    },
    dtype: Literal["float32", "float64"] = "float32",
) -> Dict[str, Tensor]:
    volumes = torch.load(path_to_gt_volumes, mmap=True, weights_only=False)

    if len(volumes.shape) == 2:
        box_size_gt = int(round((float(volumes.shape[-1]) ** (1.0 / 3.0))))

    elif len(volumes.shape) == 4:
        box_size_gt = volumes.shape[-1]

    else:
        raise ValueError(
            "The shape of the ground truth volumes is not valid. "
            "It should be either (N, box_size^3) or (N, box_size, box_size, box_size)."
        )

    mask = (
        _validate_mask(
            normalize_params["path_to_mask"],
            box_size_gt,
        )
        if normalize_params["path_to_mask"] is not None
        else None
    )

    volumes = volumes[::skip_vols]
    volumes = volumes.reshape(volumes.shape[0], box_size_gt, box_size_gt, box_size_gt)

    if mask is not None:
        logging.info("    Applying mask...")
        volumes *= mask

    if normalize_params["threshold_percentile"] is not None:
        logging.info(
            f"    Thresholding at {normalize_params['threshold_percentile']} percentile..."
        )
        volumes = threshold_submission(
            volumes, normalize_params["threshold_percentile"]
        )

    if normalize_params["downsample_box_size"] is not None:
        logging.info(
            f"    Downsampling to box size {normalize_params['downsample_box_size']}..."
        )
        volumes = downsample_submission(
            volumes, box_size_ds=normalize_params["downsample_box_size"]
        )

    volumes /= torch.norm(volumes.reshape(volumes.shape[0], -1), dim=-1)[
        :, None, None, None
    ]

    if dtype == "float32":
        volumes = volumes.float()
    else:
        volumes = volumes.double()

    logging.info("    Computing power spectrum...")
    power_spectrum_on_grid = compute_radially_averaged_powerspectrum_on_grid(volumes)

    logging.info("    Computing whitening filter...")
    whitening_filter = WhiteningFilter(
        power_spectrum_on_grid,
        get_squared=False,
    )

    gt_data = {
        "id": "ground_truth",
        "volumes": volumes,
        "whitening_filter": whitening_filter,
        "power_spectrum_on_grid": power_spectrum_on_grid,
        "populations": None,
    }

    tmp_dir = os.path.join(path_to_output_dir, "prepared_submissions_for_svd/")
    fname_for_saving = os.path.join(tmp_dir, "ground_truth_svd.pt")
    torch.save(gt_data, fname_for_saving)

    return DatasetForSVD([fname_for_saving])


def _load_submission(
    path_to_submission: str | Path,
    downsample_box_size: int,
    threshold_percentile: float,
    mask: Tensor,
    dtype: Literal["float32", "float64"],
) -> tuple[str, dict]:
    submission = torch.load(path_to_submission, weights_only=False, mmap=True)

    label = submission["id"]

    logging.info(f"    Loading submission: {label}")
    populations = submission["populations"]
    populations /= populations.sum()
    if isinstance(populations, np.ndarray):
        populations = torch.from_numpy(populations)

    volumes = submission["volumes"].clone()

    if mask is not None:
        logging.info("    Applying mask...")
        volumes *= mask[None, ...]

    if threshold_percentile is not None:
        logging.info(f"    Thresholding at {threshold_percentile} percentile...")
        volumes = threshold_submission(volumes, threshold_percentile)

    if downsample_box_size is not None:
        logging.info(f"    Downsampling to box size {downsample_box_size}...")
        volumes = downsample_submission(volumes, downsample_box_size)

    volumes /= torch.norm(volumes.reshape(volumes.shape[0], -1), dim=-1)[
        :, None, None, None
    ]

    if dtype == "float32":
        volumes = volumes.float()

    elif dtype == "float64":
        volumes = volumes.double()

    logging.info("    Computing power spectrum...")
    power_spectrum_on_grid = compute_radially_averaged_powerspectrum_on_grid(volumes)

    logging.info("    Computing whitening filter...")
    whitening_filter = WhiteningFilter(
        power_spectrum_on_grid,
        get_squared=False,
    )

    del submission

    return {
        "id": label,
        "volumes": volumes,
        "whitening_filter": whitening_filter,
        "power_spectrum_on_grid": power_spectrum_on_grid,
        "populations": populations,
    }


def _validate_mask(path_to_mask: str | Path, box_size: int) -> Tensor:
    mask = torch.tensor(mrcfile.open(path_to_mask, mode="r").data.copy())
    try:
        mask = mask.reshape(box_size, box_size, box_size)
    except RuntimeError:
        raise ValueError(
            "Mask shape does not match the box size of the volumes in the submissions."
        )
    return mask
