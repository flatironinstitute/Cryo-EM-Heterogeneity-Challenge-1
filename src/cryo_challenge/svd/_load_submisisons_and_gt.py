import torch
from typing import Dict, List
import os
from natsort import natsorted
import mrcfile
import glob
from pathlib import PurePath

from ..preprocessing._downsampling import downsample_submission
from ..config_validation._svd_analysis_validation import SVDInputConfig
from pydantic import DirectoryPath, FilePath


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


def load_submissions(
    config: SVDInputConfig,
) -> Dict[str, torch.Tensor]:
    submission_files = _get_submission_files(
        config.path_to_submissions, config.excluded_submissions
    )

    submissions_data = {}

    box_size = torch.load(submission_files[0], weights_only=False, mmap=True)[
        "volumes"
    ].shape[-1]

    if config.normalize_params["mask_path"] is not None:
        mask = torch.tensor(
            mrcfile.open(config.normalize_params["mask_path"], mode="r").data.copy()
        )
        try:
            mask = mask.reshape(box_size, box_size, box_size)
        except RuntimeError:
            raise ValueError(
                "Mask shape does not match the box size of the volumes in the submissions."
            )
        mask = mask[None, ...]

    else:
        mask = None

    for file in submission_files:
        submission = torch.load(file, weights_only=False)
        volumes = submission["volumes"]

        if mask is not None:
            volumes = volumes * mask

        if config.normalize_params["downsample_box_size"] is not None:
            volumes = downsample_submission(
                volumes, box_size_ds=config.normalize_params["downsample_box_size"]
            )

        volumes = volumes.reshape(volumes.shape[0], -1)

        if config.dtype == "float32":
            volumes = volumes.float()
        elif config.dtype == "float64":
            volumes = volumes.double()

        volumes /= torch.norm(volumes, dim=1, keepdim=True)

        if config.svd_max_rank is None:
            u_matrices, singular_values, eigenvectors = torch.linalg.svd(
                volumes - volumes.mean(0, keepdim=True), full_matrices=False
            )
            eigenvectors = eigenvectors.T

        else:
            u_matrices, singular_values, eigenvectors = torch.svd_lowrank(
                volumes - volumes.mean(0, keepdim=True), q=config.svd_max_rank
            )

        submissions_data[submission["id"]] = {
            "populations": submission["populations"],
            "u_matrices": u_matrices.clone(),
            "singular_values": singular_values.clone(),
            "eigenvectors": eigenvectors.clone(),
        }

    return submissions_data


def load_gt(config: SVDInputConfig) -> Dict[str, torch.Tensor]:
    volumes_gt = torch.load(
        config.gt_params["path_to_gt_volumes"], mmap=True, weights_only=False
    )

    if len(volumes_gt.shape) == 2:
        box_size_gt = int(round((float(volumes_gt.shape[-1]) ** (1.0 / 3.0))))

    elif len(volumes_gt.shape) == 4:
        box_size_gt = volumes_gt.shape[-1]

    if config.normalize_params["downsample_box_size"] is not None:
        box_size = config.normalize_params["downsample_box_size"]
    else:
        box_size = box_size_gt

    if config.normalize_params["mask_path"] is not None:
        mask = torch.tensor(
            mrcfile.open(config.normalize_params["mask_path"], mode="r").data.copy()
        )

        try:
            mask = mask.reshape(box_size_gt, box_size_gt, box_size_gt)[None, ...]
        except RuntimeError:
            raise ValueError(
                "Mask shape does not match the box size of the volumes in the submissions."
            )
    else:
        mask = None

    if config.dtype == "float32":
        dtype = torch.float32

    else:
        dtype = torch.float64

    volumes_gt = volumes_gt[:: config.gt_params["skip_vols"]]
    volumes_gt = volumes_gt.reshape(
        volumes_gt.shape[0], box_size_gt, box_size_gt, box_size_gt
    )

    if dtype == torch.float32:
        volumes_gt = volumes_gt.float()
    else:
        volumes_gt = volumes_gt.double()

    if mask is not None:
        volumes_gt *= mask

    if config.normalize_params["downsample_box_size"] is not None:
        volumes_gt = downsample_submission(volumes_gt, box_size_ds=box_size)

    volumes_gt = volumes_gt.reshape(volumes_gt.shape[0], -1)
    volumes_gt /= torch.norm(volumes_gt, dim=1, keepdim=True)

    if config.svd_max_rank is None:
        u_matrices, singular_values, eigenvectors = torch.linalg.svd(
            volumes_gt - volumes_gt.mean(0, keepdim=True), full_matrices=False
        )
        eigenvectors = eigenvectors.T

    else:
        u_matrices, singular_values, eigenvectors = torch.svd_lowrank(
            volumes_gt - volumes_gt.mean(0, keepdim=True), q=config.svd_max_rank
        )

    gt_data = {
        "u_matrices": u_matrices.clone(),
        "singular_values": singular_values.clone(),
        "eigenvectors": eigenvectors.clone(),
    }

    return gt_data
