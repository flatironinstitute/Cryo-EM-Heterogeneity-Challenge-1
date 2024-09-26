import torch
import numpy as np
from typing import Tuple
import os
from natsort import natsorted
import mrcfile

from ..._preprocessing.fourier_utils import downsample_volume, downsample_submission
from ..._preprocessing.bfactor_normalize import bfactor_normalize_volumes


def load_submissions_svd(
    config: dict,
) -> Tuple[torch.tensor, dict]:
    """
    Load the volumes and populations from the submissions specified in submission_list. Volumes are first downsampled, then normalized so that they sum to 1, and finally the mean volume is removed from each submission.

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration parameters.
    Returns
    -------
    submissions_data: dict
        Dictionary containing the populations, left singular vectors, singular values, and right singular vectors of each submission.
    """  # noqa: E501

    path_to_submissions = config["path_to_submissions"]
    excluded_submissions = config["excluded_submissions"]

    submissions_data = {}

    submission_files = []
    for file in os.listdir(path_to_submissions):
        if file.endswith(".pt") and "submission" in file:
            if file in excluded_submissions:
                continue
            submission_files.append(file)
    submission_files = natsorted(submission_files)

    vols = torch.load(os.path.join(path_to_submissions, submission_files[0]))["volumes"]
    box_size = vols.shape[-1]

    if config["normalize_params"]["mask_path"] is not None:
        mask = torch.tensor(
            mrcfile.open(config["normalize_params"]["mask_path"], mode="r").data.copy()
        )
        try:
            mask = mask.reshape(1, box_size, box_size, box_size)
        except RuntimeError:
            raise ValueError(
                "Mask shape does not match the box size of the volumes in the submissions."
            )

    for file in submission_files:
        sub_path = os.path.join(path_to_submissions, file)
        submission = torch.load(sub_path)

        label = submission["id"]
        populations = submission["populations"]

        if not isinstance(populations, torch.Tensor):
            populations = torch.tensor(populations)

        volumes = submission["volumes"]
        if config["normalize_params"]["mask_path"] is not None:
            volumes = volumes * mask

        if config["normalize_params"]["bfactor"] is not None:
            voxel_size = config["voxel_size"]
            volumes = bfactor_normalize_volumes(
                volumes,
                config["normalize_params"]["bfactor"],
                voxel_size,
                in_place=True,
            )

        if config["normalize_params"]["box_size_ds"] is not None:
            volumes = downsample_submission(
                volumes, box_size_ds=config["normalize_params"]["box_size_ds"]
            )
            box_size = config["normalize_params"]["box_size_ds"]
        else:
            box_size = volumes.shape[-1]

        volumes = volumes.reshape(-1, box_size * box_size * box_size)

        if config["dtype"] == "float32":
            volumes = volumes.float()
        elif config["dtype"] == "float64":
            volumes = volumes.double()

        volumes /= torch.norm(volumes, dim=1, keepdim=True)

        if config["svd_max_rank"] is None:
            u_matrices, singular_values, eigenvectors = torch.linalg.svd(
                volumes - volumes.mean(0, keepdim=True), full_matrices=False
            )
            eigenvectors = eigenvectors.T

        else:
            u_matrices, singular_values, eigenvectors = torch.svd_lowrank(
                volumes - volumes.mean(0, keepdim=True), q=config["svd_max_rank"]
            )

        submissions_data[label] = {
            "populations": populations / populations.sum(),
            "u_matrices": u_matrices.clone(),
            "singular_values": singular_values.clone(),
            "eigenvectors": eigenvectors.clone(),
        }

    return submissions_data


def load_gt_svd(config: dict) -> dict:
    """
    Load the ground truth volumes, downsample them, normalize them, and remove the mean volume. Then compute the SVD of the volumes.

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration parameters.

    Returns
    -------
    gt_data: dict
        Dictionary containing the left singular vectors, singular values, and right singular vectors of the ground truth volumes.
    """

    vols_gt = np.load(config["gt_params"]["gt_vols_file"], mmap_mode="r")

    if len(vols_gt.shape) == 2:
        box_size_gt = int(round((float(vols_gt.shape[-1]) ** (1.0 / 3.0))))

    elif len(vols_gt.shape) == 4:
        box_size_gt = vols_gt.shape[-1]

    if config["normalize_params"]["box_size_ds"] is not None:
        box_size = config["normalize_params"]["box_size_ds"]
    else:
        box_size = box_size_gt

    if config["normalize_params"]["mask_path"] is not None:
        mask = torch.tensor(
            mrcfile.open(config["normalize_params"]["mask_path"], mode="r").data.copy()
        )

        try:
            mask = mask.reshape(box_size_gt, box_size_gt, box_size_gt)
        except RuntimeError:
            raise ValueError(
                "Mask shape does not match the box size of the volumes in the submissions."
            )

    skip_vols = config["gt_params"]["skip_vols"]
    n_vols = vols_gt.shape[0] // skip_vols

    if config["dtype"] == "float32":
        dtype = torch.float32

    else:
        dtype = torch.float64

    volumes_gt = torch.zeros((n_vols, box_size * box_size * box_size), dtype=dtype)

    for i in range(n_vols):
        vol_tmp = torch.from_numpy(
            vols_gt[i * skip_vols].copy().reshape(box_size_gt, box_size_gt, box_size_gt)
        )

        if dtype == torch.float32:
            vol_tmp = vol_tmp.float()
        else:
            vol_tmp = vol_tmp.double()

        if config["normalize_params"]["mask_path"] is not None:
            vol_tmp *= mask

        if config["normalize_params"]["bfactor"] is not None:
            bfactor = config["normalize_params"]["bfactor"]
            voxel_size = config["voxel_size"]
            vol_tmp = bfactor_normalize_volumes(
                vol_tmp, bfactor, voxel_size, in_place=True
            )

        if config["normalize_params"]["box_size_ds"] is not None:
            vol_tmp = downsample_volume(vol_tmp, box_size_ds=box_size)

        volumes_gt[i] = vol_tmp.reshape(-1)
    volumes_gt /= torch.norm(volumes_gt, dim=1, keepdim=True)

    U, S, V = torch.svd_lowrank(
        volumes_gt - volumes_gt.mean(0, keepdim=True), q=config["svd_max_rank"]
    )

    gt_data = {
        "u_matrices": U.clone(),
        "singular_values": S.clone(),
        "eigenvectors": V.clone(),
    }

    return gt_data
