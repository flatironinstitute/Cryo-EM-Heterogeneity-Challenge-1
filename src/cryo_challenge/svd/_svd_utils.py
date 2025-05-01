import torch
from torch import Tensor
import numpy as np
from typing import List
from tqdm import tqdm
from ._load_submisisons_and_gt import DatasetForSVD
from ..fft._fourier import rfftn, irfftn


def compute_common_power_spectrum_on_grid(
    dataset_for_svd: DatasetForSVD,
) -> Tensor:
    """
    Computes a common power spectrum on grid for all submissions.
    The common power spectrum is computed by taking the minimum value
    of the power spectrum on grid for each submission at each frequency shell.

    This will be used to normalize the b-factor of the volumes before computing
    the SVD metrics.
    **Arguments:**
        dataset_for_svd (DatasetForSVD): The dataset containing the submissions.
    **Returns:**
        common_power_spectrum_on_grid (Tensor): The common power spectrum on grid.
    """
    common_power_spectrum_on_grid = dataset_for_svd[0]["power_spectrum_on_grid"]
    for i in range(1, len(dataset_for_svd)):
        submission = dataset_for_svd[i]

        common_power_spectrum_on_grid = torch.where(
            common_power_spectrum_on_grid < submission["power_spectrum_on_grid"],
            common_power_spectrum_on_grid,
            submission["power_spectrum_on_grid"],
        )

    return common_power_spectrum_on_grid


def compute_svd_of_submission(
    submission: dict,
    power_spectrum_on_grid: Tensor = None,
    svd_max_rank: int = None,
):
    if power_spectrum_on_grid is None:
        vols_norm = submission["volumes"].clone()
        vols_norm = vols_norm.reshape(vols_norm.shape[0], -1)

    else:
        vols_norm = torch.sqrt(power_spectrum_on_grid) * submission["whitening_filter"](
            rfftn(submission["volumes"].clone())
        )

        vols_norm = irfftn(vols_norm).reshape(vols_norm.shape[0], -1)

    if svd_max_rank is not None:
        u, s, v = torch.svd_lowrank(
            vols_norm - vols_norm.mean(dim=0, keepdim=True), q=svd_max_rank
        )

    else:
        u, s, v = torch.linalg.svd(
            vols_norm - vols_norm.mean(dim=0, keepdim=True), full_matrices=False
        )
        v = v.T  # set eigenvectors to low_rank format

    return {
        "u_matrices": u,
        "singular_values": s,
        "eigenvectors": v,
    }


def compute_svd_for_all_submission(
    submissions_data: DatasetForSVD,
    power_spectrum_on_grid: Tensor = None,
    svd_max_rank: int = None,
):
    submissions_svd = {}
    for i in tqdm(range(len(submissions_data)), desc="Computing SVD"):
        submission = submissions_data[i]
        svd_results = compute_svd_of_submission(
            submission,
            power_spectrum_on_grid=power_spectrum_on_grid,
            svd_max_rank=svd_max_rank,
        )
        submissions_svd[submission["id"]] = svd_results

    return submissions_svd


def sort_distance_matrix_with_ref_label(
    dist_matrix: Tensor, labels: List, ref_label: str
):
    """
    Sorts the distance matrix and labels based on the reference label.
    The reference label is moved to the top left corner of the distance matrix.

    **Arguments:**
        dist_matrix (Tensor): The distance matrix to be sorted.
        labels (List): The labels corresponding to the distance matrix.
        ref_label (str): The reference label to sort the matrix by.
    **Returns:**
        dist_matrix (Tensor): The sorted distance matrix.
        labels (List): The sorted labels.

    """
    assert ref_label in labels, f"Label {ref_label} not found in labels."

    labels = np.array(labels).copy()
    dist_matrix = dist_matrix.clone()
    sort_idx = np.where(labels == ref_label)[0][0]
    sorting_indices = torch.argsort(dist_matrix[sort_idx, :])
    dist_matrix = dist_matrix.T[:, sorting_indices][sorting_indices].T
    labels = labels[sorting_indices.numpy()]

    dist_matrix = torch.flip(dist_matrix, dims=(0,))
    dist_matrix = torch.flip(dist_matrix, dims=(1,))
    labels = np.flip(labels)

    return dist_matrix, labels.tolist()
