import torch
import numpy as np
from typing import List


def sort_distance_matrix_with_ref_label(
    dist_matrix: torch.Tensor, labels: List, ref_label: str
):
    """
    Sorts the distance matrix and labels based on the reference label.
    The reference label is moved to the top left corner of the distance matrix.

    **Arguments:**
        dist_matrix (torch.Tensor): The distance matrix to be sorted.
        labels (List): The labels corresponding to the distance matrix.
        ref_label (str): The reference label to sort the matrix by.
    **Returns:**
        dist_matrix (torch.Tensor): The sorted distance matrix.
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
