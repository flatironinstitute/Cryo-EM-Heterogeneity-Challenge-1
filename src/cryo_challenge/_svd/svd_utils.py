import torch
from typing import Tuple


def get_vols_svd(
    volumes: torch.tensor,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Compute the singular value decomposition of the input volumes. The volumes are flattened so that each volume is a row in the input matrix.

    Parameters
    ----------
    volumes: torch.tensor
        Tensor of shape (n_volumes, n_x, n_y, n_z) containing the volumes to be decomposed.

    Returns
    -------
    U: torch.tensor
        Left singular vectors of the input volumes.
    S: torch.tensor
        Singular values of the input volumes.
    V: torch.tensor
        Right singular vectors of the input volumes.

    Examples
    --------
    >>> volumes = torch.randn(10, 32, 32, 32)
    >>> U, S, V = get_vols_svd(volumes)
    """  # noqa: E501
    assert volumes.ndim == 4, "Input volumes must have shape (n_volumes, n_x, n_y, n_z)"
    assert volumes.shape[0] > 0, "Input volumes must have at least one volume"

    U, S, V = torch.linalg.svd(
        volumes.reshape(volumes.shape[0], -1), full_matrices=False
    )
    return U, S, V


def project_vols_to_svd(
    volumes: torch.tensor, V_reference: torch.tensor
) -> torch.tensor:
    """
    Project the input volumes onto the right singular vectors.

    Parameters
    ----------
    volumes: torch.tensor
        Tensor of shape (n_volumes, n_x, n_y, n_z) containing the volumes to be projected.
    V_reference: torch.tensor
        Right singular vectors of the reference volumes.

    Returns
    -------
    coeffs: torch.tensor
        Coefficients of the input volumes projected onto the right singular vectors.

    Examples
    --------
    >>> volumes1 = torch.randn(10, 32, 32, 32)
    >>> volumes2 = torch.randn(10, 32, 32, 32)
    >>> U, S, V = get_vols_svd(volumes1)
    >>> coeffs = project_vols_to_svd(volumes2, V)
    """  # noqa: E501
    assert volumes.ndim == 4, "Input volumes must have shape (n_volumes, n_x, n_y, n_z)"
    assert volumes.shape[0] > 0, "Input volumes must have at least one volume"
    assert (
        V_reference.ndim == 2
    ), "Right singular vectors must have shape (n_features, n_components)"
    assert (
        volumes.shape[1] * volumes.shape[2] * volumes.shape[3] == V_reference.shape[1]
    ), "Number of features in volumes must match number of features in right singular vectors"  # noqa: E501

    coeffs = volumes.reshape(volumes.shape[0], -1) @ V_reference.T

    return coeffs
