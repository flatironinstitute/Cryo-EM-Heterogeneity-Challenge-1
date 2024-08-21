import torch
from ..power_spectrum_utils import _centered_fftn, _centered_ifftn


def _compute_bfactor_scaling(b_factor, box_size, voxel_size):
    """
    Compute the B-factor scaling factor for a given B-factor, box size, and voxel size.
    The B-factor scaling factor is computed as exp(-B * s^2 / 4), where s is the squared
    distance in Fourier space.

    Parameters
    ----------
    b_factor: float
        B-factor to apply.
    box_size: int
        Size of the box.
    voxel_size: float
        Voxel size of the box.

    Returns
    -------
    bfactor_scaling_torch: torch.tensor(shape=(box_size, box_size, box_size))
        B-factor scaling factor.
    """
    x = torch.fft.fftshift(torch.fft.fftfreq(box_size, d=voxel_size))
    y = x.clone()
    z = x.clone()
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")

    s2 = x**2 + y**2 + z**2
    bfactor_scaling_torch = torch.exp(-b_factor * s2 / 4)

    return bfactor_scaling_torch


def bfactor_normalize_volumes(volumes, bfactor, voxel_size, in_place=False):
    """
    Normalize volumes by applying a B-factor correction. This is done by multiplying
    a centered Fourier transform of the volume by the B-factor scaling factor and then
    applying the inverse Fourier transform. See _compute_bfactor_scaling for details on the
    computation of the B-factor scaling.

    Parameters
    ----------
    volumes: torch.tensor
        Volumes to normalize. The volumes must have shape (N, N, N) or (n_volumes, N, N, N).
    bfactor: float
        B-factor to apply.
    voxel_size: float
        Voxel size of the volumes.
    in_place: bool - default: False
        Whether to normalize the volumes in place.

    Returns
    -------
    volumes: torch.tensor
        Normalized volumes.
    """
    # assert that volumes have the correct shape
    assert volumes.ndim in [
        3,
        4,
    ], "Input volumes must have shape (N, N, N) or (n_volumes, N, N, N)"

    if volumes.ndim == 3:
        assert (
            volumes.shape[0] == volumes.shape[1] == volumes.shape[2]
        ), "Input volumes must have equal dimensions"
    else:
        assert (
            volumes.shape[1] == volumes.shape[2] == volumes.shape[3]
        ), "Input volumes must have equal dimensions"

    if not in_place:
        volumes = volumes.clone()

    b_factor_scaling = _compute_bfactor_scaling(bfactor, volumes.shape[-1], voxel_size)

    if len(volumes.shape) == 3:
        volumes = _centered_fftn(volumes, dim=(0, 1, 2))
        volumes = volumes * b_factor_scaling
        volumes = _centered_ifftn(volumes, dim=(0, 1, 2)).real

    elif len(volumes.shape) == 4:
        volumes = _centered_fftn(volumes, dim=(1, 2, 3))
        volumes = volumes * b_factor_scaling[None, ...]
        volumes = _centered_ifftn(volumes, dim=(1, 2, 3)).real

    return volumes
