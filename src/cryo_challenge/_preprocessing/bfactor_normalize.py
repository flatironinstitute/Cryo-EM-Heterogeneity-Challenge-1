import torch
from ..power_spectrum_utils import _centered_fftn, _centered_ifftn


def _compute_bfactor_scaling(b_factor, box_size, voxel_size):
    x = torch.fft.fftshift(torch.fft.fftfreq(box_size, d=voxel_size))
    y = x.clone()
    z = x.clone()
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")

    s2 = x**2 + y**2 + z**2
    bfactor_scaling_torch = torch.exp(-b_factor * s2 / 4)

    return bfactor_scaling_torch


def bfactor_normalize_volumes(volumes, bfactor, voxel_size, in_place=True):
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

    else:
        raise ValueError("Input volumes must have 3 or 4 dimensions.")

    return volumes
