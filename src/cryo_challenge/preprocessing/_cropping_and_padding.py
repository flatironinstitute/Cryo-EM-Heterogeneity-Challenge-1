import torch
import numpy as np

from typing import Tuple
from torch import Tensor


def _get_slices_for_cropping(shape, cropped_shape):
    Nz, Ny, Nx = shape
    zc, yc, xc = Nz // 2, Ny // 2, Nx // 2
    d, h, w = cropped_shape

    start_z = zc - d // 2
    end_z = zc + d // 2 + d % 2

    start_y = yc - h // 2
    end_y = yc + h // 2 + h % 2

    start_x = xc - w // 2
    end_x = xc + w // 2 + w % 2

    return slice(start_z, end_z), slice(start_y, end_y), slice(start_x, end_x)


def crop_volume_to_shape(volume: Tensor, new_shape: Tuple[int, int, int]) -> Tensor:
    """
    Crop 3D volume to specified shape

    Parameters:
    -----------
    volume (torch.Tensor): 3D volume
        shape: (Nz, Ny, Nx)
    new_shape (Tuple[int, int, int]): shape of the cropped volume.

    Returns:
    --------
    crop_vol (torch.Tensor): cropped 3D volume
    """

    if volume.dim() != 3:
        raise ValueError("Volume must be 3D")

    slices = _get_slices_for_cropping(volume.shape, new_shape)
    return volume[slices]


@torch.no_grad()
def crop_volume_to_box_size(volume: Tensor, new_box_size) -> Tensor:
    """
    Crop 3D volume to specified box size

    Parameters:
    -----------
    volume (torch.Tensor): 3D volume
        shape: (Nz, Ny, Nx)
    new_box_size (int): box size of the cropped volume.

    Returns:
    --------
    cropped_volume (torch.Tensor): cropped 3D volume
    """

    return crop_volume_to_shape(volume, (new_box_size, new_box_size, new_box_size))


def crop_submission_to_box_size(volumes: Tensor, new_box_size: int) -> Tensor:
    """
    Crop submission volumes to specified box size

    Parameters:
    -----------
    volumes (torch.Tensor): submission volumes
        shape: (n_volumes, Ny, Nz, Nx)
    new_box_size (int): box size to crop volumes to

    Returns:
    --------
    vols_crop (torch.Tensor): cropped submission volumes
    """
    if volumes.dim() != 4:
        raise ValueError("Volumes must have shape (n_volumes, Nz, Ny, Nx)")

    slices = _get_slices_for_cropping(
        volumes.shape[1:], (new_box_size, new_box_size, new_box_size)
    )
    return volumes[:, slices[0], slices[1], slices[2]]


def pad_submission_to_box_size(volumes: Tensor, box_size_pad: int) -> Tensor:
    """
    Pad submission volumes to specified box size

    Parameters:
    -----------
    volumes (torch.Tensor): submission volumes
        shape: (n_volumes, Nz, Ny, Nx)
    box_size_pad (int): box size to pad volumes to

    Returns:
    --------
    vols_pad (torch.Tensor): padded submission volumes
    """

    box_size_sub = volumes.shape[-1]
    pad_len = (box_size_pad - box_size_sub) // 2
    vols_pad = torch.nn.functional.pad(
        volumes, (pad_len, pad_len, pad_len, pad_len, pad_len, pad_len), "constant", 0
    )

    return vols_pad


def crop_or_pad_submission(
    volumes: Tensor,
    box_size_gt: int,
    pixel_size_sub: int,
    pixel_size_gt: int,
) -> Tensor:
    """
    Crop or pad submission volumes to match ground truth pixel size and box size

    Parameters:
    -----------
    volumes (torch.Tensor): submission volumes
        shape: (n_volumes, Ny, Nz, Nx)
    box_size_gt (int): ground truth box size
    pixel_size_sub (float): submission pixel size
    pixel_size_gt (float): ground truth pixel size

    Returns:
    --------
    volumes (torch.Tensor): cropped or padded submission volumes
    """
    box_size_sub = volumes.shape[-1]
    if pixel_size_sub > pixel_size_gt:
        raise ValueError(
            "Submission pixel size is larger than ground truth pixel size. "
            "Please check the input parameters."
        )

    elif pixel_size_sub == pixel_size_gt:
        if box_size_sub == box_size_gt:
            pass

        elif box_size_sub > box_size_gt:
            volumes = crop_submission_to_box_size(volumes, box_size_gt)

        elif box_size_sub < box_size_gt:
            volumes = pad_submission_to_box_size(volumes, box_size_gt)

    elif pixel_size_sub < pixel_size_gt:
        target_box_size = np.round(box_size_gt * pixel_size_gt / pixel_size_sub).astype(
            int
        )

        if target_box_size == box_size_sub:
            pass

        elif target_box_size > box_size_sub:
            volumes = pad_submission_to_box_size(volumes, target_box_size)

        elif target_box_size < box_size_sub:
            volumes = crop_submission_to_box_size(volumes, target_box_size)

    return volumes
