import torch
from torch.fft import fftn, fftshift
from .crop_pad_utils import crop_vol_3d


def fftn_center(input, norm="forward"):
    return fftshift(fftn(fftshift(input), norm=norm))


def htn_center(input):
    output = fftn_center(input, norm="ortho")
    return output.real - output.imag


@torch.no_grad()
def downsample_volume(volume: torch.Tensor, box_size_ds: int) -> torch.Tensor:
    """
    Downsample 3D volume in Fourier Space to specified box size

    Parameters:
    -----------
    volume (torch.Tensor): 3D volume
        shape: (im_x, im_y, im_z)
    box_size_ds (int): box size to downsample volume to

    Returns:
    --------
    vol_ds (torch.Tensor): downsampled 3D volume
    """
    return htn_center(crop_vol_3d(htn_center(volume), box_size_ds))


def downsample_submission(
    volumes: torch.Tensor, box_size_ds: int, *, chunk_size=1
) -> torch.Tensor:
    """
    Downsample submission volumes in Fourier space to specified box size.

    Parameters:
    -----------
    volumes (torch.Tensor): submission volumes
        shape: (n_volumes, im_x, im_y, im_z)
    box_size_ds (int): box size to downsample volumes to
    chunck_size (int): chunk size for vmap
        default: 1 (equivalent to for loop)

    Returns:
    --------
    volumes (torch.Tensor): downsampled submission volumes
    """

    downsample_volume_batch = torch.vmap(
        downsample_volume, in_dims=(0, None), chunk_size=chunk_size
    )

    if volumes.shape[-1] == box_size_ds:
        return volumes

    else:
        return downsample_volume_batch(volumes, box_size_ds)
