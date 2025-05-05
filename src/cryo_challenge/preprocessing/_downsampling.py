"""
Downsampling functions are based on CryoDRGN's downsampling functions
https://github.com/ml-struct-bio/cryodrgn/blob/09c6f616be69e1bd392907ed966ad7304acc6ea8/cryodrgn/commands/downsample.py#L154 # noqa: E501

These use the Hartley transform instead of the Fourier transform to downsample the volume.
"""

import torch
from torch import Tensor

from ..fft._fourier import htn_center
from ._cropping_and_padding import crop_volume_to_box_size


@torch.no_grad()
def downsample_volume(volume: Tensor, box_size_ds: int) -> Tensor:
    """
    Downsample 3D volume in Fourier Space to specified box size

    Parameters:
    -----------
    volume (Tensor): 3D volume
        shape: (im_x, im_y, im_z)
    box_size_ds (int): box size to downsample volume to

    Returns:
    --------
    vol_ds (Tensor): downsampled 3D volume
    """
    return htn_center(crop_volume_to_box_size(htn_center(volume), box_size_ds))


def downsample_submission(volumes: Tensor, box_size_ds: int) -> Tensor:
    """
    Downsample submission volumes in Fourier space to specified box size.

    Parameters:
    -----------
    volumes (Tensor): submission volumes
        shape: (n_volumes, im_x, im_y, im_z)
    box_size_ds (int): box size to downsample volumes to
        shape: (im_x, im_y, im_z)

    Returns:
    --------
    volumes (Tensor): downsampled submission volumes
    """

    if volumes.shape[-1] == box_size_ds:
        return volumes

    else:
        volumes = torch.vmap(downsample_volume, in_dims=(0, None), chunk_size=20)(
            volumes, box_size_ds
        )
        return volumes
