import torch
import numpy as np
from functools import partial


@torch.no_grad()
def crop_vol_3d(volume: torch.Tensor, box_size_crop: int) -> torch.Tensor:
    """
    Crop 3D volume to specified box size

    Parameters:
    -----------
    volume (torch.Tensor): 3D volume
        shape: (im_x, im_y, im_z)
    box_size_crop (int): box size to crop volume to

    Returns:
    --------
    crop_vol (torch.Tensor): cropped 3D volume
    """
    im_y, im_x, im_z = volume.shape

    # shift terms
    start_x = im_x // 2 - box_size_crop // 2
    start_y = im_y // 2 - box_size_crop // 2
    start_z = im_z // 2 - box_size_crop // 2

    crop_vol = volume[
        start_x : start_x + box_size_crop,
        start_y : start_y + box_size_crop,
        start_z : start_z + box_size_crop,
    ]

    return crop_vol


def crop_submission(volumes: torch.Tensor, box_size_crop: int) -> torch.Tensor:
    """
    Crop submission volumes to specified box size

    Parameters:
    -----------
    volumes (torch.Tensor): submission volumes
        shape: (n_volumes, im_x, im_y, im_z)
    box_size_crop (int): box size to crop volumes to

    Returns:
    --------
    vols_crop (torch.Tensor): cropped submission volumes
    """

    wrap_crop_vol_3d = partial(crop_vol_3d, box_size_crop=box_size_crop)
    vols_crop = torch.vmap(wrap_crop_vol_3d)(volumes)

    return vols_crop


def pad_submission(volumes: torch.Tensor, box_size_pad: int) -> torch.Tensor:
    """
    Pad submission volumes to specified box size

    Parameters:
    -----------
    volumes (torch.Tensor): submission volumes
        shape: (n_volumes, im_x, im_y, im_z)
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


def crop_pad_submission(
    volumes: torch.Tensor, box_size_gt: int, pixel_size_sub: int, pixel_size_gt: int
) -> torch.Tensor:
    """
    Crop or pad submission volumes to match ground truth pixel size and box size

    Parameters:
    -----------
    volumes (torch.Tensor): submission volumes
        shape: (n_volumes, im_x, im_y, im_z)
    box_size_gt (int): ground truth box size
    pixel_size_sub (float): submission pixel size
    pixel_size_gt (float): ground truth pixel size

    Returns:
    --------
    volumes (torch.Tensor): cropped or padded submission volumes
    """
    box_size_sub = volumes.shape[-1]
    success = 0
    if pixel_size_sub > pixel_size_gt:
        print("Pixel size is less than ground truth pixel size")

    elif pixel_size_sub == pixel_size_gt:
        if box_size_sub == box_size_gt:
            success = 1

        elif box_size_sub > box_size_gt:
            volumes = crop_submission(volumes, box_size_gt)
            success = 1

        elif box_size_sub < box_size_gt:
            volumes = pad_submission(volumes, box_size_gt)
            success = 1

    elif pixel_size_sub < pixel_size_gt:
        target_box_size = np.round(box_size_gt * pixel_size_gt / pixel_size_sub).astype(
            int
        )

        if target_box_size == box_size_sub:
            success = 1

        elif target_box_size > box_size_sub:
            volumes = pad_submission(volumes, target_box_size)
            success = 1

        elif target_box_size < box_size_sub:
            volumes = crop_submission(volumes, target_box_size)
            success = 1

    return volumes, success
