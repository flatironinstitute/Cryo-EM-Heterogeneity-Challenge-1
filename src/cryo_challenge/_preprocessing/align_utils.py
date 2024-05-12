import torch
import numpy as np
from aspire.volume import Volume
from aspire.utils.rotation import Rotation
from BOTalign.utils_BO import center, align_BO


def threshold_submissions(
    volumes: torch.Tensor, thresh_percentile: float
) -> torch.Tensor:
    """
    Threshold submission volumes to specified percentile

    Parameters:
    -----------
    volumes (torch.Tensor): submission volumes
        shape: (n_volumes, im_x, im_y, im_z)
    thresh_percentile (float): percentile to threshold volumes to

    Returns:
    --------
    volumes (torch.Tensor): thresholded submission volumes
    """

    for i in range(volumes.shape[0]):
        thresh = np.percentile(volumes[i].flatten(), thresh_percentile)
        volumes[i][volumes[i] < thresh] = 0.0

    return volumes


def center_volume(volume: torch.Tensor) -> torch.Tensor:
    """
    Center 3D volume

    Parameters:
    -----------
    volume (torch.Tensor): 3D volume
        shape: (im_x, im_y, im_z)

    Returns:
    --------
    vol_copy (torch.Tensor): centered 3D volume
    """

    vol_copy = volume.numpy().copy()
    vol_copy, _ = center(vol_copy, 0)

    return torch.from_numpy(vol_copy)


def center_submission(volumes: torch.Tensor) -> torch.Tensor:
    """
    Center submission volumes

    Parameters:
    -----------
    volumes (torch.Tensor): submission volumes
        shape: (n_volumes, im_x, im_y, im_z)

    Returns:
    --------
    volumes (torch.Tensor): centered submission volumes
    """

    for i in range(volumes.shape[0]):
        volumes[i] = center_volume(volumes[i])

    return volumes


def align_submission(
    volumes: torch.Tensor, ref_volume: torch.Tensor, params: dict
) -> torch.Tensor:
    """
    Align submission volumes to ground truth volume

    Parameters:
    -----------
    volumes (torch.Tensor): submission volumes
        shape: (n_volumes, im_x, im_y, im_z)
    ref_volume (torch.Tensor): ground truth volume
        shape: (im_x, im_y, im_z)
    params (dict): dictionary containing alignment parameters

    Returns:
    --------
    volumes (torch.Tensor): aligned submission volumes
    """
    obj_vol = volumes[0].numpy()
    obj_vol = Volume(obj_vol / obj_vol.sum())
    ref_vol = Volume(ref_volume / ref_volume.sum())

    _, R_est = align_BO(
        obj_vol,
        ref_vol,
        para=[
            params["BOT_loss"],
            params["BOT_box_size"],
            params["BOT_iter"],
            params["BOT_refine"],
        ],
        reflect=params["BOT_reflect"],
    )
    R_est = Rotation(R_est.T)

    volumes = torch.from_numpy(Volume(volumes.numpy()).rotate(R_est)._data)

    return volumes
