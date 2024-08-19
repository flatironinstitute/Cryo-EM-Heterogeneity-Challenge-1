import torch
import numpy as np
from scipy.ndimage import shift
from aspire.volume import Volume
from aspire.utils.rotation import Rotation
from aspire.utils.bot_align import align_BO


def generate_grid1d(box_size: int, pixel_size: float = 1):
    grid_limit = pixel_size * box_size * 0.5
    grid = np.arange(-grid_limit, grid_limit, pixel_size)[0:box_size]
    return grid


def center(
    vol: np.array, pixel_size: float, order_shift: int, threshold: float = -np.inf
):
    """
    Center 3D volume so that its center of mass is at 0 in all dimensions.

    Parameters
    ----------
    vol : np.array
        3D volume to center
    order_shift : int
        Order of spline interpolation (see scipy.ndimage.shift for more details)
    threshold : float
        Threshold value to apply to the volume (values below this threshold will be set to 0)

    Returns
    -------
    vol : np.array
        Centered 3D volume
    center_of_mass : np.array
        Center of mass of the volume
    """  # noqa: E501
    vol = np.copy(vol)
    vol.setflags(write=1)
    vol[vol < threshold] = 0
    vol = vol / vol.sum()

    grid = generate_grid1d(vol.shape[-1], pixel_size)
    center_of_mass = np.zeros(3)

    for i, ax in enumerate([(1, 2), (0, 2), (0, 1)]):
        center_of_mass[i] = np.sum(vol, axis=ax) @ grid

    vol = shift(vol, -center_of_mass, order=order_shift, mode="constant")

    return vol, center_of_mass


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


def center_volume(volume: torch.Tensor, pixel_size: float) -> torch.Tensor:
    """
    Center 3D volume

    Parameters:
    -----------
    volume (torch.Tensor): 3D volume
        shape: (im_x, im_y, im_z)
    pixel_size (float): pixel size

    Returns:
    --------
    vol (torch.Tensor): centered 3D volume
    """

    vol = volume.numpy().copy()
    vol, _ = center(vol, pixel_size, 0)

    return torch.from_numpy(vol)


def center_submission(volumes: torch.Tensor, pixel_size: float) -> torch.Tensor:
    """
    Center submission volumes

    Parameters:
    -----------
    volumes (torch.Tensor): submission volumes
        shape: (n_volumes, im_x, im_y, im_z)
    pixel_size (float): pixel size

    Returns:
    --------
    volumes (torch.Tensor): centered submission volumes
    """

    for i in range(volumes.shape[0]):
        volumes[i] = center_volume(volumes[i], pixel_size)

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
    for i in range(len(volumes)):
        print('aligning ' + str(i) + 'th volume' )
        obj_vol = volumes[i].numpy().astype(np.float32)

        obj_vol = Volume(obj_vol / obj_vol.sum())
        ref_vol = Volume(ref_volume / ref_volume.sum())

        _, R_est = align_BO(
            ref_vol,
            obj_vol,
            loss_type=params["BOT_loss"],
            downsampled_size=params["BOT_box_size"],
            max_iters=params["BOT_iter"],
            refine=params["BOT_refine"],
        )
        R_est = Rotation(R_est.astype(np.float32))

        volumes[i] = torch.from_numpy(Volume(volumes[i].numpy()).rotate(R_est)._data)

    return volumes
