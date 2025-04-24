import torch
import numpy as np


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


# This used to align all the volumes. Too slow.

# def align_submission(
#     volumes: torch.Tensor, ref_volume: torch.Tensor, params: dict
# ) -> torch.Tensor:
#     """
#     Align submission volumes to ground truth volume

#     Parameters:
#     -----------
#     volumes (torch.Tensor): submission volumes
#         shape: (n_volumes, im_x, im_y, im_z)
#     ref_volume (torch.Tensor): ground truth volume
#         shape: (im_x, im_y, im_z)
#     params (dict): dictionary containing alignment parameters

#     Returns:
#     --------
#     volumes (torch.Tensor): aligned submission volumes
#     """
#     for i in range(len(volumes)):
#         obj_vol = volumes[i].numpy().astype(np.float32).copy()

#         obj_vol = Volume(obj_vol / obj_vol.sum())
#         ref_vol = Volume(ref_volume.copy() / ref_volume.sum())

#         _, R_est = align_BO(
#             ref_vol,
#             obj_vol,
#             loss_type=params["BOT_loss"],
#             downsampled_size=params["BOT_box_size"],
#             max_iters=params["BOT_iter"],
#             refine=params["BOT_refine"],
#         )
#         R_est = Rotation(R_est.astype(np.float32))

#         volumes[i] = torch.from_numpy(Volume(volumes[i].numpy()).rotate(R_est)._data)

#     return volumes


def align_submission_to_reference(
    volumes: torch.Tensor,
    ref_volume: torch.Tensor,
    *,
    downsampled_size: int,
    loss_type: str,
    max_iters: int,
    refine: bool,
) -> torch.Tensor:
    """
    Align submission volumes to ground truth volume

    Parameters:
    -----------
    volumes (torch.Tensor): submission volumes
        shape: (n_volumes, Nz, Ny, Nx)
    ref_volume (torch.Tensor): ground truth volume
        shape: (Nz, Ny, Nx)
    params (dict): dictionary containing alignment parameters

    Returns:
    --------
    volumes (torch.Tensor): aligned submission volumes
    """

    from aspire.volume import Volume
    from aspire.utils.rotation import Rotation
    from aspire.utils.bot_align import align_BO

    obj_vol = volumes[0].numpy().astype(np.float32)

    obj_vol = Volume(obj_vol / obj_vol.sum())
    ref_vol = Volume(ref_volume / ref_volume.sum())

    _, R_est = align_BO(
        ref_vol,
        obj_vol,
        loss_type=loss_type,
        downsampled_size=downsampled_size,
        max_iters=max_iters,
        refine=refine,
    )
    R_est = Rotation(R_est.astype(np.float32))

    volumes = torch.from_numpy(Volume(volumes.numpy()).rotate(R_est)._data)

    return volumes
