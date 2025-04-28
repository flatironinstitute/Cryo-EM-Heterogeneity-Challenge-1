import torch
import numpy as np
from typing import Literal, Any, Optional
from pydantic import PositiveFloat, PositiveInt


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
    init_rot_guess_params: dict,
    *,
    loss_type: Literal["wemd", "euclidean"] = "wemd",
    loss_params: Optional[Any] = None,
    downsampled_size: PositiveInt = 32,
    refinement_downsampled_size: PositiveInt = 32,
    max_iters: PositiveInt = 200,
    refine: bool = True,
    tau: PositiveFloat = 1e-3,
    surrogate_max_iter: PositiveFloat = 500,
    surrogate_min_grad: PositiveFloat = 0.1,
    surrogate_min_step: PositiveFloat = 0.1,
    verbosity: Literal[0, 1, 2] = 0,
    dtype: Optional[Any] = None,
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
    from scipy.spatial.transform import Rotation as R

    R_guess = (
        R.from_euler(
            init_rot_guess_params["seq"],
            angles=init_rot_guess_params["angles"],
            degrees=init_rot_guess_params["degrees"],
        )
        .as_matrix()
        .astype(np.float32)
    )

    obj_vol = volumes.mean(0).numpy().astype(np.float32)

    obj_vol = Volume(obj_vol / obj_vol.sum())
    ref_vol = Volume(ref_volume / ref_volume.sum())

    obj_vol = obj_vol.rotate(Rotation(R_guess))

    _, R_est = align_BO(
        ref_vol,
        obj_vol,
        loss_type=loss_type,
        loss_params=loss_params,
        downsampled_size=downsampled_size,
        refinement_downsampled_size=refinement_downsampled_size,
        max_iters=max_iters,
        refine=refine,
        tau=tau,
        surrogate_max_iter=surrogate_max_iter,
        surrogate_min_grad=surrogate_min_grad,
        surrogate_min_step=surrogate_min_step,
        verbosity=verbosity,
        dtype=dtype,
    )
    R_est = R_est.astype(np.float32)
    full_rotation = Rotation(R_est @ R_guess)

    volumes = torch.from_numpy(Volume(volumes.numpy()).rotate(full_rotation)._data)

    return volumes
