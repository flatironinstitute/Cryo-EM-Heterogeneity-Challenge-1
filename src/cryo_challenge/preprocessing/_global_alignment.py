import torch
import numpy as np
from typing import Literal, Any, Optional, Tuple
from pydantic import PositiveFloat, PositiveInt
from scipy.ndimage import shift


def center_submission(volumes, order_shift):
    v = volumes.mean(0).clone().numpy()
    v = v / v.sum()

    L = volumes.shape[-1]
    X = np.zeros(L)
    mid = int(L / 2)
    for i in range(L):
        X[i] = i - mid

    vx = np.sum(v, axis=(1, 2))
    vy = np.sum(v, axis=(0, 2))
    vz = np.sum(v, axis=(0, 1))
    m = np.array([X @ vx, X @ vy, X @ vz])

    for i in range(volumes.shape[0]):
        volumes[i] = torch.from_numpy(
            shift(volumes[i].numpy(), m, order_shift=order_shift)
        )
    return volumes


def threshold_volume(volume: torch.Tensor, thresh_percentile: float) -> torch.Tensor:
    """
    Threshold a volume to specified percentile

    Parameters:
    -----------
    volume (torch.Tensor): volume
        shape: (im_x, im_y, im_z)
    thresh_percentile (float): percentile to threshold volume to

    Returns:
    --------
    volume (torch.Tensor): thresholded volume
    """

    # thresh = torch.quantile(volume.flatten(), thresh_percentile / 100.0)
    thresh = torch.tensor(np.percentile(volume.flatten().numpy(), thresh_percentile))
    volume[volume < thresh] = 0.0

    return volume


def threshold_submission(
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
        volumes[i] = threshold_volume(volumes[i], thresh_percentile)

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
    threshold_percentile: float,
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
) -> Tuple[torch.Tensor, np.ndarray]:
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
    R_est (np.ndarray): estimated rotation matrix
        If a guess is provided, the final rotation matrix is R_est @ R_guess
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

    obj_vol = (
        threshold_submission(volumes.clone(), threshold_percentile)
        .mean(0)
        .numpy()
        .astype(np.float32)
    )

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

    return volumes, R_est @ R_guess
