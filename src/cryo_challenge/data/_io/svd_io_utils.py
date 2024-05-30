import torch
from typing import Tuple

from ..._preprocessing.fourier_utils import downsample_volume


def load_volumes(
    box_size_ds: float,
    submission_list: list,
    path_to_submissions: str,
    dtype=torch.float32,
) -> Tuple[torch.tensor, dict]:
    """
    Load the volumes and populations from the submissions specified in submission_list. Volumes are first downsampled, then normalized so that they sum to 1, and finally the mean volume is removed from each submission.

    Parameters
    ----------
    box_size_ds: float
        Size of the downsampled box.
    submission_list: list
        List of submission indices to load.
    path_to_submissions: str
        Path to the directory containing the submissions.
    dtype: torch.dtype
        Data type of the volumes.

    Returns
    -------
    volumes: torch.tensor
        Tensor of shape (n_volumes, n_x, n_y, n_z) containing the volumes.
    populations: dict
        Dictionary containing the populations of each submission.
    vols_per_submission: dict
        Dictionary containing the number of volumes per submission.

    Examples
    --------
    >>> box_size_ds = 64
    >>> submission_list = [0, 1, 2, 3, 4] # submission 5 is ignored
    >>> path_to_submissions = "/path/to/submissions" # under this folder submissions should be name submission_i.pt
    >>> volumes, populations = load_volumes(box_size_ds, submission_list, path_to_submissions)
    """  # noqa: E501

    metadata = {}
    volumes = torch.empty((0, box_size_ds, box_size_ds, box_size_ds), dtype=dtype)
    mean_volumes = torch.empty(
        (len(submission_list), box_size_ds, box_size_ds, box_size_ds), dtype=dtype
    )
    counter = 0

    for i, idx in enumerate(submission_list):
        submission = torch.load(f"{path_to_submissions}/submission_{idx}.pt")
        vols = submission["volumes"]
        pops = submission["populations"]

        vols_tmp = torch.empty(
            (vols.shape[0], box_size_ds, box_size_ds, box_size_ds), dtype=dtype
        )
        counter_start = counter
        for j in range(vols.shape[0]):
            vol_ds = downsample_volume(vols[j], box_size_ds)
            vols_tmp[j] = vol_ds / vol_ds.sum()
            counter += 1

        metadata[submission["id"]] = {
            "n_vols": vols.shape[0],
            "populations": pops / pops.sum(),
            "indices": (counter_start, counter),
        }

        mean_volumes[i] = vols_tmp.mean(dim=0)
        vols_tmp = vols_tmp - mean_volumes[i][None, :, :, :]
        volumes = torch.cat((volumes, vols_tmp), dim=0)

    return volumes, mean_volumes, metadata


def load_ref_vols(box_size_ds: int, path_to_volumes: str, dtype=torch.float32):
    """
    Load the reference volumes, downsample them, normalize them, and remove the mean volume.

    Parameters
    ----------
    box_size_ds: int
        Size of the downsampled box.
    path_to_volumes: str
        Path to the file containing the reference volumes. Must be in PyTorch format.
    dtype: torch.dtype
        Data type of the volumes.

    Returns
    -------
    volumes_ds: torch.tensor
        Tensor of shape (n_volumes, n_x, n_y, n_z) containing the downsampled, normalized, and mean-removed reference volumes.

    Examples
    --------
    >>> box_size_ds = 64
    >>> path_to_volumes = "/path/to/volumes.pt"
    >>> volumes_ds = load_ref_vols(box_size_ds, path_to_volumes)
    """  # noqa: E501
    try:
        volumes = torch.load(path_to_volumes)
    except (FileNotFoundError, EOFError):
        raise ValueError("Volumes not found or not in PyTorch format.")

    # Reshape volumes to correct size
    box_size = int(round((float(volumes.shape[-1]) ** (1. / 3.))))
    volumes = torch.reshape(volumes, (-1, box_size, box_size, box_size))

    volumes_ds = torch.empty(
        (volumes.shape[0], box_size_ds, box_size_ds, box_size_ds), dtype=dtype
    )
    for i, vol in enumerate(volumes):
        volumes_ds[i] = downsample_volume(vol, box_size_ds)
        volumes_ds[i] = volumes_ds[i] / volumes_ds[i].sum()

    mean_volume = volumes_ds.mean(dim=0)
    volumes_ds = volumes_ds - mean_volume[None, :, :, :]

    return volumes_ds, mean_volume
