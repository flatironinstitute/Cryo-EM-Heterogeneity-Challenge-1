import torch
from typing import Tuple

from ..._preprocessing.fourier_utils import downsample_volume
from ..._preprocessing.normalize import compute_power_spectrum, normalize_power_spectrum


def _remove_mean_volumes_sub(volumes, metadata):
    box_size = volumes.shape[-1]
    n_subs = len(list(metadata.keys()))
    mean_volumes = torch.zeros((n_subs, box_size, box_size, box_size))

    for i, key in enumerate(metadata.keys()):
        indices = metadata[key]["indices"]

        mean_volumes[i] = torch.mean(volumes[indices[0] : indices[1]], dim=0)
        volumes[indices[0] : indices[1]] = (
            volumes[indices[0] : indices[1]] - mean_volumes[i][None, ...]
        )

    return volumes, mean_volumes


def remove_mean_volumes(volumes, metadata=None):
    volumes = volumes.clone()
    if metadata is None:
        mean_volumes = torch.mean(volumes, dim=0)
        volumes = volumes - mean_volumes[None, ...]

    else:
        volumes, mean_volumes = _remove_mean_volumes_sub(volumes, metadata)

    return volumes, mean_volumes


def normalize_power_spectrum_sub(volumes, metadata, ref_vol_key, ref_vol_index):
    volumes = volumes.clone()
    idx_ref_vol = metadata[ref_vol_key]["indices"][0] + ref_vol_index
    ref_power_spectrum = compute_power_spectrum(volumes[idx_ref_vol])

    for key in metadata.keys():
        indices = metadata[key]["indices"]
        volumes[indices[0] : indices[1]] = normalize_power_spectrum(
            volumes[indices[0] : indices[1]], ref_power_spectrum
        )

    return volumes


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
    metadata: dict
        Dictionary containing the metadata for each submission.
        The keys are the id (ice cream name) of each submission.
        The values are dictionaries containing the number of volumes, the populations, and the indices of the volumes in the volumes tensor.

    Examples
    --------
    >>> box_size_ds = 64
    >>> submission_list = [0, 1, 2, 3, 4] # submission 5 is ignored
    >>> path_to_submissions = "/path/to/submissions" # under this folder submissions should be name submission_i.pt
    >>> volumes, populations = load_volumes(box_size_ds, submission_list, path_to_submissions)
    """  # noqa: E501

    metadata = {}
    volumes = torch.empty((0, box_size_ds, box_size_ds, box_size_ds), dtype=dtype)

    counter = 0

    for idx in submission_list:
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

        volumes = torch.cat((volumes, vols_tmp), dim=0)

    return volumes, metadata


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
    if volumes.dim() == 2:
        box_size = int(round((float(volumes.shape[-1]) ** (1.0 / 3.0))))
        volumes = torch.reshape(volumes, (-1, box_size, box_size, box_size))
    elif volumes.dim() == 4:
        pass
    else:
        raise ValueError(
            f"The shape of the volumes stored in {path_to_volumes} have the unexpected shape "
            f"{torch.shape}. Please, review the file and regenerate it so that volumes stored hasve the "
            f"shape (num_vols, box_size ** 3) or (num_vols, box_size, box_size, box_size)."
        )

    volumes_ds = torch.empty(
        (volumes.shape[0], box_size_ds, box_size_ds, box_size_ds), dtype=dtype
    )
    for i, vol in enumerate(volumes):
        volumes_ds[i] = downsample_volume(vol, box_size_ds)
        volumes_ds[i] = volumes_ds[i] / volumes_ds[i].sum()

    volumes_ds = volumes_ds

    return volumes_ds
