import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R


@torch.no_grad()
def interpolate_volume(volume, rotation, translation, grid):
    """

    Notes:
    -----
    translation is normalized coordinates, since grid is from [-1,+1]. Invariant to n_pix (from downsampling volume)
    """
    n_pix = len(volume)
    grid = grid @ rotation.T + translation
    # Interpolate the 3D array at the grid points
    interpolated_volume = F.grid_sample(
        volume.reshape(1, 1, n_pix, n_pix, n_pix),
        grid[..., [2, 1, 0]],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(n_pix, n_pix, n_pix)
    return interpolated_volume


@torch.no_grad()
def prepare_grid(n_pix, torch_dtype):
    x = y = z = torch.linspace(-1, 1, n_pix).to(torch_dtype)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    grid = torch.stack([xx, yy, zz], dim=-1)  # Shape: (D, H, W, 3)
    # Reshape grid to match the expected input shape for grid_sample
    grid = grid.unsqueeze(0)  # Add batch dimension, shape: (1, D, H, W, 3)
    return grid


@torch.no_grad()
def interpolate_and_project(volume, rotation, translation, grid):
    posed_volume = interpolate_volume(volume, rotation, translation, grid)
    pixel_strip = posed_volume.mean(dim=(-1, -2))  # Average over two axes
    return pixel_strip


@torch.no_grad()
def wasserstein_1d_torch_pairwise(a, b, p):
    """
    Compute all pairwise 1D Wasserstein-2^2 distances between two batches of histograms.
    Assumes spatial bins are equally spaced.

    Args:
        a: (N1, n) tensor of histograms (each row sums to 1)
        b: (N2, n) tensor of histograms
        eps: numerical stability value for normalization

    Returns:
        w2_matrix: (N1, N2) tensor where w2_matrix[i, j] = W2^2(a[i], b[j])

    Notes:
    Eq 2 in https://openreview.net/forum?id=yPBtJ4JPwi
    """
    eps = 1e-8
    # Normalize histograms
    a = a / (a.sum(dim=1, keepdim=True) + eps)  # (N1, n)
    b = b / (b.sum(dim=1, keepdim=True) + eps)  # (N2, n)

    # Compute CDFs
    cdf_a = torch.cumsum(a, dim=1)  # (N1, n)
    cdf_b = torch.cumsum(b, dim=1)  # (N2, n)

    # Compute pairwise squared L2 distances between CDFs
    # Want (N1, N2): pairwise distances
    # Use broadcasting: (N1, 1, n) - (1, N2, n) → (N1, N2, n)
    diff = cdf_a[:, None, :] - cdf_b[None, :, :]  # (N1, N2, n)
    if p == 2:
        # For p=2, compute squared L2 distance
        w = (diff**2).sum(dim=2)
    elif p == 1:
        # For p=1, compute L1 distance
        w = diff.abs().sum(dim=2)
    else:
        raise ValueError(f"Unsupported p value: {p}. Only p=1 or p=2 are supported.")

    return w


@torch.no_grad()
def get_distance_matrix_real_space_sliced_wasserstein(volumes_gt, volumes_sub, config):
    dev = config["dev"]
    grid = prepare_grid(config["downsample_box_size"], volumes_gt.dtype).to(dev)
    n_rotations = config["n_rotations"]
    n_vols_gt = volumes_gt.shape[0]
    n_vols_sub = volumes_sub.shape[0]
    map_to_map_distance_matrix = torch.zeros(
        (n_vols_gt, n_vols_sub), dtype=volumes_gt.dtype
    ).to(dev)
    for _ in range(n_rotations):
        if _ % 100 == 0:
            print("rotation number", _)
        rotation = torch.from_numpy(R.random().as_matrix()).to(volumes_gt.dtype).to(dev)
        translation = torch.zeros(3, dtype=volumes_gt.dtype).to(dev)
        pixel_strips_gt = torch.vmap(
            interpolate_and_project,
            in_dims=(0, None, None, None),
            chunk_size=config["vmap_chunk_size_gt"],
        )(volumes_gt, rotation, translation, grid)  # shape (n_vols, box_size_ds,)
        pixel_strips_sub = torch.vmap(
            interpolate_and_project,
            in_dims=(0, None, None, None),
            chunk_size=config["vmap_chunk_size_submission"],
        )(volumes_sub, rotation, translation, grid)
        sliced_w_matrix = wasserstein_1d_torch_pairwise(
            pixel_strips_gt, pixel_strips_sub, config["wasserstein_p"]
        )
        map_to_map_distance_matrix += sliced_w_matrix
    map_to_map_distance_matrix /= n_rotations
    return map_to_map_distance_matrix
