import os
import torch
from cryo_challenge.preprocessing import downsample_submission
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
import matplotlib.pyplot as plt


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
def get_distance_matrix_real_space_slicing(volumes_gt, volumes_sub, config):
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
        # sliced_wasserstein pixel strip example
        sliced_w_matrix = wasserstein_1d_torch_pairwise(
            pixel_strips_gt, pixel_strips_sub, config["wasserstein_p"]
        )
        map_to_map_distance_matrix += sliced_w_matrix
    map_to_map_distance_matrix /= n_rotations
    return map_to_map_distance_matrix


@dataclass
class Config:
    dev: str = "cuda" if torch.cuda.is_available() else "cpu"
    downsample_box_size: int = 10
    vmap_chunk_size_gt: int = 80
    vmap_chunk_size_submission: int = 80
    n_rotations: int = 1000

    def __getitem__(self, key):
        return getattr(self, key)


if __name__ == "__main__":
    config = Config()
    # n_pix = 64
    # n_vols_gt = 4000
    # n_vols_sub = 80
    # volumes_gt = torch.empty(n_vols_gt, n_pix, n_pix, n_pix)  # Example submission volumes
    # volumes_sub = torch.empty(n_vols_sub, n_pix, n_pix, n_pix)  # Example submission volumes
    fname = "/mnt/home/smbp/ceph/smbpchallenge/preprocessing_submissions/mock_submissions/submission_mint_chocolate_chip_80.pt"
    volumes_gt = torch.load(fname, weights_only=False)["volumes"]
    volumes_sub = torch.load(fname, weights_only=False)["volumes"]
    downsample_box_size = config.downsample_box_size
    dev = config.dev
    downsampled_volumes_gt = downsample_submission(volumes_gt, downsample_box_size).to(
        dev
    )
    downsampled_volumes_sub = downsample_submission(
        volumes_sub, downsample_box_size
    ).to(dev)

    n_rotations_list = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
    n_trials = 30
    results = torch.zeros(
        len(n_rotations_list),
        n_trials,
        len(downsampled_volumes_gt),
        len(downsampled_volumes_sub),
    )
    overall_std = []
    wdir = "/mnt/home/gwoollard/ceph/repos/Cryo-EM-Heterogeneity-Challenge-1/tmp/"
    for idx_n_rotations, n_rotations in enumerate(n_rotations_list):
        config.n_rotations = n_rotations
        for idx_trial in range(n_trials):
            # Get the map-to-map distance matrix
            map_to_map_distance_matrix = get_distance_matrix_real_space_slicing(
                downsampled_volumes_gt, downsampled_volumes_sub, config
            )
            # ax = plt.imshow(map_to_map_distance_matrix.cpu().numpy(), cmap='gray')
            # plt.colorbar(ax)
            # plt.savefig(f'/mnt/home/gwoollard/ceph/repos/Cryo-EM-Heterogeneity-Challenge-1/tmp/map_to_map_distance_matrix_nrotations{n_rotations}.png')
            # plt.show()
            # plt.close()
            results[idx_n_rotations, idx_trial] = map_to_map_distance_matrix

        mean_results = results[idx_n_rotations].mean(dim=0)
        std_results = results[idx_n_rotations].std(dim=0)
        overall_std.append(std_results.mean().item())

        ax = plt.imshow(mean_results.cpu().numpy(), cmap="gray")
        plt.colorbar(ax)
        plt.savefig(
            os.path.join(
                wdir, f"map_to_map_distance_matrix_nrotations{n_rotations}_mean.png"
            )
        )
        plt.show()
        plt.close()

        ax = plt.imshow(std_results.cpu().numpy(), cmap="gray")
        plt.colorbar(ax)
        plt.savefig(
            os.path.join(
                wdir, f"map_to_map_distance_matrix_nrotations{n_rotations}_std.png"
            )
        )
        plt.show()
        plt.close()

        ax = plt.hist(std_results.cpu().numpy().flatten(), alpha=0.5, label="Std")
        plt.xlabel("Standard Deviation")
        plt.ylabel("Frequency")
        plt.title(f"Standard Deviation Histogram for n_rotations={n_rotations}")
        plt.legend()
        plt.savefig(
            os.path.join(
                wdir, f"map_to_map_distance_matrix_nrotations{n_rotations}_std_hist.png"
            )
        )
        plt.show()
        plt.close()

    ax = plt.plot(n_rotations_list, overall_std)
    plt.xlabel("Number of Rotations")
    plt.ylabel("Overall Standard Deviation")
    plt.title("Overall Standard Deviation vs Number of Rotations")
    plt.savefig(os.path.join(wdir, "overall_std_vs_n_rotations.png"))
    plt.show()
    plt.close()

    torch.save(results, os.path.join(wdir, "map_to_map_distance_matrix_results.pt"))
