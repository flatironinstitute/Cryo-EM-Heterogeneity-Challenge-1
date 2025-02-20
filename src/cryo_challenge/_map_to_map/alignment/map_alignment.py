import torch
import torch.nn.functional as F
import numpy as np
import argparse

import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import SpecialOrthogonalGroup, Euclidean, Product
from pymanopt.optimizers import SteepestDescent

from cryo_challenge._preprocessing.fourier_utils import downsample_volume


def voxelized_f1(volume, rotaiton, translation, grid):
    """

    Notes:
    -----
    translation is normalized coordinates, since grid is from [-1,+1]
    """
    n_pix = len(volume)
    grid = grid @ rotaiton.T + translation
    # Interpolate the 3D array at the grid points
    interpolated_volume = F.grid_sample(
        volume.reshape(1, 1, n_pix, n_pix, n_pix),
        grid[..., [2, 1, 0]],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(n_pix, n_pix, n_pix)
    return interpolated_volume


def loss_l2(volume_i, volume_j):
    return torch.linalg.norm(volume_i - volume_j) ** 2


def prepare_grid(n_pix, torch_dtype):
    x = y = z = torch.linspace(-1, 1, n_pix).to(torch_dtype)
    xx, yy, zz = torch.meshgrid(x, y, z)
    grid = torch.stack([xx, yy, zz], dim=-1)  # Shape: (D, H, W, 3)
    # Reshape grid to match the expected input shape for grid_sample
    grid = grid.unsqueeze(0)  # Add batch dimension, shape: (1, D, H, W, 3)
    return grid


def align(volume_i, volume_j):
    assert volume_i.shape == volume_j.shape
    assert volume_i.ndim == 3

    # Generate grid points
    torch_dtype = torch.float64
    n_pix = len(volume_i)
    grid = prepare_grid(n_pix, torch_dtype)

    SO3 = SpecialOrthogonalGroup(3)
    E3 = Euclidean(3)
    SE3 = Product([SO3, E3])

    @pymanopt.function.pytorch(SE3)
    def loss(rotation, translation):
        # Apply the rotation R to the volume
        interpolated_volume = voxelized_f1(volume_i, rotation, translation, grid)
        # Compute the L2 loss between the two functions
        return loss_l2(interpolated_volume, volume_j)

    # Define the problem
    problem = Problem(manifold=SE3, cost=loss)

    # Solve the problem with the custom solver
    optimizer = SteepestDescent(
        max_iterations=100,
    )

    initial_point = np.eye(3), np.zeros(3)
    result = optimizer.run(problem, initial_point=initial_point)

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--n_i", type=int, default=80, help="Number of volumes in set i"
    )
    parser.add_argument(
        "--n_j", type=int, default=80, help="Number of volumes in set j"
    )
    parser.add_argument(
        "--downsample_box_size_ds",
        type=int,
        default=32,
        help="Box size to downsample volumes to",
    )

    return parser.parse_args()


def main(args):
    fname = "/mnt/home/smbp/ceph/smbpchallenge/round2/set2/processed_submissions/submission_23.pt"
    submission = torch.load(fname, weights_only=False)

    torch_dtype = torch.float64
    volumes = submission["volumes"].to(torch_dtype)
    volumes_i = volumes[: args.n_i]
    volumes_j = volumes[: args.n_j]
    box_size_ds = args.downsample_box_size_ds

    size_of_rotation_matrix = (3, 3)
    size_of_translation_vector = (3,)
    rotations = torch.empty(
        (
            args.n_i,
            args.n_j,
        )
        + size_of_rotation_matrix
    )
    translations = torch.empty(
        (
            args.n_i,
            args.n_j,
        )
        + size_of_translation_vector
    )
    loss_initial = torch.empty(args.n_i, args.n_j, 1)
    loss_final = torch.empty(args.n_i, args.n_j, 1)

    n_pix = len(volumes_i[0])
    grid = prepare_grid(n_pix, torch_dtype)

    for idx_i, volume_i in enumerate(volumes_i):
        volume_i_ds = downsample_volume(volume_i, box_size_ds)

        for idx_j, volume_j in enumerate(volumes_j):
            volume_j_ds = downsample_volume(volume_j, box_size_ds)
            result = align(volume_i_ds, volume_j_ds)
            rotation, translation = result.point
            rotations[idx_i, idx_j] = torch.from_numpy(rotation)
            translations[idx_i, idx_j] = torch.from_numpy(translation)
            volume_i_aligned_to_j = voxelized_f1(
                volume_i, rotation, translation, grid
            ).reshape(n_pix, n_pix, n_pix)
            loss_initial[idx_i, idx_j] = loss_l2(volume_i, volume_j)
            loss_final[idx_i, idx_j] = loss_l2(volume_i_aligned_to_j, volume_j)

    return {
        "rotation": rotation,
        "translation": translation,
        "loss_initial": loss_initial,
        "loss_final": loss_final,
    }


if __name__ == "__main__":
    args = parse_args()
    results = main(args)
    torch.save(
        results,
        f"alignments_ni{args.n_i}_nj{args.n_j}_ds{args.downsample_box_size_ds}.pt",
    )
