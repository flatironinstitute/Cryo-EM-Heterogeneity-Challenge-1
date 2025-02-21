import torch
import torch.nn.functional as F
import argparse
import time
import multiprocessing as mp
import logging

import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import SpecialOrthogonalGroup, Euclidean, Product
from pymanopt.optimizers import SteepestDescent

from cryo_challenge._preprocessing.fourier_utils import downsample_volume


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
    torch_dtype = torch.float32
    n_pix = len(volume_i)
    grid = prepare_grid(n_pix, torch_dtype)

    SO3 = SpecialOrthogonalGroup(3)
    E3 = Euclidean(3)
    SE3 = Product([SO3, E3])

    @pymanopt.function.pytorch(SE3)
    def loss(rotation, translation):
        """Objective function.

        Takes rotation then translation (in that order) because of the product manifold is SO(3) x E(3).
        """
        # Apply the rotation and tralsation to the volume
        interpolated_volume = interpolate_volume(volume_i, rotation, translation, grid)
        # Compute the L2 loss between the two functions
        return loss_l2(interpolated_volume, volume_j)

    # Define the problem
    problem = Problem(manifold=SE3, cost=loss)

    # Solve the problem with the custom solver
    optimizer = SteepestDescent(
        max_iterations=100,
    )

    initial_point = (
        torch.eye(3).to(torch_dtype).numpy(),
        torch.zeros(3).to(torch_dtype).numpy(),
    )
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
        "--downsample_box_size",
        type=int,
        default=32,
        help="Box size to downsample volumes to",
    )

    return parser.parse_args()


def run_all_by_all_naive_loop(args):
    fname = "/mnt/home/smbp/ceph/smbpchallenge/round2/set2/processed_submissions/submission_23.pt"
    submission = torch.load(fname, weights_only=False)

    torch_dtype = torch.float32
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
    loss_initial = torch.empty(args.n_i, args.n_j)
    loss_final = torch.empty(args.n_i, args.n_j)

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
            volume_i_aligned_to_j = interpolate_volume(
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


# Enable logging to debug errors
# logging.basicConfig(level=logging.ERROR)
logging.getLogger("pymanopt").setLevel(logging.ERROR)


# Ensure the multiprocessing context uses 'spawn'
mp.set_start_method("spawn", force=True)


def process_pair(idx_i, idx_j, volume_i, volume_j, box_size_ds):
    """Aligns two volumes and returns the results."""
    try:
        # volume_i_ds = downsample_volume(volume_i, box_size_ds)
        # volume_j_ds = downsample_volume(volume_j, box_size_ds)
        volume_i = volume_i.clone()
        volume_j = volume_j.clone()
        logging.info(f"Starting alignment for pair ({idx_i}, {idx_j})")
        logging.info(f"({idx_i}, {idx_j}) is shared memory? {volume_i.is_shared()}")
        result = align(volume_i, volume_j)
        rotation, translation = result.point

        # loss_init = loss_l2(volume_i, volume_j)
        # volume_i_aligned_to_j = interpolate_volume(volume_i, rotation, translation).reshape(*volume_i.shape)
        # loss_final = loss_l2(volume_i_aligned_to_j, volume_j)

        return idx_i, idx_j, rotation, translation

    except Exception as e:
        logging.error(f"Error in alignment for pair ({idx_i}, {idx_j}): {e}")
        return idx_i, idx_j, None, None, None, None


def mp_main(args):
    fname = "/mnt/home/smbp/ceph/smbpchallenge/round2/set2/processed_submissions/submission_23.pt"
    submission = torch.load(fname, weights_only=False)

    torch_dtype = torch.float32  # Use float32 to reduce memory usage
    volumes = submission["volumes"].to(torch_dtype)

    _volumes_i = volumes[: args.n_i]
    _volumes_j = volumes[: args.n_j]
    box_size_ds = args.downsample_box_size_ds

    volumes_i = torch.empty(
        (args.n_i, box_size_ds, box_size_ds, box_size_ds), dtype=torch_dtype
    )
    volumes_j = torch.empty(
        (args.n_j, box_size_ds, box_size_ds, box_size_ds), dtype=torch_dtype
    )
    for i, v in enumerate(_volumes_i):
        volumes_i[i] = downsample_volume(v, box_size_ds)
    for j, v in enumerate(_volumes_j):
        volumes_j[j] = downsample_volume(v, box_size_ds)

    box_size_ds = args.downsample_box_size_ds

    rotations = torch.empty((args.n_i, args.n_j, 3, 3))
    translations = torch.empty((args.n_i, args.n_j, 3))
    loss_initial = torch.empty((args.n_i, args.n_j, 1))
    loss_final = torch.empty((args.n_i, args.n_j, 1))

    # Prepare arguments for starmap
    tasks = []
    for idx_i, volume_i in enumerate(volumes_i):
        for idx_j, volume_j in enumerate(volumes_j):
            tasks.append(
                (idx_i, idx_j, volume_i.clone(), volume_j.clone(), box_size_ds)
            )

    # Use multiprocessing with starmap
    s = time.time()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(process_pair, tasks)
    e = time.time()
    logging.info(f"Time taken: {e-s:.2f}s")

    # Store results
    for idx_i, idx_j, rotation, translation in results:
        if rotation is not None:
            rotations[idx_i, idx_j] = torch.from_numpy(rotation)
            translations[idx_i, idx_j] = torch.from_numpy(translation)
            # loss_initial[idx_i, idx_j] = loss_init
            # loss_final[idx_i, idx_j] = loss_fin

    return {
        "rotation": rotations,
        "translation": translations,
        "loss_initial": loss_initial,
        "loss_final": loss_final,
    }


if __name__ == "__main__":
    args = parse_args()
    results = mp_main(args)
    torch.save(
        results,
        f"alignments_se3_ni{args.n_i}_nj{args.n_j}_ds{args.downsample_box_size}.pt",
    )
