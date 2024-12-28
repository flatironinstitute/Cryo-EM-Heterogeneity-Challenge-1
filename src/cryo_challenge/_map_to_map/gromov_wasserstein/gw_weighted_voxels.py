import argparse
import torch
import ot
import numpy as np
from dask import delayed, compute
from dask.distributed import Client
from dask.diagnostics import ProgressBar

from cryo_challenge._preprocessing.fourier_utils import downsample_volume

precision = 64
if precision == 32:
    numpy_dtype = np.float32
    torch_dtype = torch.float32
elif precision == 64:
    numpy_dtype = np.float64
    torch_dtype = torch.float64
elif precision == 128:
    numpy_dtype = np.float128
    torch_dtype = torch.float64


def return_top_k_voxel_idxs(volume, top_k):
    thresh = np.sort(volume.flatten())[-top_k - 1]
    idx_above_thresh = volume > thresh
    return idx_above_thresh


def make_sparse_cost(idx_above_thresh, dtype):
    n_downsample_pix = len(idx_above_thresh)
    one_dim = torch.arange(n_downsample_pix, dtype=dtype)
    coordinates = torch.meshgrid(one_dim, one_dim, one_dim)
    coordinates = torch.stack(coordinates, dim=-1)
    coordinates = coordinates.reshape(-1, 3)
    sparse_coordiantes = coordinates[idx_above_thresh.flatten()]
    pairwise_distances = torch.cdist(sparse_coordiantes, sparse_coordiantes)
    return pairwise_distances


def normalize_mass_to_one(p):
    p = p - p.min()
    return p / p.sum()


def prepare_volume_and_distance(volume, top_k, n_downsample_pix, exponent):
    volume = downsample_volume(volume, n_downsample_pix).numpy().astype(numpy_dtype)
    idx_above_thresh = return_top_k_voxel_idxs(volume, top_k)
    volume = normalize_mass_to_one(volume[idx_above_thresh].flatten())
    pairwise_distances = (
        make_sparse_cost(idx_above_thresh, dtype=torch_dtype)
        .numpy()
        .astype(numpy_dtype)
    )
    pairwise_distances = (pairwise_distances / pairwise_distances.max()) ** exponent
    return volume, pairwise_distances


def gw_distance_wrapper(
    gw_distance_function,
    volumes_i,
    volumes_j,
    i,
    j,
    top_k,
    n_downsample_pix,
    exponent,
    **kwargs,
):
    volume_i, pairwise_distances_i = prepare_volume_and_distance(
        volumes_i[i], top_k, n_downsample_pix, exponent
    )
    volume_j, pairwise_distances_j = prepare_volume_and_distance(
        volumes_j[j], top_k, n_downsample_pix, exponent
    )
    gw_dist, results_dict = gw_distance(
        gw_distance_function,
        volume_i,
        volume_j,
        pairwise_distances_i,
        pairwise_distances_j,
        **kwargs,
    )

    ij_results = {
        "i": i,
        "j": j,
        "results_dict": results_dict,
    }
    return gw_dist, ij_results


gw_distance_function_d = {
    "entropic_gromov_wasserstein2": ot.gromov.entropic_gromov_wasserstein2,
    "gromov_wasserstein2": ot.gromov.gromov_wasserstein2,
}


def gw_distance(
    gw_distance_function,
    volume_i,
    volume_j,
    pairwise_distances_i,
    pairwise_distances_j,
    **kwargs,
):
    gw_dist, results_dict = gw_distance_function(
        pairwise_distances_i,
        pairwise_distances_j,
        volume_i,
        volume_j,
        log=True,
        **kwargs,
    )
    return gw_dist, results_dict


def get_distance_matrix_dask(
    volumes_i,
    volumes_j,
    distance_function,
    gw_distance_function,
    top_k,
    n_downsample_pix,
    exponent,
    **gw_kwargs,
):
    n_vols_i = len(volumes_i)
    n_vols_j = len(volumes_j)
    distance_matrix = np.zeros((n_vols_i, n_vols_j), dtype=numpy_dtype)

    # Create a list to hold the delayed computations
    tasks = []
    symmetric_volumes = True
    for i in range(n_vols_i):
        if symmetric_volumes:
            lower_bound_j = 1 + i
        else:
            lower_bound_j = 0

        for j in range(lower_bound_j, n_vols_j):
            # Use dask.delayed to delay the computation
            compute_task = delayed(distance_function)(
                gw_distance_function,
                volumes_i,
                volumes_j,
                i,
                j,
                top_k,
                n_downsample_pix,
                exponent,
                loss_fun=gw_kwargs["loss_fun"],
                tol_abs=gw_kwargs["tol_abs"],
                tol_rel=gw_kwargs["tol_rel"],
                symmetric=gw_kwargs["symmetric"],
                max_iter=gw_kwargs["max_iter"],
                verbose=gw_kwargs["verbose"],
            )
            tasks.append((i, j, compute_task))

    # Compute all tasks in parallel using the Dask client
    with ProgressBar():
        idx_of_return = 0
        idx_of_compute_task = 2
        results = compute(
            *[task[idx_of_compute_task][idx_of_return] for task in tasks],
        )

    # Fill in the distance matrix with the results
    for (i, j, _), result in zip(tasks, results):
        distance_matrix[i, j] = result
        if symmetric_volumes:
            distance_matrix[j, i] = distance_matrix[i, j]

    return distance_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--n_downsample_pix", type=int, default=20, help="Number of downsample pixels"
    )
    parser.add_argument("--top_k", type=int, default=500, help="Top K value")
    parser.add_argument(
        "--exponent",
        type=float,
        default=1.0,
        help="Exponent on distance matrices of marginals",
    )
    return parser.parse_args()


def main(args):
    fname = "/mnt/home/smbp/ceph/smbpchallenge/round2/set2/processed_submissions/submission_23.pt"
    submission = torch.load(fname)[::20]
    volumes = submission["volumes"].to(torch_dtype)

    client = Client(local_directory="/tmp")
    assert isinstance(
        client, type(client)
    )  # linter thinks client is unused, so need to do something with client as a workaround

    gw_distance_function_key = "gromov_wasserstein2"
    n_downsample_pix = args.n_downsample_pix
    top_k = args.top_k
    exponent = args.exponent

    get_distance_matrix_dask_gw = get_distance_matrix_dask(
        volumes_i=volumes,
        volumes_j=volumes,
        distance_function=gw_distance_wrapper,
        gw_distance_function=gw_distance_function_d[gw_distance_function_key],
        top_k=top_k,
        n_downsample_pix=n_downsample_pix,
        exponent=exponent,
        tol_abs=1e-14,
        tol_rel=1e-14,
        max_iter=10000,
        symmetric=True,
        verbose=False,
        loss_fun="square_loss",
    )

    np.save(
        f"/mnt/home/gwoollard/ceph/repos/Cryo-EM-Heterogeneity-Challenge-1/src/cryo_challenge/_map_to_map/gromov_wasserstein/gw_weighted_voxel_topk{top_k}_ds{n_downsample_pix}_float{precision}_exponent{exponent}_23.npy",
        get_distance_matrix_dask_gw,
    )
    return get_distance_matrix_dask_gw


if __name__ == "__main__":
    args = parse_args()
    main(args)
