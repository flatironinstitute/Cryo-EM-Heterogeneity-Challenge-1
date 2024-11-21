import torch
import ot
import numpy as np
from dask import delayed, compute
from dask.distributed import Client
from dask.diagnostics import ProgressBar

from cryo_challenge._preprocessing.fourier_utils import downsample_volume


def return_top_k_voxel_idxs(volume, top_k):
    thresh = np.sort(volume.flatten())[-top_k - 1]
    idx_above_thresh = volume > thresh
    return idx_above_thresh


def make_sparse_cost(idx_above_thresh):
    n_downsample_pix = len(idx_above_thresh)
    coordinates = torch.meshgrid(
        torch.arange(n_downsample_pix),
        torch.arange(n_downsample_pix),
        torch.arange(n_downsample_pix),
    )
    coordinates = torch.stack(coordinates, dim=-1)
    coordinates = coordinates.reshape(-1, 3).float()
    sparse_coordiantes = coordinates[idx_above_thresh.flatten()]
    pairwise_distances = torch.cdist(sparse_coordiantes, sparse_coordiantes)
    return pairwise_distances


def normalize_mass_to_one(p):
    p = p - p.min()
    return p / p.sum()


def prepare_volume_and_distance(volume, top_k, n_downsample_pix):
    volume = downsample_volume(volume, n_downsample_pix)
    idx_above_thresh = return_top_k_voxel_idxs(volume, top_k)
    volume = normalize_mass_to_one(volume[idx_above_thresh].numpy().flatten())
    pairwise_distances = make_sparse_cost(idx_above_thresh)
    pairwise_distances = pairwise_distances.numpy() / pairwise_distances.numpy().max()
    return volume, pairwise_distances


def gw_distance_wrapper(
    gw_distance_function, volumes, i, j, top_k, n_downsample_pix, **kwargs
):
    volume_i, pairwise_distances_i = prepare_volume_and_distance(
        volumes[i], top_k, n_downsample_pix
    )
    volume_j, pairwise_distances_j = prepare_volume_and_distance(
        volumes[j], top_k, n_downsample_pix
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
    volumes,
    distance_function,
    gw_distance_function,
    top_k,
    n_downsample_pix,
    **gw_kwargs,
):
    n_vols = len(volumes)
    distance_matrix = np.zeros((n_vols, n_vols))

    # Create a list to hold the delayed computations
    tasks = []

    for i in range(n_vols):
        for j in range(i + 1, n_vols):
            # Use dask.delayed to delay the computation
            compute_task = delayed(distance_function)(
                gw_distance_function,
                volumes,
                i,
                j,
                top_k,
                n_downsample_pix,
                loss_fun=gw_kwargs["loss_fun"],
                tol=gw_kwargs["tol"],
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
            scheduler="single-threaded",
        )

    # Fill in the distance matrix with the results
    for (i, j, _), result in zip(tasks, results):
        distance_matrix[i, j] = distance_matrix[j, i] = result

    return distance_matrix


def main():
    fname = "/mnt/home/smbp/ceph/smbpchallenge/round2/set2/processed_submissions/submission_23.pt"
    submission = torch.load(fname)
    volumes = submission["volumes"]

    client = Client(local_directory="/tmp")

    n_interval = 20
    gw_distance_function_key = "gromov_wasserstein2"
    get_distance_matrix_dask_gw = get_distance_matrix_dask(
        volumes[::n_interval],
        distance_function=gw_distance_wrapper,
        gw_distance_function=gw_distance_function_d[gw_distance_function_key],
        top_k=100,
        n_downsample_pix=20,
        tol=1e-11,
        max_iter=10000,
        symmetric=True,
        verbose=False,
        loss_fun="square_loss",
    )

    np.save(
        "/mnt/home/gwoollard/ceph/repos/Cryo-EM-Heterogeneity-Challenge-1/tmp/gw_weighted_voxel_23.npy",
        get_distance_matrix_dask_gw,
    )
    del client
    return get_distance_matrix_dask_gw  #


if __name__ == "__main__":
    main()
