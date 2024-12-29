import os
import argparse
import torch
import ot
import numpy as np
from dask import delayed, compute
from dask.distributed import Client
from dask.diagnostics import ProgressBar
from dask_hpc_runner import SlurmRunner

from cryo_challenge._preprocessing.fourier_utils import downsample_volume

precision = 32
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
    marginal = normalize_mass_to_one(volume[idx_above_thresh].flatten())
    pairwise_distance = (
        make_sparse_cost(idx_above_thresh, dtype=torch_dtype)
        .numpy()
        .astype(numpy_dtype)
    )
    pairwise_distance = (pairwise_distance / pairwise_distance.max()) ** exponent
    return marginal, pairwise_distance


def gw_distance_wrapper(
    gw_distance_function,
    marginal_i,
    marginal_j,
    pairwise_distance_i,
    pairwise_distance_j,
    i,
    j,
    **kwargs,
):
    gw_dist, results_dict = gw_distance(
        gw_distance_function,
        marginal_i,
        marginal_j,
        pairwise_distance_i,
        pairwise_distance_j,
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
    marginal_i,
    marginal_j,
    pairwise_distance_i,
    pairwise_distance_j,
    **kwargs,
):
    gw_dist, results_dict = gw_distance_function(
        pairwise_distance_i,
        pairwise_distance_j,
        marginal_i,
        marginal_j,
        log=True,
        **kwargs,
    )
    return gw_dist, results_dict


def get_distance_matrix_dask(
    marginals_i,
    marginals_j,
    pairwise_distances_i,
    pairwise_distances_j,
    distance_function,
    gw_distance_function,
    scheduler,
    **gw_kwargs,
):
    n_i = len(marginals_i)
    n_j = len(marginals_j)
    distance_matrix = np.zeros((n_i, n_j), dtype=numpy_dtype)

    # Create a list to hold the delayed computations
    tasks = []
    symmetric_volumes = True
    for i in range(n_i):
        if symmetric_volumes:
            lower_bound_j = 1 + i
        else:
            lower_bound_j = 0

        for j in range(lower_bound_j, n_j):
            # Use dask.delayed to delay the computation
            compute_task = delayed(distance_function)(
                gw_distance_function,
                marginals_i[i],
                marginals_j[j],
                pairwise_distances_i[i],
                pairwise_distances_j[j],
                i,
                j,
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
            [task[idx_of_compute_task][idx_of_return] for task in tasks],
            scheduler=scheduler,
        )
    # Fill in the distance matrix with the results
    for (i, j, _), result in zip(tasks, results[0]):
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
    parser.add_argument(
        "--scheduler", type=str, default=None, help="Dask scheduler to use"
    )
    return parser.parse_args()


def main(args):
    # client = Client(local_directory="/tmp")

    job_id = os.environ["SLURM_JOB_ID"]

    # if True:
    #     with Client() as client:
    with SlurmRunner(
        scheduler_file=f"/mnt/home/gwoollard/ceph/repos/Cryo-EM-Heterogeneity-Challenge-1/src/cryo_challenge/_map_to_map/gromov_wasserstein/scheduler-{job_id}.json"
    ) as runner:
        # The runner object contains the scheduler address and can be passed directly to a client
        with Client(runner) as client:
            assert isinstance(
                client, type(client)
            )  # linter thinks client is unused, so need to do something with client as a workaround
            fname = "/mnt/home/smbp/ceph/smbpchallenge/round2/set2/processed_submissions/submission_23.pt"
            submission = torch.load(fname, weights_only=False)
            volumes = submission["volumes"].to(torch_dtype)[::40]
            volumes_i = volumes
            volumes_j = volumes

            gw_distance_function_key = "gromov_wasserstein2"
            n_downsample_pix = args.n_downsample_pix
            top_k = args.top_k
            exponent = args.exponent
            scheduler = args.scheduler

            marginals_i = np.empty((len(volumes_i), top_k))
            marginals_j = np.empty((len(volumes_j), top_k))
            pairwise_distances_i = np.empty((len(volumes), top_k, top_k))
            pairwise_distances_j = np.empty((len(volumes), top_k, top_k))

            for i in range(len(volumes_i)):
                volume_i, pairwise_distance_i = prepare_volume_and_distance(
                    volumes_i[i], top_k, n_downsample_pix, exponent
                )
                marginals_i[i] = volume_i
                pairwise_distances_i[i] = pairwise_distance_i
            for j in range(len(volumes_j)):
                volume_j, pairwise_distance_j = prepare_volume_and_distance(
                    volumes_j[j], top_k, n_downsample_pix, exponent
                )
                marginals_j[j] = volume_j
                pairwise_distances_j[j] = pairwise_distance_j

            get_distance_matrix_dask_gw = get_distance_matrix_dask(
                marginals_i=marginals_i,
                marginals_j=marginals_j,
                pairwise_distances_i=pairwise_distances_i,
                pairwise_distances_j=pairwise_distances_j,
                distance_function=gw_distance_wrapper,
                gw_distance_function=gw_distance_function_d[gw_distance_function_key],
                scheduler=scheduler,
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
