import os
import argparse
import torch
import ot
import numpy as np
from dask import delayed, compute
from dask.distributed import Client
from dask.diagnostics import ProgressBar
from dask_jobqueue.slurm import SLURMRunner

from cryo_challenge.preprocessing._downsampling import downsample_volume
from cryo_challenge.map_to_map.gromov_wasserstein.frank_wolfe import (
    frank_wolfe_emd,
    gw_objective_cost,
)

precision = 128
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
    flat_volume = volume.flatten()
    idx = torch.zeros(len(flat_volume), dtype=torch.bool)
    idx[torch.topk(flat_volume, top_k).indices] = True
    return idx.reshape(volume.shape)


def make_sparse_cost(idx_above_thresh, dtype):
    box_size = len(idx_above_thresh)
    one_dim = torch.arange(box_size, dtype=dtype)
    coordinates = torch.meshgrid(one_dim, one_dim, one_dim, indexing="ij")
    coordinates = torch.stack(coordinates, dim=-1)
    coordinates = coordinates.reshape(-1, 3)
    sparse_coordiantes = coordinates[idx_above_thresh.flatten()]
    pairwise_distances = torch.cdist(sparse_coordiantes, sparse_coordiantes)
    return pairwise_distances, sparse_coordiantes


def normalize_mass_to_one(p):
    """Normalize mass to one"""
    if p.min() < 0:  # if all positive, no need to shift
        p = p - p.min()
    return p / p.sum()


def prepare_volume_and_distance(
    volume, top_k, box_size, exponent, cost_scale_factor, normalize
):
    downsampled_volume = downsample_volume(volume, box_size).numpy().astype(numpy_dtype)
    idx_above_thresh = return_top_k_voxel_idxs(volume, top_k)
    marginal_volume = normalize_mass_to_one(volume[idx_above_thresh].flatten())
    pairwise_distance, sparse_coordiantes = [
        x.numpy().astype(numpy_dtype)
        for x in make_sparse_cost(idx_above_thresh, dtype=torch_dtype)
    ]

    if normalize:
        pairwise_distance /= pairwise_distance.max()
    pairwise_distance = (cost_scale_factor * pairwise_distance) ** exponent
    return downsampled_volume, marginal_volume, sparse_coordiantes, pairwise_distance


def gw_distance_wrapper_element_wise(
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


def gw_distance_wrapper_row_wise(
    gw_distance_function,
    marginal_i,
    marginals_j,
    pairwise_distance_i,
    pairwise_distances_j,
    i,
    **kwargs,
):
    ij_results = []
    gw_dists = []
    for j in range(len(marginals_j)):
        gw_dist, results_dict = gw_distance(
            gw_distance_function,
            marginal_i,
            marginals_j[j],
            pairwise_distance_i,
            pairwise_distances_j[j],
            **kwargs,
        )
        gw_dists.append(gw_dist)

        ij_results.append(
            {
                "i": i,
                "j": j,
                "results_dict": results_dict,
            }
        )
    return gw_dists, ij_results


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


def get_distance_matrix_dask_element_wise(
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


def get_distance_matrix_dask_row_wise(
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
    for i in range(n_i):
        # Use dask.delayed to delay the computation
        compute_task = delayed(distance_function)(
            gw_distance_function,
            marginals_i[i],
            marginals_j,
            pairwise_distances_i[i],
            pairwise_distances_j,
            i,
            loss_fun=gw_kwargs["loss_fun"],
            tol_abs=gw_kwargs["tol_abs"],
            tol_rel=gw_kwargs["tol_rel"],
            symmetric=gw_kwargs["symmetric"],
            max_iter=gw_kwargs["max_iter"],
            verbose=gw_kwargs["verbose"],
        )
        tasks.append((i, compute_task))

    # Compute all tasks in parallel using the Dask client
    with ProgressBar():
        idx_of_return = 0
        idx_of_compute_task = 1
        results = compute(
            [task[idx_of_compute_task][idx_of_return] for task in tasks],
            scheduler=scheduler,
        )

    # Fill in the distance matrix with the results
    for (i, _), result in zip(tasks, results[0]):
        distance_matrix[i] = result

    return distance_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--n_i", type=int, default=80, help="Number of volumes in set i"
    )
    parser.add_argument(
        "--n_j", type=int, default=80, help="Number of volumes in set j"
    )
    parser.add_argument(
        "--box_size", type=int, default=20, help="Number of downsample pixels"
    )
    parser.add_argument("--top_k", type=int, default=500, help="Top K value")
    parser.add_argument(
        "--exponent",
        type=float,
        default=1.0,
        help="Exponent on distance matrices of marginals",
    )
    parser.add_argument(
        "--cost_scale_factor",
        type=float,
        default=1.0,
        help="Scaling of distance matrices of marginals (before exponentiation)",
    )
    parser.add_argument(
        "--scheduler", type=str, default=None, help="Dask scheduler to use"
    )
    parser.add_argument("--element_wise", action="store_true", default=False)
    parser.add_argument("--skip_normalize", action="store_false", default=False)
    parser.add_argument("--slurm", action="store_true", default=False)
    parser.add_argument(
        "--local_directory", type=str, default="/tmp", help="Local directory for dask"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/mnt/home/gwoollard/ceph/repos/Cryo-EM-Heterogeneity-Challenge-1/src/cryo_challenge/map_to_map/gromov_wasserstein/output/",
        help="Output directory for npy file",
    )

    try:
        job_id = os.environ["SLURM_JOB_ID"]
    except KeyError:
        job_id = "local" + str(np.random.randint(0, 1000))

    parser.add_argument(
        "--scheduler_file",
        type=str,
        default=f"/mnt/home/gwoollard/ceph/repos/Cryo-EM-Heterogeneity-Challenge-1/src/cryo_challenge/_map_to_map/gromov_wasserstein/scheduler-{job_id}.json",
        help="Dask scheduler file (output file name)",
    )
    parser.add_argument(
        "--fname",
        type=str,
        help="Path to submission file",
    )
    parser.add_argument("--frank_wolfe", action="store_true", default=False)

    return parser.parse_args()


def get_distance_matrix_dask_gw(
    marginals_i,
    marginals_j,
    pairwise_distances_i,
    pairwise_distances_j,
    scheduler,
    element_wise,
    gw_distance_function_key,
):
    # TODO: dask.array.from_array for marginals and pairwise_distances

    if element_wise:
        print("element wise")
        distance_matrix_dask_gw = get_distance_matrix_dask_element_wise(
            marginals_i=marginals_i,
            marginals_j=marginals_j,
            pairwise_distances_i=pairwise_distances_i,
            pairwise_distances_j=pairwise_distances_j,
            distance_function=gw_distance_wrapper_element_wise,
            gw_distance_function=gw_distance_function_d[gw_distance_function_key],
            scheduler=scheduler,
            tol_abs=1e-14,
            tol_rel=1e-14,
            max_iter=10000,
            symmetric=True,
            verbose=False,
            loss_fun="square_loss",
        )
    else:
        print("row wise")
        distance_matrix_dask_gw = get_distance_matrix_dask_row_wise(
            marginals_i=marginals_i,
            marginals_j=marginals_j,
            pairwise_distances_i=pairwise_distances_i,
            pairwise_distances_j=pairwise_distances_j,
            distance_function=gw_distance_wrapper_row_wise,
            gw_distance_function=gw_distance_function_d[gw_distance_function_key],
            scheduler=scheduler,
            tol_abs=1e-18,
            tol_rel=1e-18,
            max_iter=10000,
            symmetric=True,
            verbose=False,
            loss_fun="square_loss",
        )
    return distance_matrix_dask_gw


def setup_volume_and_distance(
    volumes_i,
    volumes_j,
    box_size,
    top_k,
    exponent,
    cost_scale_factor,
    normalize,
):
    downsampled_volumes_i = np.empty((len(volumes_i), box_size, box_size, box_size))
    downsampled_volumes_j = np.empty((len(volumes_j), box_size, box_size, box_size))
    marginals_i = np.empty((len(volumes_i), top_k))
    marginals_j = np.empty((len(volumes_j), top_k))
    sparse_coordinates_sets_i = np.empty((len(volumes_i), top_k, 3))
    sparse_coordinates_sets_j = np.empty((len(volumes_j), top_k, 3))
    pairwise_distances_i = np.empty((len(volumes_i), top_k, top_k))
    pairwise_distances_j = np.empty((len(volumes_j), top_k, top_k))

    for i in range(len(volumes_i)):
        (
            downsampled_volume_i,
            marginal_volume_i,
            sparse_coordinates_i,
            pairwise_distance_i,
        ) = prepare_volume_and_distance(
            volumes_i[i],
            top_k,
            box_size,
            exponent,
            cost_scale_factor,
            normalize,
        )
        downsampled_volumes_i[i] = downsampled_volume_i
        marginals_i[i] = marginal_volume_i
        sparse_coordinates_sets_i[i] = sparse_coordinates_i
        pairwise_distances_i[i] = pairwise_distance_i
    for j in range(len(volumes_j)):
        (
            downsampled_volume_j,
            marginal_volume_j,
            sparse_coordinates_j,
            pairwise_distance_j,
        ) = prepare_volume_and_distance(
            volumes_j[j],
            top_k,
            box_size,
            exponent,
            cost_scale_factor,
            normalize,
        )
        downsampled_volumes_j[j] = downsampled_volume_j
        marginals_j[j] = marginal_volume_j
        sparse_coordinates_sets_j[j] = sparse_coordinates_j
        pairwise_distances_j[j] = pairwise_distance_j

    return (
        downsampled_volumes_i,
        downsampled_volumes_j,
        marginals_i,
        marginals_j,
        sparse_coordinates_sets_i,
        sparse_coordinates_sets_j,
        pairwise_distances_i,
        pairwise_distances_j,
        volumes_i,
        volumes_j,
    )


def get_distance_matrix_gw_via_fw(
    marginals_i,
    marginals_j,
    sparse_coordinates_sets_i,
    sparse_coordinates_sets_j,
    pairwise_distances_i,
    pairwise_distances_j,
):
    n_i = len(marginals_i)
    n_j = len(marginals_j)
    distance_matrix_gw = torch.empty((n_i, n_j), dtype=torch_dtype)
    for i in range(n_i):
        for j in range(n_j):
            # Compute the cost matrix
            #
            Gamma0 = np.outer(marginals_i[i], marginals_j[j])

            # Compute the optimal transport plan using Frank-Wolfe algorithm
            Gamma, log = frank_wolfe_emd(
                sparse_coordinates_sets_i[i],
                sparse_coordinates_sets_j[j],
                Gamma0,
                marginals_i[i],
                marginals_j[j],
                num_iters=30,
                Gamma_atol=1e-6,
            )
            Cx = pairwise_distances_i[i] ** 2
            Cy = pairwise_distances_j[j] ** 2
            gw_frank_wolfe = gw_objective_cost(Cx, Cy, Gamma)
            distance_matrix_gw[i, j] = gw_frank_wolfe

    return distance_matrix_gw


def run_fw(args):
    n_i = args.n_i
    n_j = args.n_j
    box_size = args.box_size
    top_k = args.top_k
    exponent = args.exponent
    cost_scale_factor = args.cost_scale_factor
    normalize = not args.skip_normalize

    fname = args.fname  # e.g. /path/to/submission_23.pt
    submission = torch.load(fname, weights_only=False)
    volumes = submission["volumes"].to(torch_dtype)

    (
        _,
        _,
        marginals_i,
        marginals_j,
        sparse_coordinates_sets_i,
        sparse_coordinates_sets_j,
        pairwise_distances_i,
        pairwise_distances_j,
        volumes_i,
        volumes_j,
    ) = setup_volume_and_distance(
        volumes[:n_i],
        volumes[:n_j],
        box_size,
        top_k,
        exponent,
        cost_scale_factor,
        normalize,
    )

    distance_matrix_dask_gw = get_distance_matrix_gw_via_fw(
        marginals_i,
        marginals_j,
        sparse_coordinates_sets_i,
        sparse_coordinates_sets_j,
        pairwise_distances_i**2,
        pairwise_distances_j**2,
    )

    np.save(
        os.path.join(
            args.outdir,
            f"gw_weighted_voxel_fw_topk{top_k}_ds{box_size}_float{precision}_exponent{exponent}_{len(volumes_i)}x{len(volumes_j)}_23.npy",
        ),
        distance_matrix_dask_gw,
    )
    return distance_matrix_dask_gw


def main(args):
    n_i = args.n_i
    n_j = args.n_j
    box_size = args.box_size
    top_k = args.top_k
    exponent = args.exponent
    cost_scale_factor = args.cost_scale_factor
    normalize = not args.skip_normalize

    fname = args.fname  # e.g. /path/to/submission_23.pt
    submission = torch.load(fname, weights_only=False)
    volumes = submission["volumes"].to(torch_dtype)

    (
        _,
        _,
        marginals_i,
        marginals_j,
        sparse_coordinates_sets_i,
        sparse_coordinates_sets_j,
        pairwise_distances_i,
        pairwise_distances_j,
        volumes_i,
        volumes_j,
    ) = setup_volume_and_distance(
        volumes[:n_i],
        volumes[:n_j],
        box_size,
        top_k,
        exponent,
        cost_scale_factor,
        normalize,
    )

    gw_distance_function_key = "gromov_wasserstein2"
    scheduler = args.scheduler
    element_wise = args.element_wise

    if args.frank_wolfe:
        distance_matrix_dask_gw = get_distance_matrix_gw_via_fw(
            marginals_i,
            marginals_j,
            sparse_coordinates_sets_i,
            sparse_coordinates_sets_j,
            pairwise_distances_i,
            pairwise_distances_j,
        )

        np.save(
            os.path.join(
                args.outdir,
                f"gw_weighted_voxel_fw_topk{top_k}_ds{box_size}_float{precision}_exponent{exponent}_{len(volumes_i)}x{len(volumes_j)}_23.npy",
            ),
            distance_matrix_dask_gw,
        )
        return distance_matrix_dask_gw

    else:
        distance_matrix_dask_gw = get_distance_matrix_dask_gw(
            marginals_i,
            marginals_j,
            pairwise_distances_i,
            pairwise_distances_j,
            scheduler,
            element_wise,
            gw_distance_function_key,
        )

        np.save(
            os.path.join(
                args.outdir,
                f"gw_weighted_voxel_topk{top_k}_ds{box_size}_float{precision}_costscalefactor{cost_scale_factor}_exponent{exponent}_{len(volumes_i)}x{len(volumes_j)}_23.npy",
            ),
            distance_matrix_dask_gw,
        )
        return distance_matrix_dask_gw


if __name__ == "__main__":
    args = parse_args()
    if args.slurm:
        job_id = os.environ["SLURM_JOB_ID"]
        with SLURMRunner(
            scheduler_file=args.scheduler_file,
        ) as runner:
            # The runner object contains the scheduler address and can be passed directly to a client
            with Client(runner) as client:
                get_distance_matrix_dask_gw = main(args)

    else:
        with Client(local_directory=args.local_directory) as client:
            get_distance_matrix_dask_gw = main(args)

    # linter thinks client is unused, so need to do something with client as a workaround
    assert isinstance(client, type(client))
