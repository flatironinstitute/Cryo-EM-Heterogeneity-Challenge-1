import os
import subprocess
import math
import torch
import logging
from typing import Optional, Sequence
from typing_extensions import override
import mrcfile
import numpy as np
from dask.distributed import Client
from dask_jobqueue.slurm import SLURMRunner
import torch.multiprocessing as mp

from .gromov_wasserstein.gw_weighted_voxels import (
    get_distance_matrix_dask_gw,
    get_distance_matrix_gw_via_fw,
)  # TODO: rename get_distance_matrix_dask_gw to python_ot
from .gromov_wasserstein.gw_weighted_voxels import (
    setup_volume_and_distance,
)
from .procrustes_wasserstein.procrustes_wasserstein import (
    procrustes_wasserstein,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize(maps, method):
    if method == "median_zscore":
        maps -= maps.median(dim=1, keepdim=True).values
        maps /= maps.std(dim=1, keepdim=True)
    else:
        raise NotImplementedError(f"Normalization method {method} not implemented.")
    return maps


class MapToMapDistance:
    def __init__(self, config):
        self.config = config
        self.do_low_memory_mode = (
            self.config["metrics"]["shared_params"]["low_memory"] is not None
        )
        self.chunk_size_gt = self.config["metrics"]["shared_params"]["chunk_size_gt"]
        self.chunk_size_submission = self.config["metrics"]["shared_params"][
            "chunk_size_submission"
        ]
        self.box_size = self.config["data_params"]["box_size"]
        if self.do_low_memory_mode:
            self.chunk_size = self.config["metrics"]["shared_params"]["low_memory"][
                "chunk_size"
            ]
        if self.config["data_params"]["mask_params"]["apply_mask"]:
            self.mask = (
                mrcfile.open(self.config["data_params"]["mask_params"]["path_to_mask"])
                .data.astype(bool)
                .flatten()
            )

    def get_distance(self, map1, map2):
        """Compute the distance between two maps."""
        raise NotImplementedError()

    def get_sub_distance_matrix(self, maps1, maps2, idxs):
        """Compute the distance matrix between two sets of maps."""
        sub_distance_matrix = torch.vmap(
            lambda maps1: torch.vmap(
                lambda maps2: self.get_distance(maps1, maps2),
                chunk_size=self.chunk_size_submission,
            )(maps2),
            chunk_size=self.chunk_size_gt,
        )(maps1)
        return sub_distance_matrix

    def distance_matrix_precomputation(maps1, maps2, global_store_of_running_results):
        """Pre-compute any assets needed for the distance matrix computation."""
        return

    def get_distance_matrix(self, maps1, maps2, global_store_of_running_results):
        """Compute the distance matrix between two sets of maps."""
        if self.config["data_params"]["mask_params"]["apply_mask"]:
            maps2 = maps2[:, self.mask]

        else:
            maps2 = maps2.reshape(len(maps2), -1)

        if self.config["metrics"]["shared_params"]["normalize_params"] is not None:
            maps2 = normalize(
                maps2,
                method=self.config["metrics"]["shared_params"]["normalize_params"][
                    "method"
                ],
            )
        if self.do_low_memory_mode:
            self.n_chunks_low_memory = len(maps1) // self.chunk_size
            distance_matrix = torch.empty(len(maps1), len(maps2))
            for idxs in torch.arange(len(maps1)).chunk(self.n_chunks_low_memory):
                maps1_in_memory = maps1[idxs]
                if self.config["data_params"]["mask_params"]["apply_mask"]:
                    maps1_in_memory = maps1_in_memory.reshape(len(idxs), -1)[
                        :, self.mask
                    ]
                else:
                    maps1_in_memory = maps1_in_memory.reshape(len(maps1_in_memory), -1)
                if (
                    self.config["metrics"]["shared_params"]["normalize_params"]
                    is not None
                ):
                    maps1_in_memory = normalize(
                        maps1_in_memory,
                        method=self.config["metrics"]["shared_params"][
                            "normalize_params"
                        ]["method"],
                    )
                sub_distance_matrix = self.get_sub_distance_matrix(
                    maps1_in_memory,
                    maps2,
                    idxs,
                )
                distance_matrix[idxs] = sub_distance_matrix

        else:
            maps1 = maps1.reshape(len(maps1), -1)
            if self.config["data_params"]["mask_params"]["apply_mask"]:
                maps1 = maps1.reshape(len(maps1), -1)[:, self.mask]
            maps2 = maps2.reshape(len(maps2), -1)
            distance_matrix = torch.vmap(
                lambda maps1: torch.vmap(
                    lambda maps2: self.get_distance(maps1, maps2),
                    chunk_size=self.chunk_size_submission,
                )(maps2),
                chunk_size=self.chunk_size_gt,
            )(maps1)
        return distance_matrix

    def get_computed_assets(self, maps1, maps2, global_store_of_running_results):
        """Return any computed assets that are needed for (downstream) analysis."""
        return {}


def norm2(map1, map2):
    return torch.norm(map1 - map2) ** 2


class L2DistanceNorm(MapToMapDistance):
    """L2 distance norm"""

    def __init__(self, config):
        super().__init__(config)

    @override
    def get_distance(self, map1, map2):
        return norm2(map1, map2)


def correlation(map1, map2):
    return (map1 * map2).sum()


class Correlation(MapToMapDistance):
    """Correlation.

    Not technically a distance metric, but a similarity."""

    def __init__(self, config):
        super().__init__(config)

    @override
    def get_distance(self, map1, map2):
        return correlation(map1, map2)


def compute_bioem3d_cost(map1, map2):
    """
    Compute the cost between two maps using the BioEM cost function in 3D.

    Notes
    -----
    See Eq. 10 in 10.1016/j.jsb.2013.10.006

    Parameters
    ----------
    map1 : torch.Tensor
        shape (box_size,box_size,box_size)
    map2 : torch.Tensor
        shape (box_size,box_size,box_size)

    Returns
    -------
    cost : torch.Tensor
        shape (1,)
    """
    m1, m2 = map1.reshape(-1), map2.reshape(-1)
    co = m1.sum()
    cc = m2.sum()
    coo = m1.pow(2).sum()
    ccc = m2.pow(2).sum()
    coc = (m1 * m2).sum()

    N = len(m1)

    t1 = 2 * torch.pi * math.exp(1)
    t2 = N * (ccc * coo - coc * coc) + 2 * co * coc * cc - ccc * co * co - coo * cc * cc
    t3 = (N - 2) * (N * ccc - cc * cc)

    smallest_float = torch.finfo(m1.dtype).tiny
    log_prob = (
        0.5 * torch.pi
        + math.log(t1) * (1 - N / 2)
        + t2.clamp(smallest_float).log() * (3 / 2 - N / 2)
        + t3.clamp(smallest_float).log() * (N / 2 - 2)
    )
    cost = -log_prob
    return cost


class BioEM3dDistance(MapToMapDistance):
    """BioEM 3D distance."""

    def __init__(self, config):
        super().__init__(config)

    @override
    def get_distance(self, map1, map2):
        return compute_bioem3d_cost(map1, map2)


def fourier_shell_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: Sequence[int] = (-3, -2, -1),
    normalize: bool = True,
    max_k: Optional[int] = None,
):
    """Computes Fourier Shell / Ring Correlation (FSC) between x and y.

    Parameters
    ----------
    x : torch.Tensor
        First input tensor.
    y : torch.Tensor
        Second input tensor.
    dim : Tuple[int, ...]
        Dimensions over which to take the Fourier transform.
    normalize : bool
        Whether to normalize (i.e. compute correlation) or not (i.e. compute covariance).
        Note that when `normalize=False`, we still divide by the number of elements in each shell.
    max_k : int
        The maximum shell to compute the correlation for.

    Returns
    -------
    torch.Tensor
        The correlation between x and y for each Fourier shell.
    """
    batch_shape = x.shape[: -len(dim)]

    freqs = [torch.fft.fftfreq(x.shape[d], d=1 / x.shape[d]).to(x) for d in dim]
    freq_total = (
        torch.cartesian_prod(*freqs).view(*[len(f) for f in freqs], -1).norm(dim=-1)
    )

    x_f = torch.fft.fftn(x, dim=dim)
    y_f = torch.fft.fftn(y, dim=dim)

    n = min(x.shape[d] for d in dim)

    if max_k is None:
        max_k = n // 2

    result = x.new_zeros(batch_shape + (max_k,))

    for i in range(1, max_k + 1):
        mask = (freq_total >= i - 0.5) & (freq_total < i + 0.5)
        x_ri = x_f[..., mask]
        y_fi = y_f[..., mask]

        if x.is_cuda:
            c_i = torch.linalg.vecdot(x_ri, y_fi).real
        else:
            # vecdot currently bugged on CPU for torch 2.0 in some configurations
            c_i = torch.sum(x_ri * y_fi.conj(), dim=-1).real

        if normalize:
            c_i /= torch.linalg.norm(x_ri, dim=-1) * torch.linalg.norm(y_fi, dim=-1)
        else:
            c_i /= x_ri.shape[-1]

        result[..., i - 1] = c_i

    return result


class FSCDistance(MapToMapDistance):
    """Fourier Shell Correlation distance.

    One minus the correlation between two maps in Fourier space."""

    def __init__(self, config):
        super().__init__(config)

    def compute_cost_fsc_chunk(self, maps_gt_flat, maps_user_flat, box_size):
        """
        Compute the cost between two maps using the Fourier Shell Correlation in 3D.

        Notes
        -----
        fourier_shell_correlation can only batch on first input set of maps,
        so we compute the cost one row (gt map idx) at a time
        """
        cost_matrix = torch.empty(len(maps_gt_flat), len(maps_user_flat))
        fsc_matrix = torch.zeros(len(maps_gt_flat), len(maps_user_flat), box_size // 2)
        for idx in range(len(maps_gt_flat)):
            corr_vector = fourier_shell_correlation(
                maps_user_flat.reshape(-1, box_size, box_size, box_size),
                maps_gt_flat[idx].reshape(box_size, box_size, box_size),
            )
            dist = 1 - corr_vector.mean(dim=1)  # TODO: spectral cutoff
            fsc_matrix[idx] = corr_vector
            cost_matrix[idx] = dist
        return cost_matrix, fsc_matrix

    @override
    def distance_matrix_precomputation(self, maps1, maps2):
        self.len_maps1 = len(maps1)
        self.len_maps2 = len(maps2)
        self.stored_computed_assets = {
            "fsc_matrix": torch.empty(
                self.len_maps1, self.len_maps2, self.box_size // 2
            )
        }
        return

    @override
    def get_sub_distance_matrix(self, maps1, maps2, idxs):
        """
        Applies a mask to the maps and computes the cost matrix using the Fourier Shell Correlation.
        """
        maps_gt_flat = maps1
        maps_user_flat = maps2
        box_size = self.config["data_params"]["box_size"]
        maps_gt_flat_cube = torch.zeros(len(maps_gt_flat), box_size**3)
        maps_user_flat_cube = torch.zeros(len(maps_user_flat), box_size**3)

        if self.config["data_params"]["mask_params"]["apply_mask"]:
            maps_gt_flat_cube[:, self.mask] = maps_gt_flat[:]
            maps_user_flat_cube[:, self.mask] = maps_user_flat

        else:
            maps_gt_flat_cube = maps_gt_flat
            maps_user_flat_cube = maps_user_flat

        cost_matrix, fsc_matrix = self.compute_cost_fsc_chunk(
            maps_gt_flat_cube, maps_user_flat_cube, box_size
        )
        self.stored_computed_assets["fsc_matrix"][idxs] = fsc_matrix
        return cost_matrix

    @override
    def get_distance_matrix(self, maps1, maps2, global_store_of_running_results):
        """Compute the distance matrix between two sets of maps."""
        if self.config["data_params"]["mask_params"]["apply_mask"]:
            maps2 = maps2[:, self.mask]
        else:
            maps2 = maps2.reshape(len(maps2), -1)

        if self.config["metrics"]["shared_params"]["normalize_params"] is not None:
            maps2 = normalize(
                maps2,
                method=self.config["metrics"]["shared_params"]["normalize_params"][
                    "method"
                ],
            )
        if self.chunk_size is None:
            self.n_chunks_low_memory = 1
        else:
            self.n_chunks_low_memory = len(maps1) // self.chunk_size
        distance_matrix = torch.empty(len(maps1), len(maps2))
        for idxs in torch.arange(len(maps1)).chunk(self.n_chunks_low_memory):
            maps1_in_memory = maps1[idxs]

            if self.config["data_params"]["mask_params"]["apply_mask"]:
                maps1_in_memory = maps1_in_memory[:].reshape(len(idxs), -1)[
                    :, self.mask
                ]

            else:
                maps1_in_memory = maps1_in_memory.reshape(len(maps1_in_memory), -1)
            if self.config["metrics"]["shared_params"]["normalize_params"] is not None:
                maps1_in_memory = normalize(
                    maps1_in_memory,
                    method=self.config["metrics"]["shared_params"]["normalize_params"][
                        "method"
                    ],
                )
            sub_distance_matrix = self.get_sub_distance_matrix(
                maps1_in_memory,
                maps2,
                idxs,
            )
            distance_matrix[idxs] = sub_distance_matrix
        return distance_matrix

    @override
    def get_computed_assets(self, maps1, maps2, global_store_of_running_results):
        return self.stored_computed_assets  # must run get_distance_matrix first


class FSCResDistance(MapToMapDistance):
    """FSC Resolution distance.

    The resolution at which the Fourier Shell Correlation reaches 0.5.
    Built on top of the FSCDistance class. This needs to be run first and store the FSC matrix in the computed assets.
    """

    def __init__(self, config):
        super().__init__(config)
        self.fsc_label = "fsc"

    @override
    def get_distance_matrix(
        self, maps1, maps2, global_store_of_running_results
    ):  # custom method
        # get fsc matrix
        fourier_pixel_max = (
            self.config["data_params"]["box_size"] // 2
        )  # TODO: check for odd voxel_sizes if this should be +1
        voxel_size = self.config["data_params"]["voxel_size"]
        fsc_matrix = global_store_of_running_results[self.fsc_label]["computed_assets"][
            "fsc_matrix"
        ]

        units_Angstroms = (
            2 * voxel_size / (np.arange(1, fourier_pixel_max + 1) / fourier_pixel_max)
        )

        def res_at_fsc_threshold(fscs, threshold=0.5):
            res_fsc_half = np.argmin(fscs > threshold, axis=-1)
            fraction_nyquist = 0.5 * res_fsc_half / fscs.shape[-1]
            return res_fsc_half, fraction_nyquist

        res_fsc_half, fraction_nyquist = res_at_fsc_threshold(fsc_matrix)
        self.stored_computed_assets = {"fraction_nyquist": fraction_nyquist}
        return units_Angstroms[res_fsc_half]


class Zernike3DDistance(MapToMapDistance):
    """Zernike3D based distance.

    Zernike3D distance relies on the estimation of the non-linear transformation needed to align two different maps.
    The RMSD of the associated non-linear alignment represented as a deformation field is then used as the distance
    between two maps
    """

    @override
    def get_distance_matrix(self, maps1, maps2, global_store_of_running_results):
        gpuID = self.config["metrics"]["zernike3d"]["gpuID"]
        outputPath = self.config["metrics"]["zernike3d"]["tmpDir"]
        thr = self.config["metrics"]["zernike3d"]["thr"]
        numProjections = self.config["metrics"]["zernike3d"]["numProjections"]

        # Create output directory
        if not os.path.isdir(outputPath):
            os.mkdir(outputPath)

        # Prepare data to call external
        targets_paths = os.path.join(outputPath, "target_maps.npy")
        references_path = os.path.join(outputPath, "reference_maps.npy")
        if not os.path.isfile(targets_paths):
            np.save(targets_paths, maps1)
        if not os.path.isfile(references_path):
            np.save(references_path, maps2)

        # Check conda is in PATH (otherwise abort as external software is not installed)
        try:
            subprocess.check_call("conda", shell=True, stdout=subprocess.PIPE)
        except FileNotFoundError:
            raise Exception("Conda not found in PATH... Aborting")

        # Check if conda env is installed
        env_installed = subprocess.run(
            r"conda env list | grep 'flexutils-tensorflow '",
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
        ).stdout
        env_installed = bool(
            env_installed.decode("utf-8").replace("\n", "").replace("*", "")
        )
        if not env_installed:
            raise Exception("External software not found... Aborting")

        # Find conda executable (needed to activate conda envs in a subprocess)
        condabin_path = subprocess.run(
            r"which conda | sed 's: ::g'",
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
        ).stdout
        condabin_path = condabin_path.decode("utf-8").replace("\n", "").replace("*", "")

        # Call external program
        subprocess.check_call(
            f'eval "$({condabin_path} shell.bash hook)" &&'
            f" conda activate flexutils-tensorflow && "
            f"compute_distance_matrix_zernike3deep.py --references_file {references_path} "
            f"--targets_file {targets_paths} --out_path {outputPath} --gpu {gpuID} --num_projections {numProjections} "
            f"--thr {thr}",
            shell=True,
        )

        # Read distance matrix
        dists = np.load(os.path.join(outputPath, "dist_mat.npy")).T
        self.stored_computed_assets = {"zernike3d": dists}
        return dists

    @override
    def get_computed_assets(self, maps1, maps2, global_store_of_running_results):
        """Note: must run get_distance_matrix first"""
        return self.stored_computed_assets


def compute_distance(args):
    (
        idx_i,
        sparse_coordinates_i,
        sparse_coordinates_sets_j,
        marginal_i,
        marginals_j,
        extra_params,
    ) = args
    results = []
    for idx_j in range(len(sparse_coordinates_sets_j)):
        if idx_i == idx_j:
            continue
        logger.info(f"Computing distance between {idx_i} and {idx_j}")
        _, _, logs = procrustes_wasserstein(
            torch.from_numpy(sparse_coordinates_i),
            torch.from_numpy(sparse_coordinates_sets_j[idx_j]),
            torch.from_numpy(marginal_i),
            torch.from_numpy(marginals_j[idx_j]),
            max_iter=extra_params["max_iter"],
            tol=extra_params["tol"],
        )
        results.append((idx_i, idx_j, logs[-1]["cost"], logs))
        logger.info(f"Done      distance between {idx_i} and {idx_j}")

    return results


class ProcrustesWassersteinDistance(MapToMapDistance):
    """Procrustes-Wasserstein

    Procrustes-Wasserstein distance is invariant to SE(3) map alignment, because it maximizes the transport plan and rototranslation.
    """

    mp.set_start_method("spawn", force=True)

    @override
    def get_distance_matrix(self, maps1, maps2, global_store_of_running_results):
        logger.info("Computing Procrustes-Wasserstein distance")
        extra_params = self.config["metrics"]["procrustes_wasserstein"]

        box_size = self.config["data_params"]["box_size"]
        maps1 = maps1.reshape((len(maps1), box_size, box_size, box_size))
        maps2 = maps2.reshape((len(maps2), box_size, box_size, box_size))

        logger.info("Setting up volumes and distances")
        (
            _,
            _,
            marginals_i,
            marginals_j,
            sparse_coordinates_sets_i,
            sparse_coordinates_sets_j,
            _,
            _,
            _,
            _,
        ) = setup_volume_and_distance(
            maps1,
            maps2,
            extra_params["downsample_box_size"],
            extra_params["top_k"],
            exponent=1,
            cost_scale_factor=1,
            normalize=False,
        )
        logger.info("Done setting up volumes and distances")

        dists = torch.zeros(len(maps1), len(maps2))
        self.stored_computed_assets = {"procrustes_wasserstein": {}}

        def parallel_compute_distances(
            maps1,
            maps2,
            sparse_coordinates_sets_i,
            sparse_coordinates_sets_j,
            marginals_i,
            marginals_j,
            extra_params,
        ):
            dists = torch.zeros(len(maps1), len(maps2))
            local_stored_computed_assets = {}

            logger.info("Setting up tasks")
            with mp.Pool(processes=mp.cpu_count()) as pool:
                args_list = [
                    (
                        idx_i,
                        sparse_coordinates_sets_i[idx_i],
                        sparse_coordinates_sets_j,
                        marginals_i[idx_i],
                        marginals_j,
                        extra_params,
                    )
                    for idx_i in range(len(maps1))
                ]

                logger.info("Computing distances")
                results_list = pool.map(compute_distance, args_list)

            logger.info("Unpacking results")
            for results in results_list:
                for idx_i, idx_j, cost, logs in results:
                    dists[idx_i, idx_j] = cost
                    local_stored_computed_assets[(idx_i, idx_j)] = logs

            return dists, local_stored_computed_assets

        dists, local_stored_computed_assets = parallel_compute_distances(
            maps1,
            maps2,
            sparse_coordinates_sets_i,
            sparse_coordinates_sets_j,
            marginals_i,
            marginals_j,
            extra_params,
        )
        self.stored_computed_assets["procrustes_wasserstein"] = (
            local_stored_computed_assets
        )
        return dists

    @override
    def get_computed_assets(self, maps1, maps2, global_store_of_running_results):
        return self.stored_computed_assets  # must run get_distance_matrix first


class GromovWassersteinDistance(MapToMapDistance):
    """Gromov-Wasserstein distance.

    Gromov-Wasserstein distance is invariant to map alignment, because it compares the self-distances in a map, which are S3(3) equivariant.
    """

    @override
    def get_distance_matrix(self, maps1, maps2, global_store_of_running_results):
        extra_params = self.config["metrics"]["gromov_wasserstein"]
        box_size = self.config["data_params"]["box_size"]
        maps1 = maps1.reshape((len(maps1), box_size, box_size, box_size))
        maps2 = maps2.reshape((len(maps2), box_size, box_size, box_size))

        (
            _,
            _,
            marginals_i,
            marginals_j,
            sparse_coordinates_sets_i,
            sparse_coordinates_sets_j,
            pairwise_distances_i,
            pairwise_distances_j,
            _,
            _,
        ) = setup_volume_and_distance(
            maps1,
            maps2,
            extra_params["downsample_box_size"],
            extra_params["top_k"],
            extra_params["exponent"],
            extra_params["cost_scale_factor"],
            normalize=False,
        )

        if extra_params["solver"] == "frank_wolfe":
            distance_matrix_gw = get_distance_matrix_gw_via_fw(
                marginals_i,
                marginals_j,
                sparse_coordinates_sets_i,
                sparse_coordinates_sets_j,
                pairwise_distances_i**2,
                pairwise_distances_j**2,
                max_iter=extra_params["frank_wolfe_params"]["max_iter"],
                gamma_atol=extra_params["frank_wolfe_params"]["gamma_atol"],
            )
            self.stored_computed_assets = {"gromov_wasserstein": distance_matrix_gw}
            return distance_matrix_gw

        if extra_params["dask"]["slurm"]:
            job_id = os.environ["SLURM_JOB_ID"]
            scheduler_file = os.path.join(
                extra_params["dask"]["scheduler_file_directory"],
                f"scheduler-{job_id}.json",
            )

            with SLURMRunner(
                scheduler_file=scheduler_file,
            ) as runner:
                # The runner object contains the scheduler address and can be passed directly to a client
                with Client(runner) as client:
                    if extra_params["solver"] == "python_ot":
                        distance_matrix_gw = get_distance_matrix_dask_gw(
                            marginals_i=marginals_i,
                            marginals_j=marginals_j,
                            pairwise_distances_i=pairwise_distances_i,
                            pairwise_distances_j=pairwise_distances_j,
                            scheduler=extra_params["dask"]["scheduler"],
                            elementwise_not_rowwise=extra_params[
                                "elementwise_not_rowwise"
                            ],
                            gw_distance_function_key=extra_params["python_ot_params"][
                                "gw_distance_function_key"
                            ],
                            tol_abs=extra_params["python_ot_params"]["tol_abs"],
                            tol_rel=extra_params["python_ot_params"]["tol_rel"],
                            max_iter=extra_params["python_ot_params"]["max_iter"],
                            verbose=extra_params["python_ot_params"]["verbose"],
                            loss_fun=extra_params["python_ot_params"]["loss_fun"],
                        )
                    else:
                        raise NotImplementedError(
                            f"Method {extra_params['solver']} not implemented for SLURM."
                        )

        else:
            local_directory = extra_params["dask"]["local_directory"]
            with Client(local_directory=local_directory) as client:
                if extra_params["solver"] == "python_ot":
                    distance_matrix_gw = get_distance_matrix_dask_gw(
                        marginals_i=marginals_i,
                        marginals_j=marginals_j,
                        pairwise_distances_i=pairwise_distances_i,
                        pairwise_distances_j=pairwise_distances_j,
                        scheduler=extra_params["dask"]["scheduler"],
                        elementwise_not_rowwise=extra_params["elementwise_not_rowwise"],
                        gw_distance_function_key=extra_params["python_ot_params"][
                            "gw_distance_function_key"
                        ],
                        tol_abs=extra_params["python_ot_params"]["tol_abs"],
                        tol_rel=extra_params["python_ot_params"]["tol_rel"],
                        max_iter=extra_params["python_ot_params"]["max_iter"],
                        verbose=extra_params["python_ot_params"]["verbose"],
                        loss_fun=extra_params["python_ot_params"]["loss_fun"],
                    )
                else:
                    raise NotImplementedError(
                        f"Method {extra_params['solver']} not implemented for local directory."
                    )
        assert isinstance(client, type(client))

        self.stored_computed_assets = {"gromov_wasserstein": distance_matrix_gw}
        return distance_matrix_gw

    @override
    def get_computed_assets(self, maps1, maps2, global_store_of_running_results):
        return self.stored_computed_assets  # must run get_distance_matrix first
