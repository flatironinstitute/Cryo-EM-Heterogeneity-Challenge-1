import math
import torch
from typing import Optional, Sequence
from typing_extensions import override
import mrcfile
import numpy as np
from torch.utils.data import Dataset


class GT_Dataset(Dataset):
    def __init__(self, npy_file):
        self.npy_file = npy_file
        self.data = np.load(npy_file, mmap_mode="r+")

        self.shape = self.data.shape
        self._dim = len(self.data.shape)

    def dim(self):
        return self._dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        return torch.from_numpy(sample.copy())


class MapToMapDistance:
    def __init__(self, config):
        self.config = config

    def get_distance(self, map1, map2):
        """Compute the distance between two maps."""
        raise NotImplementedError()

    def get_distance_matrix(self, maps1, maps2, global_store_of_running_results):
        """Compute the distance matrix between two sets of maps."""
        chunk_size_submission = self.config["analysis"]["chunk_size_submission"]
        chunk_size_gt = self.config["analysis"]["chunk_size_gt"]
        # load in memory as torch tensors
        if True:  # config.low_memory:
            distance_matrix = torch.empty(len(maps1), len(maps2))
            n_chunks_low_memory = 100
            for idxs in torch.arange(len(maps1)).chunk(n_chunks_low_memory):
                maps1_in_memory = maps1[idxs]
                sub_distance_matrix = torch.vmap(
                    lambda maps1: torch.vmap(
                        lambda maps2: self.get_distance(maps1, maps2),
                        chunk_size=chunk_size_submission,
                    )(maps2),
                    chunk_size=chunk_size_gt,
                )(maps1_in_memory)
                distance_matrix[idxs] = sub_distance_matrix

        else:
            assert False, "Not implemented"
            distance_matrix = torch.vmap(
                lambda maps1: torch.vmap(
                    lambda maps2: self.get_distance(maps1, maps2),
                    chunk_size=chunk_size_submission,
                )(maps2),
                chunk_size=chunk_size_gt,
            )(maps1)
        return distance_matrix

    def get_computed_assets(self, maps1, maps2, global_store_of_running_results):
        """Return any computed assets that are needed for (downstream) analysis."""
        return {}


class MapToMapDistanceLowMemory(MapToMapDistance):
    """General class for map-to-map distance metrics that require low memory."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def compute_cost(self, map_1, map_2):
        raise NotImplementedError()

    @override
    def get_distance(self, map1, map2, global_store_of_running_results):
        map1 = map1.flatten()
        if self.config["analysis"]["normalize"]["do"]:
            if self.config["analysis"]["normalize"]["method"] == "median_zscore":
                map1 -= map1.median()
                map1 /= map1.std()
            else:
                raise NotImplementedError(
                    f"Normalization method {self.config['analysis']['normalize']['method']} not implemented."
                )
        map1 = map1[global_store_of_running_results["mask"]]

        return self.compute_cost(map1, map2)

    @override
    def get_distance_matrix(self, maps1, maps2, global_store_of_running_results):
        maps_gt_flat = maps1
        maps_user_flat = maps2
        cost_matrix = torch.empty(len(maps_gt_flat), len(maps_user_flat))
        for idx_gt in range(len(maps_gt_flat)):
            for idx_user in range(len(maps_user_flat)):
                cost_matrix[idx_gt, idx_user] = self.get_distance(
                    maps_gt_flat[idx_gt],
                    maps_user_flat[idx_user],
                    global_store_of_running_results,
                )
        return cost_matrix


def norm2(map1, map2):
    return torch.norm(map1 - map2) ** 2


class L2DistanceNorm(MapToMapDistance):
    """L2 distance norm"""

    def __init__(self, config):
        super().__init__(config)

    @override
    def get_distance(self, map1, map2):
        return norm2(map1, map2)


class L2DistanceNormLowMemory(MapToMapDistanceLowMemory):
    """L2 distance norm"""

    def __init__(self, config):
        super().__init__(config)

    @override
    def compute_cost(self, map1, map2):
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


class CorrelationLowMemory(MapToMapDistanceLowMemory):
    """Correlation."""

    def __init__(self, config):
        super().__init__(config)

    @override
    def compute_cost(self, map1, map2):
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
        shape (n_pix,n_pix,n_pix)
    map2 : torch.Tensor
        shape (n_pix,n_pix,n_pix)

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


class BioEM3dDistanceLowMemory(MapToMapDistanceLowMemory):
    """BioEM 3D distance."""

    def __init__(self, config):
        super().__init__(config)

    @override
    def compute_cost(self, map1, map2):
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
    """  # noqa: E501
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

    def compute_cost_fsc_chunk(self, maps_gt_flat, maps_user_flat, n_pix):
        """
        Compute the cost between two maps using the Fourier Shell Correlation in 3D.

        Notes
        -----
        fourier_shell_correlation can only batch on first input set of maps,
        so we compute the cost one row (gt map idx) at a time
        """
        cost_matrix = torch.empty(len(maps_gt_flat), len(maps_user_flat))
        fsc_matrix = torch.zeros(len(maps_gt_flat), len(maps_user_flat), n_pix // 2)
        for idx in range(len(maps_gt_flat)):
            corr_vector = fourier_shell_correlation(
                maps_user_flat.reshape(-1, n_pix, n_pix, n_pix),
                maps_gt_flat[idx].reshape(n_pix, n_pix, n_pix),
            )
            dist = 1 - corr_vector.mean(dim=1)  # TODO: spectral cutoff
            fsc_matrix[idx] = corr_vector
            cost_matrix[idx] = dist
        return cost_matrix, fsc_matrix

    @override
    def get_distance_matrix(self, maps1, maps2, global_store_of_running_results):
        """
        Applies a mask to the maps and computes the cost matrix using the Fourier Shell Correlation.
        """
        maps_gt_flat = maps1
        maps_user_flat = maps2
        n_pix = self.config["data"]["n_pix"]
        maps_gt_flat_cube = torch.zeros(len(maps_gt_flat), n_pix**3)
        maps_user_flat_cube = torch.zeros(len(maps_user_flat), n_pix**3)

        if self.config["data"]["mask"]["do"]:
            mask = (
                mrcfile.open(self.config["data"]["mask"]["volume"])
                .data.astype(bool)
                .flatten()
            )
            maps_gt_flat_cube[:, mask] = maps_gt_flat
            maps_user_flat_cube[:, mask] = maps_user_flat
        else:
            maps_gt_flat_cube = maps_gt_flat
            maps_user_flat_cube = maps_user_flat

        cost_matrix, fsc_matrix = self.compute_cost_fsc_chunk(
            maps_gt_flat_cube, maps_user_flat_cube, n_pix
        )
        self.stored_computed_assets = {"fsc_matrix": fsc_matrix}
        return cost_matrix

    @override
    def get_computed_assets(self, maps1, maps2, global_store_of_running_results):
        return self.stored_computed_assets  # must run get_distance_matrix first


class FSCDistanceLowMemory(MapToMapDistance):
    """Fourier Shell Correlation distance."""

    def __init__(self, config):
        super().__init__(config)
        self.n_pix = self.config["data"]["n_pix"]
        self.config = config

    def compute_cost(self, map_1, map_2):
        raise NotImplementedError()

    @override
    def get_distance(self, map1, map2, global_store_of_running_results):
        map_gt_flat = map1 = map1.flatten()
        map_gt_flat_cube = torch.zeros(self.n_pix**3)
        if self.config["data"]["mask"]["do"]:
            map_gt_flat = map_gt_flat[global_store_of_running_results["mask"]]
            map_gt_flat_cube[global_store_of_running_results["mask"]] = map_gt_flat
        else:
            map_gt_flat_cube = map_gt_flat

        corr_vector = fourier_shell_correlation(
            map_gt_flat_cube.reshape(self.n_pix, self.n_pix, self.n_pix),
            map2.reshape(self.n_pix, self.n_pix, self.n_pix),
        )
        dist = 1 - corr_vector.mean()  # TODO: spectral cutoff
        self.stored_computed_assets = {"corr_vector": corr_vector}
        return dist

    @override
    def get_distance_matrix(self, maps1, maps2, global_store_of_running_results):
        maps_gt_flat = maps1
        maps_user_flat = maps2
        cost_matrix = torch.empty(len(maps_gt_flat), len(maps_user_flat))
        fsc_matrix = torch.zeros(
            len(maps_gt_flat), len(maps_user_flat), self.n_pix // 2
        )
        for idx_gt in range(len(maps_gt_flat)):
            for idx_user in range(len(maps_user_flat)):
                cost_matrix[idx_gt, idx_user] = self.get_distance(
                    maps_gt_flat[idx_gt],
                    maps_user_flat[idx_user],
                    global_store_of_running_results,
                )
                fsc_matrix[idx_gt, idx_user] = self.stored_computed_assets[
                    "corr_vector"
                ]
        self.stored_computed_assets = {"fsc_matrix": fsc_matrix}
        return cost_matrix

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
            self.config["data"]["n_pix"] // 2
        )  # TODO: check for odd psizes if this should be +1
        psize = self.config["data"]["psize"]
        fsc_matrix = global_store_of_running_results[self.fsc_label]["computed_assets"][
            "fsc_matrix"
        ]
        units_Angstroms = (
            2 * psize / (np.arange(1, fourier_pixel_max + 1) / fourier_pixel_max)
        )

        def res_at_fsc_threshold(fscs, threshold=0.5):
            res_fsc_half = np.argmin(fscs > threshold, axis=-1)
            fraction_nyquist = 0.5 * res_fsc_half / fscs.shape[-1]
            return res_fsc_half, fraction_nyquist

        res_fsc_half, fraction_nyquist = res_at_fsc_threshold(fsc_matrix)
        self.stored_computed_assets = {"fraction_nyquist": fraction_nyquist}
        return units_Angstroms[res_fsc_half]


class FSCResDistanceLowMemory(FSCResDistance):
    """FSC Resolution distance.

    The resolution at which the Fourier Shell Correlation reaches 0.5.
    Built on top of the FSCDistance class. This needs to be run first and store the FSC matrix in the computed assets.
    """

    def __init__(self, config):
        super().__init__(config)
        self.fsc_label = "fsc_low_memory"
