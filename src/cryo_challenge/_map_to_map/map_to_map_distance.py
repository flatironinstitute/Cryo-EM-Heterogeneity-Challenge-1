import math
import torch
from typing import Optional, Sequence


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


def compute_cost_l2(map_1, map_2):
    return ((map_1 - map_2) ** 2).sum()


def compute_cost_corr(map_1, map_2):
    return (map_1 * map_2).sum()


def compute_cost_fsc_chunk(maps_gt_flat, maps_user_flat, n_pix):
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
        print(idx)
        corr_vector = fourier_shell_correlation(
            maps_user_flat.reshape(-1, n_pix, n_pix, n_pix),
            maps_gt_flat[idx].reshape(n_pix, n_pix, n_pix),
        )
        dist = 1 - corr_vector.mean(dim=1)  # TODO: spectral cutoff
        fsc_matrix[idx] = corr_vector
        cost_matrix[idx] = dist
    return cost_matrix, fsc_matrix
