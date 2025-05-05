"""
These functions are ported from the cryoJAX package

https://github.com/mjo22/cryojax
"""

import torch
from torch import Tensor
import math
from functools import partial
import numpy as np

from ..fft._fourier import rfftn
from ._grid_utils import make_radial_frequency_grid


def compute_radially_averaged_powerspectrum_on_grid(
    volume_stack: Tensor,
    voxel_size: float = 1.0,
    interpolation_mode: str = "linear",
    *,
    minimum_frequency: float = 0.0,
    maximum_frequency: float = math.sqrt(2) / 2,
    chunk_size: int = 20,
):
    """Compute the radially averaged power spectrum of a volume stack.
    The power spectrum is computed by taking the squared magnitude of the
    Fourier transform of the volume stack. The radially averaged power
    spectrum is computed by averaging the power spectrum over spherical
    shells in Fourier space. The result is interpolated onto a grid
    defined by `radial_frequency_grid`.

    **Arguments:**
    - `volume_stack`:
        A stack of volumes in real space. The shape of the stack is
        (n_volumes, z, y, x).
    - `voxel_size`:
        The voxel size of the volumes. If `radial_frequency_grid` is passed
        in inverse angstroms, this argument must be included.
    - `interpolation_mode`:
        If `"linear"`, evaluate the grid using linear
        interpolation. If `"nearest"`, use nearest-neighbor
        interpolation.
    - `minimum_frequency`:
        The minimum frequency to include in the power spectrum.
    - `maximum_frequency`:
        The maximum frequency to include in the power spectrum.
    - `chunk_size`:
        The chunk size used for vmapped operations.
    **Returns:**
    - `radially_averaged_powerspectrum_on_grid`:
        The average of the radially averaged power spectrums of the volumes in the stack
        interpolated onto a radial frequency grid.
    """

    radially_averaged_powerspectrum, radial_frequency_grid, frequency_bins = (
        _compute_averaged_powerspectrum_from_stack(
            volume_stack,
            voxel_size=voxel_size,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            chunk_size=chunk_size,
        )
    )
    radially_averaged_powerspectrum_on_grid = _interpolate_radial_average_on_grid(
        radially_averaged_powerspectrum,
        frequency_bins,
        radial_frequency_grid,
        interpolation_mode=interpolation_mode,
    )

    return radially_averaged_powerspectrum_on_grid


def compute_radially_averaged_powerspectrum(
    fourier_volume: Tensor,
    radial_frequency_grid: Tensor,
    voxel_size: float = 1.0,
    *,
    minimum_frequency: float = 0.0,
    maximum_frequency: float = math.sqrt(2) / 2,
) -> Tensor:
    """
    Compute the radially averaged power spectrum of a volume in fourier space.

    **Arguments:**
    - `fourier_volume`:
        A volume in fourier space, such as the output of `cryo_challenge.fft.rfftn`.
    - `radial_frequency_grid`:
        The radial frequency coordinate system of the fourier_volume.
    - `voxel_size`:
        The voxel size of the volume. If not provided, the power spectrum
        is computed in grid units.
    - `minimum_frequency`:
        The minimum frequency to include in the power spectrum.
    - `maximum_frequency`:
        The maximum frequency to include in the power spectrum.

    **Returns:**
    - `radially_averaged_powerspectrum`:
        The radially averaged power spectrum of the volume.
    - `frequency_bins`:
        The array of frequencies for which we have calculated the
        radially averaged power spectrum.
    """
    squared_fourier_amplitudes = (fourier_volume * fourier_volume.conj()).real
    frequency_bins = _make_radial_frequency_bins(
        fourier_volume.shape,
        minimum_frequency,
        maximum_frequency,
        voxel_size,
    )

    radially_averaged_powerspectrum = _compute_binned_radial_average(
        squared_fourier_amplitudes, radial_frequency_grid, frequency_bins
    )

    return radially_averaged_powerspectrum, frequency_bins


def compute_fourier_shell_correlation(
    fourier_volume_1: Tensor,
    fourier_volume_2: Tensor,
    radial_frequency_grid: Tensor,
    voxel_size: float = 1.0,
    *,
    minimum_frequency: float = 0.0,
    maximum_frequency: float = math.sqrt(2) / 2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute the fourier shell correlation for two voxel maps.

    **Arguments:**

    - `fourier_volume_1`:
        A volume in fourier space. It should be from the output of `cryo_challenge.fft.rfftn` (e.g.
        the zero-frequency component should be in the corner).
    - `fourier_volume_2`:
        A volume in fourier space. See documentation for `fourier_volume_1`
        for conventions.
    - `radial_frequency_grid`:
        The radial frequency coordinate system of the fourier_volume.
    - `voxel_size`:
        The voxel size of the volumes. If `radial_frequency_grid` is passed
        in inverse angstroms, this argument must be included.
    - `threshold`:
        The threshold at which to draw the distinction between input maps.
        By default, `threshold = 0.5` for two 'known' volumes according to
        the half-bit criterion. If using half-maps derived from ab initio
        refinements, set `threshold = 0.143` by convention.

    **Returns:**

    - `fsc_curve`:
        The fourier shell correlations as a function of `frequency_bins`.
    - `frequency_bins`:
        The array of frequencies for which we have calculated the
        correlations.
    - `frequency_threshold`:
        The frequencies at which the correlation drops below the
        specified threshold.

    !!! warning

        It is common to obtain a `frequency_threshold` given in inverse angstroms.
        This function achieves this behavior if the `voxel_size` argument is passed
        and the `radial_frequency_grid` argument is given in inverse angstroms.
    """  # noqa: E501

    assert fourier_volume_1.shape == fourier_volume_2.shape, (
        "The two volumes must have the same shape. "
        f"Volume 1 shape: {fourier_volume_1.shape}, Volume 2 shape: {fourier_volume_2.shape}"
    )

    assert fourier_volume_1.ndim == 3, "The two volumes must be 3D. "

    fsc_curve, frequency_bins = _compute_fourier_correlation(
        fourier_volume_1,
        fourier_volume_2,
        radial_frequency_grid,
        voxel_size,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )
    return fsc_curve, frequency_bins


def _compute_binned_radial_average(
    volume: Tensor,
    radial_coordinate_grid: Tensor,
    bins: Tensor,
) -> Tensor:
    digitized_radial_grid = torch.bucketize(radial_coordinate_grid, bins, right=False)

    binned_radial_average = torch.bincount(
        digitized_radial_grid.ravel(), weights=volume.ravel(), minlength=bins.numel()
    ) / torch.bincount(digitized_radial_grid.ravel(), minlength=bins.numel())

    if binned_radial_average.shape[0] > bins.shape[0]:
        binned_radial_average = binned_radial_average[:-1]
    return binned_radial_average


def _compute_averaged_powerspectrum_from_stack(
    volume_stack,
    voxel_size=1.0,
    *,
    minimum_frequency=0.0,
    maximum_frequency=math.sqrt(2) / 2,
    chunk_size=20,
):
    # Transform to fourier space
    fourier_volumes = rfftn(volume_stack, dim=(1, 2, 3))

    radial_frequency_grid = make_radial_frequency_grid(
        volume_stack.shape[1:], voxel_size=voxel_size
    )
    # Compute stack of powe r spectra
    compute_radially_averaged_powerspectrum_stack = torch.vmap(
        partial(
            compute_radially_averaged_powerspectrum,
            radial_frequency_grid=radial_frequency_grid,
            voxel_size=voxel_size,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
        ),
        in_dims=(0),
        out_dims=(0, 0),
        chunk_size=chunk_size,
    )

    radially_averaged_powerspectrum_stack, frequency_bins = (
        compute_radially_averaged_powerspectrum_stack(fourier_volumes)
    )
    frequency_bins = frequency_bins[0]
    radial_frequency_grid = radial_frequency_grid[0]

    # Take the mean over the stack
    radially_averaged_powerspectrum = torch.mean(
        radially_averaged_powerspectrum_stack, axis=0
    )

    return radially_averaged_powerspectrum, radial_frequency_grid, frequency_bins


def _compute_fourier_correlation(
    fourier_array_1: Tensor,
    fourier_array_2: Tensor,
    radial_frequency_grid: (Tensor),
    grid_spacing: float,
    minimum_frequency: float,
    maximum_frequency: float,
) -> tuple[Tensor, Tensor, Tensor]:
    # Compute FSC/FRC radially averaged 1D profile
    correlation_map = (
        (fourier_array_1 * fourier_array_2.conj())
        / (torch.abs(fourier_array_1) * torch.abs(fourier_array_2))
    ).real
    frequency_bins = _make_radial_frequency_bins(
        fourier_array_1.shape, minimum_frequency, maximum_frequency, grid_spacing
    )
    correlation_curve = _compute_binned_radial_average(
        correlation_map,
        radial_frequency_grid,
        frequency_bins,
    )

    return correlation_curve, frequency_bins


def _make_radial_frequency_bins(
    shape, minimum_frequency, maximum_frequency, pixel_size
):
    q_min, q_max = minimum_frequency, maximum_frequency
    q_step = 1.0 / max(*shape)
    n_bins = 1 + int((q_max - q_min) / q_step)
    return torch.linspace(q_min, q_max, n_bins) / pixel_size


def _interpolate_radial_average_on_grid(
    binned_radial_average: Tensor,
    bins: Tensor,
    radial_coordinate_grid: (Tensor),
    interpolation_mode: str = "linear",
) -> Tensor:
    """Interpolate a binned radially averaged profile onto a grid.

    **Arguments:**

    - `binned_radial_average`:
        The binned, radially averaged profile.
    - `bins`:
        Radial bins over which `binned_radial_average` is computed.
    - `radial_coordinate_grid`:
        Radial coordinate system of image or volume.
    - `interpolation_mode`:
        If `"linear"`, evaluate the grid using linear
        interpolation. If `"nearest"`, use nearest-neighbor
        interpolation.

    **Returns:**

    The `binned_radial_average` evaluated on the `radial_coordinate_grid`.
    """
    if interpolation_mode == "nearest":
        radial_average_on_grid = torch.take(
            binned_radial_average,
            torch.bucketize(radial_coordinate_grid, bins, right=False),
            mode="clip",
        )
    elif interpolation_mode == "linear":
        radial_average_on_grid = np.interp(
            radial_coordinate_grid.ravel().numpy(),
            bins.numpy(),
            binned_radial_average.numpy(),
        ).reshape(radial_coordinate_grid.shape)
        radial_average_on_grid = torch.from_numpy(radial_average_on_grid)
    else:
        raise ValueError(
            f"`interpolation_mode` = {interpolation_mode} not supported. Supported "
            "interpolation modes are 'nearest' or 'linear'."
        )
    return radial_average_on_grid
