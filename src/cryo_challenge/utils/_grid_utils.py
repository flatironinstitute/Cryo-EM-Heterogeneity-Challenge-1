import torch
from torch import Tensor
from typing import Optional, Tuple


def _make1d_real_space(size, dx):
    return torch.fft.fftshift(torch.fft.fftfreq(size, 1 / dx)) * size


def _make1d_fourier_space(size, dx, outputs_rfftfreqs):
    if outputs_rfftfreqs:
        return torch.fft.rfftfreq(size, dx)
    else:
        return torch.fft.fftfreq(size, dx)


def _make_coordinates_or_frequencies_1d(
    size: int,
    grid_spacing: float,
    outputs_real_space: bool = False,
    outputs_rfftfreqs: Optional[bool] = None,
) -> Tensor:
    """One-dimensional coordinates in real or fourier space"""
    if outputs_real_space:
        make_1d_fn = _make1d_real_space
    else:
        if outputs_rfftfreqs is None:
            raise ValueError("Internal error in `cryojax.coordinates`.")
        else:
            make_1d_fn = _make1d_fourier_space

    return make_1d_fn(size, grid_spacing)


def _make_coordinates_or_frequencies(
    shape: tuple[int, ...],
    grid_spacing: float,
    outputs_real_space: bool = False,
    outputs_rfftfreqs: bool = True,
) -> Tensor:
    ndim = len(shape)
    coords1D = []
    for idx in range(ndim):
        if outputs_real_space:
            c1D = _make_coordinates_or_frequencies_1d(
                shape[idx], grid_spacing, outputs_real_space
            )
        else:
            if not outputs_rfftfreqs:
                rfftfreq = False
            else:
                rfftfreq = False if idx < ndim - 1 else True
            c1D = _make_coordinates_or_frequencies_1d(
                shape[idx], grid_spacing, outputs_real_space, rfftfreq
            )
        coords1D.append(c1D)

    if ndim == 3:
        z, y, x = coords1D
        xv, yv, zv = torch.meshgrid(x, y, z, indexing="xy")
        xv, yv, zv = [
            torch.permute(rv, dims=(2, 0, 1)) for rv in [xv, yv, zv]
        ]  # Change axis ordering to [z, y, x]
        coords = torch.stack([xv, yv, zv], axis=-1)
    else:
        raise ValueError(
            "Only 3D coordinate grids are supported. "
            f"Tried to create a grid of shape {shape}."
        )

    return coords


def make_frequency_grid(
    shape: tuple[int, ...],
    grid_spacing: float = 1.0,
    outputs_rfftfreqs: bool = True,
) -> Tensor:
    """Create a fourier-space cartesian coordinate system on a grid.
    The zero-frequency component is in the corner.

    **Arguments:**
    - `shape`:
        Shape of the grid, with `ndim = len(shape)`.
    - `grid_spacing`:
        The grid spacing (i.e. pixel/voxel size),
        in units of length.
    - `outputs_rfftfreqs`:
        Return a frequency grid for use with `jax.numpy.fft.rfftn`.
        `shape[-1]` is the axis on which the negative
        frequencies are omitted.

    **Returns:**

    A cartesian coordinate system in frequency space.
    """
    frequency_grid = _make_coordinates_or_frequencies(
        shape,
        grid_spacing=grid_spacing,
        outputs_real_space=False,
        outputs_rfftfreqs=outputs_rfftfreqs,
    )
    return frequency_grid


def make_radial_frequency_grid(
    shape: Tuple[int, int, int], voxel_size: float = 1.0
) -> Tensor:
    """
    Create a radially averaged frequency grid. If a voxel size is provided,
    the frequencies are given in inverse Angstroms.

    **Arguments:**
        shape: Shape of the grid to be computed.
        voxel_size: Size of the voxel in Angstroms.
    **Returns:**
        A tensor of shape (box_size, box_size, box_size//2 + 1) containing the radial frequencies in Angstroms.
    """
    frequency_grid_in_angstroms = make_frequency_grid(shape) / voxel_size
    return torch.norm(frequency_grid_in_angstroms, dim=-1)
