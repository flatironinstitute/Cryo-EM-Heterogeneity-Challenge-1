"""
Power spectrum normalization and required utility functions
"""

import torch


def _cart2sph(x, y, z):
    """
    Converts a grid in cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x: torch.tensor
        x-coordinate of the grid.
    y: torch.tensor
        y-coordinate of the grid.
    z: torch.tensor
    """
    hxy = torch.hypot(x, y)
    r = torch.hypot(hxy, z)
    el = torch.atan2(z, hxy)
    az = torch.atan2(y, x)
    return az, el, r


def _grid_3d(n, dtype=torch.float32):
    start = -n // 2 + 1
    end = n // 2

    if n % 2 == 0:
        start -= 1 / 2
        end -= 1 / 2

    grid = torch.linspace(start, end, n, dtype=dtype)
    z, x, y = torch.meshgrid(grid, grid, grid, indexing="ij")

    phi, theta, r = _cart2sph(x, y, z)

    # TODO: Should this theta adjustment be moved inside _cart2sph?
    theta = torch.pi / 2 - theta

    return {"x": x, "y": y, "z": z, "phi": phi, "theta": theta, "r": r}


def _centered_fftn(x, dim=None):
    x = torch.fft.fftn(x, dim=dim)
    x = torch.fft.fftshift(x, dim=dim)
    return x


def _centered_ifftn(x, dim=None):
    x = torch.fft.fftshift(x, dim=dim)
    x = torch.fft.ifftn(x, dim=dim)
    return x


def _compute_power_spectrum_shell(index, volume, radii):
    inner_diameter = 0.5 + index
    outer_diameter = 0.5 + (index + 1)
    mask = (radii > inner_diameter) & (radii < outer_diameter)
    return torch.norm(mask * volume) ** 2


def compute_power_spectrum(volume):
    L = volume.shape[0]
    dtype = torch.float32
    radii = _grid_3d(L, dtype=dtype)["r"]

    # Compute centered Fourier transforms.
    vol_fft = _centered_fftn(volume)

    power_spectrum = torch.vmap(_compute_power_spectrum_shell, in_dims=(0, None, None))(
        torch.arange(0, L // 2), vol_fft, radii
    )
    return power_spectrum


def normalize_power_spectrum(volumes, power_spectrum_ref):
    L = volumes.shape[-1]
    dtype = torch.float32
    radii = _grid_3d(L, dtype=dtype)["r"]

    # Compute centered Fourier transforms.
    vols_fft = _centered_fftn(volumes, dim=(1, 2, 3))

    inner_diameter = 0.5
    for i in range(0, L // 2):
        # Compute ring mask
        outer_diameter = 0.5 + (i + 1)
        ring_mask = (radii > inner_diameter) & (radii < outer_diameter)

        power_spectrum = torch.norm(ring_mask[None, ...] * vols_fft)

        vols_fft[:, ring_mask] = (
            vols_fft[:, ring_mask] / (power_spectrum + 1e-5) * power_spectrum_ref[i]
        )

        # # Update ring
        inner_diameter = outer_diameter

    return _centered_ifftn(vols_fft, dim=(1, 2, 3)).real
