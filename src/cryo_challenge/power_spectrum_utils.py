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


def _compute_power_spectrum_shell(index, volume, radii, shell_width=0.5):
    inner_diameter = shell_width + index
    outer_diameter = shell_width + (index + 1)
    mask = (radii > inner_diameter) & (radii < outer_diameter)
    return torch.sum(mask * volume) / torch.sum(mask)


def compute_power_spectrum(volume, shell_width=0.5):
    L = volume.shape[0]
    dtype = torch.float32
    radii = _grid_3d(L, dtype=dtype)["r"]

    # Compute centered Fourier transforms.
    vol_fft = torch.abs(_centered_fftn(volume)) ** 2

    power_spectrum = torch.vmap(
        _compute_power_spectrum_shell, in_dims=(0, None, None, None)
    )(torch.arange(0, L // 2), vol_fft, radii, shell_width)
    return power_spectrum