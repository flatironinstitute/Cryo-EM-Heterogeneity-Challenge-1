import torch
from torch import Tensor
from torch.nn import Module


class WhiteningFilter(Module):
    array: Tensor

    def __init__(
        self,
        radially_averaged_powerspectrum_on_grid: Tensor,
        *,
        get_squared: bool = False,
    ):
        """**Arguments:**

        -  `radially_averaged_powerspectrum_on_grid`:
            The radially averaged power spectrum on a grid.
            This is the output of `compute_radially_averaged_powerspectrum`.
        - `interpolation_mode`:
            The method of interpolating the binned, radially averaged
            power spectrum onto a 3D grid. Either `nearest` or `linear`.
        - `get_squared`:
            If `False`, the whitening filter is the inverse square root of the volume
            power. If `True`, the filter is the inverse of the volume power.
        """

        self.array = _compute_whitening_filter(
            radially_averaged_powerspectrum_on_grid,
            get_squared=get_squared,
        )

    def __call__(self, volume):
        return volume * self.array


def _compute_whitening_filter(
    radially_averaged_powerspectrum_on_grid: Tensor,
    get_squared: bool = False,
) -> Tensor:
    # Compute inverse square root (or inverse square)
    inverse_fun = torch.reciprocal if get_squared else torch.rsqrt

    dtype = radially_averaged_powerspectrum_on_grid.dtype
    whitening_filter = torch.where(
        torch.isclose(
            radially_averaged_powerspectrum_on_grid, torch.tensor(0.0, dtype=dtype)
        ),
        torch.tensor(0.0, dtype=dtype),
        inverse_fun(radially_averaged_powerspectrum_on_grid),
    )
    return whitening_filter
