from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.fft import fftshift, ifftshift


def fftn_center(
    input: Tensor,
    s: Optional[Tuple[int]] = None,
    dim: Optional[Tuple[int]] = None,
    norm: Optional[str] = None,
    *,
    out=None,
) -> Tensor:
    """
    A centered version of `torch.fft.fftn`. For inputs
    `torch.fft.fftn`.

    **Returns:**
        The equivalent of running `fftshift(fftn(fftshift))` on the input
        tensor.
    """
    return fftshift(torch.fft.fftn(fftshift(input), s=s, dim=dim, norm=norm, out=out))


def ifftn_center(
    input: Tensor,
    s: Optional[Tuple[int]] = None,
    dim: Optional[Tuple[int]] = None,
    norm: Optional[str] = None,
    *,
    out=None,
) -> Tensor:
    """
    A centered version of `torch.fft.ifftn`. For inputs
    `torch.fft.ifftn`.

    **Returns:**
        The equivalent of running `ifftshift(ifftn(ifftshift))` on the input
        tensor.
    """
    return ifftshift(
        torch.fft.ifftn(ifftshift(input), s=s, dim=dim, norm=norm, out=out)
    )


def htn_center(input: Tensor, dim: Optional[Tuple[int]] = None) -> Tensor:
    """
    Centered Hartley transform. The Hartley transform closely related
    to the Fourier transform, but it is defined from real-valued
    inputs to real-valued outputs. The Hartley transform is defined
    as:
    $$ H(x) = F(x) + F^*(x) $$
    where $F(x)$ is the Fourier transform of $x$.

    The inverse Hartley transform is itself.

    **Arguments:**
        input: Tensor
            The input tensor to be transformed.
        dim: Tuple[int]
            The dimensions along which to compute the Hartley transform.
            If None, the transform is computed over all dimensions.

    **Returns:**
        output: Tensor
            The Hartley transform of the input tensor.

    """
    output = fftn_center(input, norm="ortho", dim=dim)
    return output.real - output.imag


def ifftn(
    ft: Tensor,
    s: Optional[tuple[int, ...]] = None,
    dim: Optional[tuple[int, ...]] = None,
) -> Tensor:
    """The equivalent of `torch.fft.ifftn` in `cryo_challenge` conventions.

    **Arguments:**
    ft :
        Fourier transform array. Assumes that the zero
        frequency component is in the corner.

    **Returns:**
    ift :
        Inverse fourier transform.
    """
    ift = fftshift(torch.fft.ifftn(ft, s=s, dim=dim), dim=dim)

    return ift


def fftn(
    ift: Tensor,
    s: Optional[tuple[int, ...]] = None,
    dim: Optional[tuple[int, ...]] = None,
) -> Tensor:
    """The equivalent of `torch.fft.fftn` in `cryo_challenge` conventions.

    **Arguments:**
    ift :
        Array in real space. Assumes that the zero
        frequency component is in the center.

    **Returns:**
    ft :
        Fourier transform of array.
    """
    ft = torch.fft.fftn(torch.fft.ifftshift(ift, dim=dim), s=s, dim=dim)

    return ft


def irfftn(
    ft: Tensor,
    s: Optional[tuple[int, ...]] = None,
    dim: Optional[tuple[int, ...]] = None,
) -> Tensor:
    """The equivalent of `torch.fft.irfftn` in `cryo_challenge` conventions.

    **Arguments:**
    ft :
        Fourier transform array. Assumes that the zero
        frequency component is in the corner.

    **Returns:**
    ift :
        Inverse fourier transform.
    """
    ift = torch.fft.fftshift(torch.fft.irfftn(ft, s=s, dim=dim), dim=dim)

    return ift


def rfftn(
    ift: Tensor,
    s: Optional[tuple[int, ...]] = None,
    dim: Optional[tuple[int, ...]] = None,
) -> Tensor:
    """The equivalent of `torch.fft.rfftn` in `cryo_challenge` conventions.

    **Arguments:*
    ift :
        Array in real space.

    **Returns:**
    ft :
        Fourier transform of array.
    """
    ft = torch.fft.rfftn(torch.fft.ifftshift(ift, dim=dim), s=s, dim=dim)

    return ft
