from typing import Optional, Tuple
import torch
from torch.fft import fftn, ifftn, fftshift, ifftshift


def fftn_center(
    input: torch.Tensor,
    s: Optional[Tuple[int]] = None,
    dim: Optional[Tuple[int]] = None,
    norm: Optional[str] = None,
    *,
    out=None,
) -> torch.Tensor:
    return fftshift(fftn(fftshift(input), s=s, dim=dim, norm=norm, out=out))


def ifftn_center(
    input: torch.Tensor,
    s: Optional[Tuple[int]] = None,
    dim: Optional[Tuple[int]] = None,
    norm: Optional[str] = None,
    *,
    out=None,
) -> torch.Tensor:
    return ifftshift(ifftn(ifftshift(input), s=s, dim=dim, norm=norm, out=out))


def htn_center(input: torch.Tensor, dim=None) -> torch.Tensor:
    output = fftn_center(input, norm="ortho", dim=dim)
    return output.real - output.imag
