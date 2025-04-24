import torch
from cryo_challenge.power_spectrum_utils import _centered_ifftn, compute_power_spectrum
from cryo_challenge.utils._bfactor_filters import (
    _compute_bfactor_filter,
    apply_bfactor_filter,
)


def test_compute_power_spectrum():
    """
    Test the computation of the power spectrum of a radially symmetric Gaussian volume.
    Since the volume is radially symmetric, the power spectrum of the whole volume should be
    approximately the power spectrum in a central slice. The computation is not exact as our
    averaging over shells is approximated.
    """
    box_size = 224
    volume_shape = (box_size, box_size, box_size)
    voxel_size = 1.073 * 2

    freq = torch.fft.fftshift(torch.fft.fftfreq(box_size, d=voxel_size))
    x = freq.clone()
    y = freq.clone()
    z = freq.clone()
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")

    s2 = x**2 + y**2 + z**2

    b_factor = 170

    gaussian_volume = torch.exp(-b_factor / 4 * s2).reshape(volume_shape)
    gaussian_volume = _centered_ifftn(gaussian_volume)

    power_spectrum = compute_power_spectrum(gaussian_volume)
    power_spectrum_slice = (
        torch.abs(torch.fft.fftn(gaussian_volume)[: box_size // 2, 0, 0]) ** 2
    )

    mean_squared_error = torch.mean((power_spectrum - power_spectrum_slice) ** 2)

    assert mean_squared_error < 1e-3

    return


def test_bfactor_normalize_volumes():
    """
    Similarly to the other test, we test the normalization of a radially symmetric volume.
    In this case we test with an oscillatory volume, which is a volume with a sinusoidal.
    Since both the b-factor correction volume and the volume are radially symmetric, the
    power spectrum of the normalized volume should be the same as the power spectrum of
    a normalized central slice
    """
    box_size = 128
    volume_shape = (box_size, box_size, box_size)
    voxel_size = 1.5

    freq = torch.fft.fftshift(torch.fft.fftfreq(box_size, d=voxel_size))
    x = freq.clone()
    y = freq.clone()
    z = freq.clone()
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")

    s2 = x**2 + y**2 + z**2

    oscillatory_volume = torch.sin(300 * s2).reshape(volume_shape)
    oscillatory_volume = _centered_ifftn(oscillatory_volume)
    bfactor_scaling_vol = _compute_bfactor_filter(-170, box_size, voxel_size)

    norm_oscillatory_vol = apply_bfactor_filter(
        oscillatory_volume, -170, voxel_size, in_place=False
    )

    ps_osci = torch.fft.fftn(oscillatory_volume, dim=(-3, -2, -1), norm="backward")[
        : box_size // 2, 0, 0
    ]
    ps_norm_osci = torch.fft.fftn(
        norm_oscillatory_vol, dim=(-3, -2, -1), norm="backward"
    )[: box_size // 2, 0, 0]
    ps_bfactor_scaling = torch.fft.fftshift(bfactor_scaling_vol)[: box_size // 2, 0, 0]

    ps_osci = torch.abs(ps_osci) ** 2
    ps_norm_osci = torch.abs(ps_norm_osci) ** 2
    ps_bfactor_scaling = torch.abs(ps_bfactor_scaling) ** 2

    assert torch.allclose(ps_norm_osci, ps_osci * ps_bfactor_scaling)
