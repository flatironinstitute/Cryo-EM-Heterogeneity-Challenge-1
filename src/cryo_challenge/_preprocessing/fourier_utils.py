import torch
from .crop_pad_utils import crop_vol_3d


@torch.no_grad()
def downsample_volume(volume: torch.Tensor, box_size_ds: int) -> torch.Tensor:
    """
    Downsample 3D volume in Fourier Space to specified box size

    Parameters:
    -----------
    volume (torch.Tensor): 3D volume
        shape: (im_x, im_y, im_z)
    box_size_ds (int): box size to downsample volume to

    Returns:
    --------
    vol_ds (torch.Tensor): downsampled 3D volume
    """
    vol_fft = torch.fft.fftshift(torch.fft.fftn(volume))
    crop_vol_fft = crop_vol_3d(vol_fft, box_size_ds)

    vol_ds = (
        torch.fft.ifftn(torch.fft.ifftshift(crop_vol_fft))
        * box_size_ds**3
        / volume.shape[-1] ** 3
    )

    return vol_ds.real


def downsample_submission(volumes: torch.Tensor, box_size_ds: int) -> torch.Tensor:
    """
    Downsample submission volumes in Fourier space to specified box size.

    Parameters:
    -----------
    volumes (torch.Tensor): submission volumes
        shape: (n_volumes, im_x, im_y, im_z)
    box_size_ds (int): box size to downsample volumes to

    Returns:
    --------
    volumes (torch.Tensor): downsampled submission volumes
    """
    if volumes.shape[-1] == box_size_ds:
        pass

    else:
        volumes_ds = torch.zeros(
            (volumes.shape[0], box_size_ds, box_size_ds, box_size_ds)
        )
        for i in range(volumes.shape[0]):
            volumes_ds[i] = downsample_volume(volumes[i], box_size_ds)
        volumes = volumes_ds

    return volumes
