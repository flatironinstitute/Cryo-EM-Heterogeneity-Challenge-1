"""For set 2 (synthetic) mock ground truth"""

import argparse
import torch
import mrcfile


def main():
    parser = argparse.ArgumentParser(
        description="Average a set of 3D maps and save as MRC file."
    )
    parser.add_argument(
        "--fname",
        type=str,
        required=True,
        help="Path to input .pt file of 3D maps as an array without any dictionary keys (flattened ground truth maps).",
    )
    parser.add_argument(
        "--fname_out", required=True, type=str, help="Path to output .mrc file"
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        required=True,
        help="Voxel size in Angstroms.",
        default=2.146,
    )
    args = parser.parse_args()

    # open mmap .pt
    maps = torch.load(args.fname, mmap=True, weights_only=False)
    assert maps.ndim == 2, "expected flannened 3D maps"
    npix = int(round(maps.shape[-1] ** (1 / 3)))

    # average
    maps = maps.mean(dim=0).reshape(npix, npix, npix)

    # write to .mrc
    with mrcfile.new(args.fname_out, overwrite=True) as mrc:
        mrc.set_data(maps.numpy())
        mrc.set_volume()
        mrc.voxel_size = (args.voxel_size, args.voxel_size, args.voxel_size)


if __name__ == "__main__":
    main()
