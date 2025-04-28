"""for set 2"""

import argparse
import torch
import mrcfile


def main():
    parser = argparse.ArgumentParser(
        description="Average a set of 3D maps and save as MRC file."
    )
    parser.add_argument(
        "--fname", type=str, required=True, help="Path to input .pt file"
    )
    parser.add_argument(
        "--fname_out", required=True, type=str, help="Path to output .mrc file"
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


if __name__ == "__main__":
    main()
