# take in two mrc files and return the cost

import torch
import mrcfile
import argparse
import glob
import os
from natsort import natsorted

from .vol_pca import spherical_mask
from .optimal_transport import compute_cost_ot_wrapper


def main():
    parser = argparse.ArgumentParser(
        description="Calculate cost between two mrc files."
    )
    parser.add_argument("fname", type=str, help="Path to first mrc file")
    parser.add_argument("dir", type=str, help="director mrc files")
    parser.add_argument(
        "mask_fname", default=None, type=str, nargs="?", help="mask file name"
    )
    parser.add_argument(
        "n_pix",
        default=None,
        type=int,
        nargs="?",
        help="number of pixels in each dimension",
    )
    parser.add_argument(
        "ot_method", default="emd2", type=str, nargs="?", help="ot cost method"
    )
    parser.add_argument(
        "n_pix_skip", default=6, type=int, nargs="?", help="number of pixels to skip"
    )
    parser.add_argument(
        "n_skip", default=1, type=int, nargs="?", help="chose every n_skip-th file"
    )
    parser.add_argument(
        "n_trunc",
        default=None,
        type=int,
        nargs="?",
        help="number of files to truncate to",
    )
    parser.add_argument(
        "--spherical_mask", action="store_true", help="make a spherical mask?"
    )
    parser.add_argument(
        "thresh",
        default=-float("inf"),
        type=float,
        help="map values lower than this will be clipped to zero",
    )
    args = parser.parse_args()

    n_pix_skip = args.n_pix_skip
    assert not (
        args.mask_fname is not None and args.spherical_mask
    ), "must provide only one of mask_fname or spherical_mask=True"

    if args.mask_fname is not None:
        mask = (
            mrcfile.open(args.mask_fname, "r")
            .data[::n_pix_skip, ::n_pix_skip, ::n_pix_skip]
            .copy()
            .astype(bool)
            .flatten()
        )
    elif args.spherical_mask:
        mask = spherical_mask(args.n_pix, rad=args.n_pix // 2).flatten()
    else:
        mask = None

    with mrcfile.open(args.fname, "r") as mrc:
        map1 = torch.from_numpy(
            mrc.data[::n_pix_skip, ::n_pix_skip, ::n_pix_skip].copy()
        )

    files = natsorted(glob.glob(os.path.join(args.dir, "*.mrc")))[:: args.n_skip][
        : args.n_trunc
    ]
    for fname2 in files:
        with mrcfile.open(fname2, "r") as mrc:
            map2 = torch.from_numpy(
                mrc.data[::n_pix_skip, ::n_pix_skip, ::n_pix_skip].copy()
            )

        cost = compute_cost_ot_wrapper(
            map1, map2, args.ot_method, mask=mask, numItermax=10**8, thresh=args.thresh
        )
        print(fname2, cost)


if __name__ == "__main__":
    main()
