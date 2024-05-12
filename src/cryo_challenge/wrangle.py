import numpy as np
import pandas as pd
import torch
import mrcfile
import os
import glob
from natsort import natsorted
import json
import argparse


def make_metadata_gt(fname, n_bins):
    """
    Bins continuous latent coordinates into discrete intervals,
      calculates the probability of each interval,
      and returns a dataframe with the volume names and their probabilities.

    Parameters
    ----------
    fname : str
        Path to the .npy numpy.ndarray latent coordinates file.
    n_bins : int
        Number of bins to use for the histogram.

    Returns
    -------
    metadata_gt_df : pandas.DataFrame
        Dataframe with the volume names and their probabilities.

    Notes
    -----
        visually inspect choice of n_bins with a histogram: pd.Series(pc1_coord).plot.hist(bins=n_bins)

    """
    idx_pc1 = 0
    pc1_coord = np.load(fname)[idx_pc1]
    hist, bin_edges = np.histogram(pc1_coord, bins=n_bins)
    interval_idx = np.digitize(pc1_coord, bin_edges[:-1])
    population = hist / hist.sum()
    metadata_gt_df = pd.merge(
        pd.DataFrame(
            {"populations": population, "interval_idx": np.arange(1, n_bins + 1)}
        ),
        pd.DataFrame(
            {
                "interval_idx": interval_idx,
                "pc1": pc1_coord.tolist(),
                "volumes": np.array(
                    [
                        "{:02}".format(idx) + ".mrc"
                        for idx in range(1, 1 + len(pc1_coord))
                    ]
                ),
            }
        ),
    )
    return metadata_gt_df


def make_meta_data_user(mrc_dir):
    fnames = glob.glob(os.path.join(mrc_dir, "*.mrc"))
    metadata_df = pd.DataFrame(
        {
            "volumes": [os.path.basename(fname) for fname in fnames],
            "populations": 1 / len(fnames),
        }
    )
    return metadata_df


def make_mock_user_submission_closeto_gt(
    fnames_gt_volumes,
    metadata_gt_df,
    outdir,
    n_pix=114,
    scale_volumes=0.00003,
    scale_populations=1 / 80 / 100,
    random_seed=42,
):
    subset_fnames = fnames_gt_volumes
    maps = torch.empty(len(subset_fnames), n_pix, n_pix, n_pix)

    for idx, fname in enumerate(subset_fnames):
        with mrcfile.open(fname, "r") as mrc:
            maps[idx] = torch.from_numpy(mrc.data.copy())

    # subset metadata for user submission
    metadata_df = pd.merge(
        pd.DataFrame({"volumes": [os.path.basename(fname) for fname in subset_fnames]}),
        metadata_gt_df,
    )
    metadata_df.rename(
        columns={"volumes": "original_volumes", "populations": "original_populations"},
        inplace=True,
    )
    metadata_df["volumes"] = ["{:02}".format(idx) + ".mrc" for idx in range(1, 80 + 1)]
    metadata_df["populations"] = (
        metadata_df["original_populations"] / metadata_df["original_populations"].sum()
    )
    torch.manual_seed(random_seed)
    metadata_df["populations"] += (
        scale_populations * torch.randn(len(metadata_df)).abs().numpy()
    )
    metadata_df["populations"] /= metadata_df["populations"].sum()
    metadata_df.to_csv(
        os.path.join(outdir, "metadata.csv"),
        columns=[
            "volumes",
            "populations",
            "original_volumes",
            "original_populations",
            "pc1",
        ],
        index=False,
    )

    # add noise to gt maps
    torch.manual_seed(random_seed)
    noise = torch.randn_like(maps)
    maps_noisy = maps + scale_volumes * noise

    # write maps to disk
    for idx in range(len(metadata_df)):
        with mrcfile.new(
            os.path.join(outdir, "{:02}.mrc".format(idx + 1)), overwrite=True
        ) as mrc:
            # Access the data and header information
            mrc.set_data(maps_noisy[idx].numpy())


def save_one_big_file(fnames, n_pix_skip=1, n_pix=114):
    _n_pix = n_pix // n_pix_skip
    assert n_pix % n_pix_skip == 0
    n_maps = len(fnames)

    subset_fnames = fnames[:n_maps]

    maps = torch.empty(len(subset_fnames), _n_pix, _n_pix, _n_pix)

    for idx, fname in enumerate(subset_fnames):
        if idx % 100 == 0 or len(subset_fnames) <= 100:
            print(f"idx = {idx}")
        with mrcfile.open(fname, "r") as mrc:
            maps[idx] = torch.from_numpy(
                mrc.data[::n_pix_skip, ::n_pix_skip, ::n_pix_skip].copy()
            )
    maps_flat = maps.flatten(start_dim=1, end_dim=-1)
    return maps_flat


def unique_gt(
    maps_gt_flat, unique_mapping_fname, pc_fname, mrc_dir_source, mrc_dir_dest
):
    with open(unique_mapping_fname) as f:
        unique_mapping = json.load(f)

    maps_unique_flat = torch.empty((len(unique_mapping), maps_gt_flat.shape[1]))

    mrcs = []
    hists = []

    idx_pc1 = 0
    pc1_coord = np.load(pc_fname)[idx_pc1]
    pc1s = []
    idx_unique = 0
    for k, v in unique_mapping.items():
        idx = v[0] - 1
        maps_unique_flat[idx_unique] = maps_gt_flat[idx]
        idx_unique += 1
        mrc_file_source = os.path.join(mrc_dir_source, "{}.mrc".format(int(idx + 1)))
        mrc_file_dest = os.path.join(mrc_dir_dest, "{:05}.mrc".format(int(idx + 1)))
        try:
            os.symlink(mrc_file_source, mrc_file_dest)
        except FileExistsError:
            print("File already exists?: {}".format(mrc_file_dest))
        basename = os.path.basename(mrc_file_dest)
        mrcs.append(basename)
        hists.append(len(v))
        pc1s.append(pc1_coord[idx])

    metadata_df = pd.DataFrame(
        {"volumes": mrcs, "populations_count": hists, "pc1": pc1s}
    )
    assert metadata_df["populations_count"].sum() == len(maps_gt_flat)
    metadata_df["populations"] = (
        metadata_df["populations_count"] / metadata_df["populations_count"].sum()
    )

    return maps_unique_flat, metadata_df


def parse_args():
    parser = argparse.ArgumentParser(description="Wrangle data")
    parser.add_argument(
        "--wrangle_gt",
        action="store_true",
        default=False,
        help="make metadata for ground truth",
    )
    parser.add_argument(
        "--do_save_one_big_file",
        action="store_true",
        default=False,
        help="Save one big .pt file",
    )
    parser.add_argument(
        "--wrangle_unique_gt",
        action="store_true",
        default=False,
        help="make metadata for unique ground truth",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.wrangle_gt:
        # fname = '/mnt/ceph/users/mastore/heterogeneity_challenge_thyroglobulin_2023/making_ground_truth_volumes/stack_reweighted_trajectory_PCs_not_symm.npy'
        # metadata_gt_df = make_metadata_gt(fname,n_bins=100)
        metadata_gt_df = pd.read_csv("submissions/ref_unique_npix114/metadata.csv")

        # make mock user submission
        fnames_gt_volumes = natsorted(
            glob.glob("submissions/ref_unique_npix114/[0-9]*.mrc")
        )

        do_random_subset = False
        random_seed_idx = 43
        if do_random_subset:
            torch.manual_seed(random_seed_idx)
            idx = torch.randperm(len(fnames_gt_volumes))
            subset_fnames_gt_volumes = np.array(fnames_gt_volumes)[
                idx.tolist()
            ].tolist()
        else:
            idx = torch.linspace(0, len(fnames_gt_volumes) - 1, 80).int()
            subset_volumes = (
                metadata_gt_df.sort_values("pc1", ascending=False, inplace=False)
                .loc[idx]
                .volumes.tolist()
            )
            subset_fnames_gt_volumes = [
                x for x in fnames_gt_volumes if os.path.basename(x) in subset_volumes
            ]

        user_outdir = f"submissions/ref_npix114_subset80_0scalevolumes_0scalepopulations_randomseed{random_seed_idx}"
        make_mock_user_submission_closeto_gt(
            subset_fnames_gt_volumes,
            metadata_gt_df,
            outdir=user_outdir,
            n_pix=114,
            scale_volumes=0,
            scale_populations=0,
            random_seed=random_seed_idx,
        )

    if args.do_save_one_big_file:
        fnames_gt = natsorted(
            glob.glob(
                "/mnt/home/gwoollard/ceph/repos/cryomethods_comparison_pipeline/submissions/ref_unique_npix224/[0-9]*.mrc"
            )
        )
        maps_gt_flat = save_one_big_file(fnames_gt, n_pix=224)
        torch.save(
            maps_gt_flat,
            "/mnt/home/gwoollard/ceph/repos/cryomethods_comparison_pipeline/submissions/ref_unique_npix224/maps_gt_flat.pt",
        )

        # fnames_gt = natsorted(glob.glob('/mnt/ceph/users/mastore/heterogeneity_challenge_thyroglobulin_2023/making_ground_truth_volumes/tailless_volumes_114x114/[0-9]*.mrc'))
        # maps_gt_flat = save_one_big_file(fnames_gt)
        # torch.save(maps_gt_flat,'submissions/ref_npix114/maps_gt_flat.pt')

        # # TODO: loop around all aligned submissions
        # user_outdir='submissions/dataset2_structura-Valentin_Peretroukhin/aligned'
        # fnames_user = natsorted(glob.glob(os.path.join(user_outdir,'[0-9]*.mrc')))
        # maps_user_flat = save_one_big_file(fnames_user)
        # torch.save(maps_user_flat,os.path.join(user_outdir,'maps_user_flat.pt'))

        # for user_outdir in [
        #         'dataset1_maps_dlan-Lan_Dangyen',
        #         'dataset1_structura-Valentin_Peretroukhin',
        #         'miro_3dva',
        #         'Results_HetSIREN_Heros_CGM/Maps',
        #         'Results_Zernike3Deep_DAVID_HERREROS_CALERO/Maps',
        #         'Set1_80clusters_AlongMostDominantFreedomDegree_MDSPACE_RVuillemot_SJonic/Set1_80AverageModels_and_TheirConversionIntoVolumes',
        #         'Set1_80clusters_AlongMostDominantFreedomDegree_MDSPACE_RVuillemot_SJonic/Set1_80ReconstructedVolumes_and_Populations',
        #         'set1_dynamight_johannes_schwab/volumes',
        #         'simons1_cryostar-Yuan_Jing'
        # ]:

        #     fnames_user = natsorted(glob.glob(os.path.join('submissions/set1', user_outdir,'aligned','[0-9]*.mrc')))
        #     maps_user_flat = save_one_big_file(fnames_user)
        #     torch.save(maps_user_flat,os.path.join('submissions/set1', user_outdir,'aligned','maps_user_flat.pt'))

    if args.wrangle_unique_gt:
        # TODO: refactor so only need to read in fnames that will go into unique
        unique_mapping_fname = "/mnt/ceph/users/mastore/heterogeneity_challenge_thyroglobulin_2023/making_ground_truth_volumes/unique_label_to_lines.json"
        maps_gt_flat_fname = "/mnt/home/gwoollard/ceph/repos/cryomethods_comparison_pipeline/submissions/ref_npix114/maps_gt_flat.pt"
        maps_gt_flat = torch.load(maps_gt_flat_fname)  # takes 20min.
        mrc_dir_source = "/mnt/ceph/users/mastore/heterogeneity_challenge_thyroglobulin_2023/making_ground_truth_volumes/old_tailless_volumes_114x114/"
        mrc_dir_dest = "/mnt/home/gwoollard/ceph/repos/cryomethods_comparison_pipeline/submissions/ref_unique_npix114/"
        pc_fname = "/mnt/ceph/users/mastore/heterogeneity_challenge_thyroglobulin_2023/making_ground_truth_volumes/stack_reweighted_trajectory_PCs_not_symm.npy"

        maps_unique_flat, metadata_df = unique_gt(
            maps_gt_flat, unique_mapping_fname, pc_fname, mrc_dir_source, mrc_dir_dest
        )

        torch.save(maps_unique_flat, os.path.join(mrc_dir_dest, "maps_gt_flat.pt"))
        metadata_df.to_csv(os.path.join(mrc_dir_dest, "metadata.csv"), index=False)


if __name__ == "__main__":
    main()
