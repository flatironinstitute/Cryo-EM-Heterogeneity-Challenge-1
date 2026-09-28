import pickle
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp

from cryo_challenge.plot_utils._common_utilities import DEFAULT_PLOT_PARAMETERS


def find_color(DEFAULT_PLOT_PARAMETERS, data):
    base_label = data["fsc"]["user_submission_label"]
    if base_label.startswith("Bubble Gum"):
        base_label = "Sampled GT"
    elif base_label.startswith("Mint Chocolate Chip"):
        base_label = "Averaged GT"
    for key in DEFAULT_PLOT_PARAMETERS:
        if base_label.startswith(key):
            color = DEFAULT_PLOT_PARAMETERS[key]["color"]
            break
    return color


def ks_statistic_rare_versus_frequent(frequencies, row_min, threshold):
    rare_idx = frequencies <= threshold
    frequent_idx = frequencies > threshold

    rare_values = row_min[rare_idx]
    frequent_values = row_min[frequent_idx]

    statistic, pvalue = ks_2samp(rare_values, frequent_values)

    return rare_values, frequent_values, statistic, pvalue


ice_cream_method_map = {
    "Pina Colada 1": r"CryoDRGN$^*$",
    "Cookie Dough 1": "3DFlex",
    "Cookie Dough 2": r"3DFlex$^*$",
    "Cherry 1": "Zernike3D + ZART",
    "Cherry 2": r"Zernike3D + ZART$^*$",
    "Chocolate Chip 1": "e2gmm",
    "Chocolate Chip 2": r"e2gmm$^*$^",
    "Chocolate 1": "HetSIREN",
    "Chocolate 2": r"HetSIREN$^*$",
    "Rocky Road 1": "ManifoldEM (Raw Images)",
    "Rocky Road 2": r"ManifoldEM (Raw Images)$^*$^",
    "Rocky Road 3": r"ManifoldEM (NLSA)$^*$^",
    "Black Raspberry 1": "Zernike3Deep",
    "Vanilla 1": "RECOVAR",
    "Vanilla 2": r"RECOVAR (deconvoluted)$^*$",
    "Mango 1": "3DVA",
    "Peanut Butter 1": "CryoSTAR",
    "Peanut Butter 2": r"CryoSTAR$^*$",
    "Neapolitan 1": "DynaMight",
    "Neapolitan 2": r"DynaMight$^*$",
    "Bubble Gum 2": "Sampled GT",
    "Mint Chocolate Chip 80": "Averaged GT",
}

if __name__ == "__main__":
    gt_ordering = pd.read_csv(
        "/mnt/home/gwoollard/ceph/repos/cryomethods_comparison_pipeline/osf/metadata.csv"
    )
    idx_map = gt_ordering.sort_values("pc1").index.values

    data_fnames = [
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_pina_colada_1.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_cookie_dough_1.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_cookie_dough_2.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_cherry_1.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_cherry_2.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_chocolate_chip_1.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_chocolate_chip_2.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_chocolate_1.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_chocolate_2.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_rocky_road_1.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_rocky_road_2.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_rocky_road_3.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_black_raspberry_1.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_vanilla_1.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_vanilla_2.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_mango_1.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_peanut_butter_1.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_peanut_butter_2.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_neapolitan_1.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_neapolitan_2.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_bubble_gum_2.pkl",
        "/mnt/home/smbp/ceph/smbpchallenge/map_to_map/set2/output/map_to_map_mint_chocolate_chip_80.pkl",
    ]

    nrows = 5
    ncols = 5
    base_size = 3
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(base_size * ncols, base_size * nrows), sharey=False
    )

    distance = "fsc"
    for i, fname in enumerate(data_fnames[: nrows * ncols]):
        print(f"Processing file {i+1}/{len(data_fnames)}: {fname}")
        with open(fname, "rb") as f:
            data = pickle.load(f)

        df = data[distance]["cost_matrix"]

        row_min = df.iloc[idx_map].values.min(axis=1)
        threshold = 5  # 0.00015

        frequencies = gt_ordering["populations_count"].values[
            idx_map
        ]  # df.index.values[idx_map]
        rare_values, frequent_values, statistic, pvalue = (
            ks_statistic_rare_versus_frequent(frequencies, row_min, threshold)
        )

        plot_row = i // ncols
        plot_col = i % ncols
        # if plot_row == 4:
        #     plot_col += 1
        color = find_color(DEFAULT_PLOT_PARAMETERS, data)
        # axes[plot_row, plot_col].scatter(x, y, marker='o', color=color, alpha=1)
        # axes[plot_row, plot_col].hexbin(x, y, gridsize=30, cmap='Blues', mincnt=1)
        # axes[plot_row, plot_col].scatter(k_lowest_pop[1], k_lowest_val[1], label='2nd lowest', marker='x')
        # axes[plot_row, plot_col].scatter(k_lowest_pop.mean(0), k_lowest_val.mean(0), label='average lowest', marker='s')

        # kde
        if plot_row == 0 and plot_col == 0:
            legend = True
        else:
            legend = False
        pd.Series(frequent_values).plot.kde(
            ax=axes[plot_row, plot_col], alpha=0.5, label="frequent", legend=legend
        )
        pd.Series(rare_values).plot.kde(
            ax=axes[plot_row, plot_col],
            alpha=0.5,
            label="rare",
            legend=legend,
            color="red",
        )
        # axes[plot_row, plot_col].scatter(df.index.values[idx_map], row_min, marker='o', color=color, alpha=0.5)
        new_label = data[distance]["user_submission_label"]
        try:
            new_label = (
                ice_cream_method_map[data[distance]["user_submission_label"]]
                + "\n"
                + f"p={pvalue:.1e}"
            )
        except Exception:
            print(
                f"Label {data[distance]['user_submission_label']} not found in mapping, using original label."
            )
            pass

        axes[plot_row, plot_col].set_title(new_label)
        # axes[plot_row, plot_col].legend()
        if not plot_col == 0:
            axes[plot_row, plot_col].set_yticklabels([])
            axes[plot_row, plot_col].set_ylabel("")
        else:
            axes[plot_row, plot_col].set_ylabel("# of Images")

        if plot_row == nrows - 1:
            # axes[plot_row, plot_col].set_xlabel('GT Population (number of images)')
            # axes[plot_row, plot_col].set_ylabel(f'Median of Min. {distance} Distances to Submission')
            axes[plot_row, plot_col].set_xlabel(
                f"Volume-to-volume {distance.capitalize()} Distance\n(minimum across GT volumes)"
            )
        else:
            axes[plot_row, plot_col].set_xlabel("")
            # axes[plot_row, plot_col].set_xticklabels([])

    for i in range(len(data_fnames), nrows * ncols):
        plot_row = i // ncols
        plot_col = i % ncols
        axes[plot_row, plot_col].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        "/mnt/home/smbp/ceph/smbpchallenge/plotting_round1_and_round2/output/accuracy_rare_states_distributions.pdf",
        dpi=300,
    )
