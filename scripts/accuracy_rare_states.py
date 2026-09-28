import os
import pickle

import hydra
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig
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


@hydra.main(version_base=None, config_path="config", config_name="accuracy_rare_states")
def main(cfg: DictConfig) -> None:
    gt_ordering = pd.read_csv(cfg.paths.metadata)
    idx_map = gt_ordering.sort_values("pc1").index.values
    frequencies = gt_ordering["populations_count"].values[idx_map]

    data_fnames = [os.path.join(cfg.paths.pkl_dir, f) for f in cfg.submissions]
    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    distance = cfg.distance
    threshold = cfg.threshold
    nrows, ncols, base_size = cfg.plot.nrows, cfg.plot.ncols, cfg.plot.base_size

    rng = np.random.default_rng(cfg.seed)
    n = len(frequencies)
    orderings = {"original": np.arange(n), "permuted": rng.permutation(n)}

    for random_label, random_idx in orderings.items():
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(base_size * ncols, base_size * nrows),
            sharey=False,
            squeeze=False,
        )
        for i, fname in enumerate(data_fnames[: nrows * ncols]):
            print(
                f"[{random_label}] Processing file {i + 1}/{len(data_fnames)}: {fname}"
            )
            with open(fname, "rb") as f:
                data = pickle.load(f)

            df = data[distance]["cost_matrix"]
            row_min = df.iloc[idx_map].values.min(axis=1)

            rare_values, frequent_values, statistic, pvalue = (
                ks_statistic_rare_versus_frequent(
                    frequencies[random_idx], row_min, threshold
                )
            )

            plot_row = i // ncols
            plot_col = i % ncols
            ax = axes[plot_row, plot_col]

            color = find_color(DEFAULT_PLOT_PARAMETERS, data)  # noqa: F841 (currently unused)

            legend = plot_row == 0 and plot_col == 0
            pd.Series(frequent_values).plot.kde(
                ax=ax, alpha=0.5, label="frequent", legend=legend
            )
            pd.Series(rare_values).plot.kde(
                ax=ax, alpha=0.5, label="rare", legend=legend, color="red"
            )

            label = data[distance]["user_submission_label"]
            if label in ice_cream_method_map:
                title = ice_cream_method_map[label] + "\n" + f"p={pvalue:.1e}"
            else:
                print(f"Label {label} not found in mapping, using original label.")
                title = label
            ax.set_title(title)

            if plot_col != 0:
                ax.set_yticklabels([])
                ax.set_ylabel("")
            else:
                ax.set_ylabel("# of Images")

            if plot_row == nrows - 1:
                ax.set_xlabel(
                    f"Volume-to-volume {distance.capitalize()} Distance\n"
                    "(minimum across GT volumes)"
                )
            else:
                ax.set_xlabel("")

        # Hide unused panels
        for j in range(len(data_fnames), nrows * ncols):
            axes[j // ncols, j % ncols].set_visible(False)

        fig.tight_layout()
        out_path = os.path.join(
            cfg.paths.output_dir, cfg.output_filename.format(label=random_label)
        )
        fig.savefig(out_path, dpi=cfg.plot.dpi)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
