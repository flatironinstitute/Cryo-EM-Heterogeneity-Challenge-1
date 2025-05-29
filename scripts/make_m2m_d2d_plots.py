import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List, Dict, Union
import yaml
import argparse
import glob
from natsort import natsorted
import warnings

from cryo_challenge.ploting.plotting_utils import COLORS, argsort_labels_manually
from cryo_challenge.map_to_map.map_to_map_pipeline import AVAILABLE_MAP2MAP_DISTANCES


@dataclass_json
@dataclass
class PlottingConfig:
    gt_metadata: str
    map2map_results: List[str]
    dist2dist_results: Dict[str, Union[str, List[str]]]
    map_to_map_distance: str
    output_paths: Dict[str, Dict[str, str]]


def plot_map_to_map_distances(
    data_d,
    gt_ordering,
    metric,
    nrows,
    ncols,
    figsize=None,
    suptitle=None,
    dpi=None,
    vmin=None,
    vmax=None,
):
    smaller_fontsize = 20
    larger_fontsize = 30
    n_plts = nrows * ncols
    if vmax is None:
        vmax = -np.inf
        do_vmax_adjust = True
    else:
        do_vmax_adjust = False
    if vmin is None:
        do_vmin_adjust = True
        vmin = np.inf
    else:
        do_vmin_adjust = False

    kwargs = {}
    if figsize is not None:
        kwargs["figsize"] = figsize
    else:
        kwargs["figsize"] = (10 * ncols, 10 * nrows)
    if dpi is not None:
        kwargs["dpi"] = dpi
    else:
        pass

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=n_plts // nrows,
        **kwargs,
    )

    if suptitle is None:
        fig.suptitle("d_{:}".format(metric), y=0.95, fontsize=larger_fontsize)
    elif isinstance(suptitle, str):
        fig.suptitle(suptitle, y=0.95, fontsize=larger_fontsize)
    else:
        raise ValueError("suptitle must be a string or None")

    available_labels = np.array(list(data_d.keys()))
    ordered_labels = available_labels[argsort_labels_manually(available_labels)]

    for idx, anonymous_label in enumerate(ordered_labels):
        data = data_d[anonymous_label]
        map2map_dist_matrix = data[metric]["cost_matrix"].iloc[gt_ordering].values

        if do_vmin_adjust and map2map_dist_matrix.min() < vmin:
            vmin = map2map_dist_matrix.min()
        if do_vmax_adjust and map2map_dist_matrix.max() > vmax:
            vmax = map2map_dist_matrix.max()

        ax = axes[idx // ncols, idx % ncols].imshow(
            map2map_dist_matrix, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax
        )
        for spine in axes[idx // ncols, idx % ncols].spines.values():
            ice_cream_no_version = [
                colour for colour in COLORS.keys() if anonymous_label.startswith(colour)
            ]
            single = ice_cream_no_version[
                np.argmax([len(single) for single in ice_cream_no_version])
            ]
            if len(ice_cream_no_version) > 1:
                warnings.warn(
                    f"Multiple ice cream flavours found that auto-match the label {anonymous_label}: {ice_cream_no_version}. Choosing the longest one: {single}",
                    UserWarning,
                )
            spine.set_edgecolor(COLORS[single])
            spine.set_linewidth(10)  # Optional: set thickness

        axes[idx // ncols, idx % ncols].tick_params(
            axis="both", labelsize=smaller_fontsize
        )
        cbar = fig.colorbar(ax)
        cbar.ax.tick_params(labelsize=smaller_fontsize)
        plot_panel_label = anonymous_label
        axes[idx // ncols, idx % ncols].set_title(
            plot_panel_label, fontsize=smaller_fontsize
        )
        if idx // ncols == nrows - 1 and idx % ncols == 0:
            axes[idx // ncols, idx % ncols].set_xlabel("Submission index", fontsize=30)
            axes[idx // ncols, idx % ncols].set_ylabel(
                "Ground truth index", fontsize=30
            )
        else:
            axes[idx // ncols, idx % ncols].set_xlabel("")
            axes[idx // ncols, idx % ncols].set_ylabel("")
            axes[idx // ncols, idx % ncols].set_xticks([])
            axes[idx // ncols, idx % ncols].set_yticks([])

    return fig, axes


def get_m2m_distances(fnames, map2map_distance):
    data_d = {}
    for fname in fnames:
        if fname not in data_d.keys():
            with open(fname, "rb") as f:
                data = pickle.load(f)
                if map2map_distance in data.keys():
                    anonymous_label = data[map2map_distance]["user_submission_label"]

                    data_d[anonymous_label] = data
    return data_d


def get_dist2dist_results(fnames):
    data_d = {}
    for fname in fnames:
        if fname not in data_d.keys():
            with open(fname, "rb") as f:
                data = pickle.load(f)
                anonymous_label = data["id"]
                data_d[anonymous_label] = data

    return data_d


def map_to_map(config):
    metadata_df = pd.read_csv(config.gt_metadata)
    metadata_df.sort_values("pc1", inplace=True)
    gt_ordering = metadata_df.index.tolist()
    data_d = get_m2m_distances(config.map2map_results, config.map_to_map_distance)
    nrows, ncols = 5, 5
    vmax = None
    fig, axis = plot_map_to_map_distances(
        data_d,
        gt_ordering,
        config.map_to_map_distance,
        nrows,
        ncols,
        suptitle=f"{config.map_to_map_distance} distance | vmax={vmax}",
        vmax=vmax,
    )

    # save the figure
    fig.savefig(
        config.output_paths["map_to_map"]["distance_matrices"],
        bbox_inches="tight",
        dpi=300,
    )


def plot_q_opt_distances(
    dist2dist_results_d, metric, suptitle, nrows, ncols, window_size, COLORS=None
):
    available_labels = np.array(list(dist2dist_results_d.keys()))
    ordered_labels = available_labels[argsort_labels_manually(available_labels)]

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(50, 50))

    fig.suptitle(suptitle, fontsize=30, y=0.95)
    alpha = 1
    linewidth = 3

    for idx_fname, label in enumerate(ordered_labels):
        data = dist2dist_results_d[label]

        axes[idx_fname // ncols, idx_fname % ncols].plot(
            data["user_submitted_populations"],
            color="black",
            label="submited",
            lw=linewidth,
        )
        axes[idx_fname // ncols, idx_fname % ncols].set_title(data["id"], fontsize=30)

        def window_q(q_opt, window_size):
            running_avg = np.convolve(
                q_opt, np.ones(window_size) / window_size, mode="same"
            )
            return running_avg

        windowed_q = np.zeros(
            (data["config"]["replicate_params"]["n_replicates"],)
            + data["user_submitted_populations"].shape
        )
        for replicate_idx in range(data["config"]["replicate_params"]["n_replicates"]):
            if replicate_idx == 0:
                label_d = {
                    "EMD": "EMD",
                    "KL": "KL",
                    "KL_raw": "Unwindowed",
                    "EMD_raw": "Unwindowed",
                }
            else:
                label_d = {"EMD": None, "KL": None, "KL_raw": None, "EMD_raw": None}
            windowed_q[replicate_idx] = window_q(
                data[metric]["replicates"][replicate_idx]["EMD"]["q_opt"], window_size
            )

        windowed_q_mean = windowed_q.mean(axis=0)
        windowed_q_std = windowed_q.std(axis=0)

        color = "blue"
        custom_color = False
        if COLORS is not None:
            for possible_id in COLORS.keys():
                if data["id"].startswith(possible_id):
                    color = COLORS[possible_id]
                    custom_color = True
                    break

        axes[idx_fname // ncols, idx_fname % ncols].plot(
            windowed_q_mean,
            color=color,
            alpha=alpha,
            label=label_d["EMD"],
            lw=linewidth,
        )

        plotting_style = "fill_between"
        if plotting_style == "errorbars":
            # Plot the data with error bars
            axes[idx_fname // ncols, idx_fname % ncols].errorbar(
                np.arange(len(windowed_q_mean)),  # X values
                windowed_q_mean,  # Y values (mean)
                yerr=windowed_q_std,  # Error bars (standard deviation)
                fmt="none",  # Format of the data points
                color="gray",  # Color of the data points and error bars
                alpha=1,  # Transparency
                capsize=0,  # Size of the caps on the error bars
            )
        elif plotting_style == "fill_between":
            if not custom_color:
                color = "gray"
            # Plot the data with a shaded region
            axes[idx_fname // ncols, idx_fname % ncols].fill_between(
                np.arange(len(windowed_q_mean)),  # X values
                windowed_q_mean - windowed_q_std,  # Lower bound of the shaded region
                windowed_q_mean + windowed_q_std,  # Upper bound of the shaded region
                color=color,  # Color of the shaded region
                alpha=0.5,  # Transparency
            )

        if idx_fname // ncols == nrows - 1 and idx_fname % ncols == 0:
            axes[idx_fname // ncols, idx_fname % ncols].set_xlabel(
                "Submission index", fontsize=30
            )
            axes[idx_fname // ncols, idx_fname % ncols].set_ylabel(
                "Population", fontsize=30
            )
        else:
            axes[idx_fname // ncols, idx_fname % ncols].set_xlabel("")
            axes[idx_fname // ncols, idx_fname % ncols].set_ylabel("")
        axes[idx_fname // ncols, idx_fname % ncols].tick_params(
            axis="both", labelsize=20
        )

    for ax in axes.flat:
        ax.tick_params(
            left=False,
            bottom=False,
            right=False,
            top=False,
            labelleft=False,
            labelbottom=False,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

    return fig, axes


def distribution_to_distribution_optimal_probability(config):
    fname = config.dist2dist_results["pkl_fnames"][0]

    with open(fname, "rb") as f:
        data = pickle.load(f)

    window_size = 5
    nrows, ncols = 5, 5
    suptitle = f"Submitted populations vs optimal populations \n {config.map_to_map_distance} distance (no rank) | n_replicates={data['config']['replicate_params']['n_replicates']} | window_size={window_size} | n_pool_ground_truth_microstates={data['config']['replicate_params']['n_pool_ground_truth_microstates']}"

    dist2dist_results_d = get_dist2dist_results(config.dist2dist_results["pkl_fnames"])

    fig, axes = plot_q_opt_distances(
        dist2dist_results_d,
        config.map_to_map_distance,
        suptitle,
        nrows,
        ncols,
        window_size,
        COLORS,
    )
    fig.savefig(
        config.output_paths["distribution_to_distribution"][
            "optimal_prob_plot_outpath"
        ],
        dpi=300,
    )


def wragle_pkl_to_dataframe(pkl_globs, metric):
    fnames = []
    for fname_glob in pkl_globs:
        fnames.extend(glob.glob(fname_glob))

    fnames = natsorted(fnames)

    df_list = []
    n_replicates = 30  # TODO: automate

    for fname in fnames:
        with open(fname, "rb") as f:
            data = pickle.load(f)

        df_list.append(
            pd.DataFrame(
                {
                    "EMD_opt": [
                        data[metric]["replicates"][i]["EMD"]["EMD_opt"]
                        for i in range(n_replicates)
                    ],
                    "EMD_submitted": [
                        data[metric]["replicates"][i]["EMD"]["EMD_submitted"]
                        for i in range(n_replicates)
                    ],
                    "klpq_opt": [
                        data[metric]["replicates"][i]["KL"]["klpq_opt"]
                        for i in range(n_replicates)
                    ],
                    "klqp_opt": [
                        data[metric]["replicates"][i]["KL"]["klqp_opt"]
                        for i in range(n_replicates)
                    ],
                    "klpq_submitted": [
                        data[metric]["replicates"][i]["KL"]["klpq_submitted"]
                        for i in range(n_replicates)
                    ],
                    "klqp_submitted": [
                        data[metric]["replicates"][i]["KL"]["klqp_submitted"]
                        for i in range(n_replicates)
                    ],
                    "id": data["id"],
                    "n_pool_ground_truth_microstates": data["config"][
                        "replicate_params"
                    ]["n_pool_ground_truth_microstates"],
                }
            )
        )

    df = pd.concat(df_list)
    df["EMD_opt_norm"] = df["EMD_opt"] / df["n_pool_ground_truth_microstates"]
    df["EMD_submitted_norm"] = (
        df["EMD_submitted"] / df["n_pool_ground_truth_microstates"]
    )

    return df


def get_d2d_emd_opt_results(pkl_globs, map_to_map_distance):
    df_fsc = wragle_pkl_to_dataframe(pkl_globs, map_to_map_distance)
    df = df_fsc[
        ["id", "EMD_opt", "EMD_submitted", "EMD_opt_norm", "EMD_submitted_norm"]
    ]
    df_average = df.groupby(["id"]).mean().reset_index()
    df_std = (
        df.groupby(["id"])
        .std()
        .reset_index()
        .filter(["EMD_opt_norm", "EMD_submitted_norm", "id"])
        .rename(
            columns={
                "EMD_opt_norm": "EMD_opt_norm_std",
                "EMD_submitted_norm": "EMD_submitted_norm_std",
            }
        )
    )

    df_average_and_error = pd.merge(df_average, df_std, on="id")

    def match_color(id, COLORS):
        for possible_id in COLORS.keys():
            if id.startswith(possible_id):
                return COLORS[possible_id]
        return "black"

    match_color("Mint Chocolate Chip 1", COLORS)

    df_average_and_error["colours"] = df_average_and_error["id"].apply(
        lambda x: match_color(x, COLORS)
    )

    sorted_idx = argsort_labels_manually(df_average_and_error.id.tolist())
    df_sorted = df_average_and_error.iloc[sorted_idx]
    return df_sorted


def distribution_to_distribution_optimal_emd(config):
    df_sorted = get_d2d_emd_opt_results(
        config.dist2dist_results["pkl_globs"], config.map_to_map_distance
    )  # TODO: skip this step if already done
    df_sorted.to_csv(
        config.output_paths["distribution_to_distribution"]["optimal_emd_data_outpath"]
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    # Set position for each bar
    indices = np.arange(len(df_sorted))
    bar_width = 0.4

    # Plot EMD_opt
    _ = ax.barh(
        indices - bar_width / 2,
        df_sorted["EMD_opt"],
        height=bar_width,
        color=df_sorted["colours"],
        label="EMD_opt",
        alpha=0.5,
    )

    # Plot EMD_submitted
    _ = ax.barh(
        indices + bar_width / 2,
        df_sorted["EMD_submitted"],
        height=bar_width,
        color=df_sorted["colours"],
        label="EMD_submitted",
    )

    # Set y-axis to show ID labels
    ax.set_yticks(indices)
    ax.set_yticklabels(df_sorted["id"])

    # Labels and title
    ax.set_xlabel("EMD Value")
    ax.set_ylabel("Submission")
    ax.set_title(
        "EMD between Ground Truth and Submission (with and without optimized populations)"
    )
    # ax.legend()

    fig.tight_layout()

    fig.savefig(
        config.output_paths["distribution_to_distribution"]["optimal_emd_plot_outpath"],
        bbox_inches="tight",
        dpi=300,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make map to map and distribution to distribution plots."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file for plotting.",
    )
    args = parser.parse_args()
    path_to_config = args.config
    # path_to_config = "/mnt/home/smbp/ceph/smbpchallenge/plotting_round1_and_round2/config_plotting_fsc_20250527.yaml"

    with open(path_to_config, "r") as file:
        config = yaml.safe_load(file)
    config = PlottingConfig.from_dict(config)
    assert config.map_to_map_distance in AVAILABLE_MAP2MAP_DISTANCES.keys()
    map_to_map(config)
    distribution_to_distribution_optimal_probability(config)
    distribution_to_distribution_optimal_emd(config)
