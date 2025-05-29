import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List, Dict, Union
import yaml
import argparse

from cryo_challenge.ploting.plotting_utils import COLORS
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
    do_sort=True,
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

    fig, axis = plt.subplots(
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

    for idx, (anonymous_label, data) in enumerate(data_d.items()):
        map2map_dist_matrix = data[metric]["cost_matrix"].iloc[gt_ordering].values
        # if do_sort:
        #     map2map_dist_matrix, _, _ = sort_by_transport(map2map_dist_matrix)

        if do_vmin_adjust and map2map_dist_matrix.min() < vmin:
            vmin = map2map_dist_matrix.min()
        if do_vmax_adjust and map2map_dist_matrix.max() > vmax:
            vmax = map2map_dist_matrix.max()

        ax = axis[idx // ncols, idx % ncols].imshow(
            map2map_dist_matrix, aspect="auto", cmap="Blues_r", vmin=vmin, vmax=vmax
        )

        axis[idx // ncols, idx % ncols].tick_params(
            axis="both", labelsize=smaller_fontsize
        )
        cbar = fig.colorbar(ax)
        cbar.ax.tick_params(labelsize=smaller_fontsize)
        plot_panel_label = anonymous_label
        axis[idx // ncols, idx % ncols].set_title(
            plot_panel_label, fontsize=smaller_fontsize
        )

    return fig, axis


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
    do_sort = False
    vmax = None
    fig, axis = plot_map_to_map_distances(
        data_d,
        gt_ordering,
        config.map_to_map_distance,
        nrows,
        ncols,
        suptitle=f"{config.map_to_map_distance} distance | do_sort={do_sort} | vmax={vmax}",
        do_sort=do_sort,
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
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(50, 50))

    fig.suptitle(suptitle, fontsize=30, y=0.95)
    alpha = 1
    linewidth = 3

    for idx_fname, (_, data) in enumerate(dist2dist_results_d.items()):
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

        if idx_fname // ncols == 0 and idx_fname % ncols == 0:
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

    return fig, axes


def distribution_to_distribution(config):
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

    with open(path_to_config, "r") as file:
        config = yaml.safe_load(file)
    config = PlottingConfig.from_dict(config)
    assert config.map_to_map_distance in AVAILABLE_MAP2MAP_DISTANCES.keys()
    map_to_map(config)
    distribution_to_distribution(config)
