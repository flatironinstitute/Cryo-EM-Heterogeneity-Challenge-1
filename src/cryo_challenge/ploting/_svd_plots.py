from typing import Tuple, Optional

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

from ._common_utilities import (
    sort_labels_manually,
    get_plot_parameters_for_labels,
    ABBREVIATIONS_FOR_LABELS,
    FORMATTED_LABELS_FOR_PLOTS,
)


def _set_default_plotting():
    sns.set_context("talk")
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["pdf.fonttype"] = 42  # TrueType fonts
    mpl.rcParams["ps.fonttype"] = 42
    return


def plot_pcv_vs_label(
    svd_pipeline_results: dict,
    ref_label: str,
    *,
    figsize: Tuple[int, int] = (10, 8),
    fontsize: int = 18,
    figure_filename: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the PCV of each label against a reference label. This reproduces Figure 2G in the paper.

    ** Arguments: **
        - svd_pipeline_results: dictionary with the results of the SVD pipeline
        - ref_label: reference label to compare against
        - figsize: size of the figure
        - fontsize: font size of the labels
        - figure_filename: if provided, saves the figure to this filename
    ** Returns: **
        - fig: matplotlib figure object
        - ax: matplotlib axes object
    """

    _set_default_plotting()
    sns.set_style("white")

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    pcv_matrix_results = svd_pipeline_results["capvar_distance_matrix_results"]
    labels = pcv_matrix_results["labels"]
    if "Ground Truth" in labels:
        labels = np.asarray(labels)[:-1]
    else:
        labels = np.asarray(labels)

    # extract PCV of each label vs the reference label
    idx_ref_label = np.where(labels == ref_label)[0][0]
    pcv_vs_ref = pcv_matrix_results["pcv_matrix"][idx_ref_label, :-1].numpy()

    # Sort labels and pcv values
    labels, sort_idxs = sort_labels_manually(labels)
    labels = labels[::-1]  # to plot from top to bottom
    pcv_vs_ref = pcv_vs_ref[sort_idxs][::-1]

    # Get plot parameters (color in this case)
    plot_setup = get_plot_parameters_for_labels(labels)
    colors = [plot_setup[label]["color"] for label in labels]

    ax.barh(np.arange(len(labels)), pcv_vs_ref, color=colors)

    # Reverse the order of the y-axis to invert
    ax.invert_yaxis()

    # Set other parameters
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels([0.0, 0.25, 0.5, 0.75, 1.0], fontsize=fontsize)
    ax.set_xlabel(f"PCV vs {FORMATTED_LABELS_FOR_PLOTS[ref_label]}", fontsize=fontsize)
    ax.set_xlim(0, 1.05)

    # Set up y-axis ticks and labels
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(
        [FORMATTED_LABELS_FOR_PLOTS[label] for label in labels], fontsize=fontsize
    )
    ax.set_ylim(-0.8, len(labels) - 0.2)

    if figure_filename is not None:
        fig.savefig(figure_filename, bbox_inches="tight", dpi=300)

    return fig, ax


def plot_pcv_matrix(
    svd_pipeline_results: dict,
    *,
    fontsize: int = 30,
    figure_filename: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the PCV matrix from the SVD pipeline results. This corresponds to Figure 2D in the paper.
    ** Arguments: **
        - svd_pipeline_results: dictionary containing the results of the SVD pipeline
        - fontsize: fontsize for the labels (default: 30)
        - figure_filename: filename to save the figure (default: None)
    ** Returns: **
        - fig: matplotlib figure object
        - ax: matplotlib axis object
    """

    _set_default_plotting()
    sns.set_style("white")

    distance_matrix_results = svd_pipeline_results["capvar_distance_matrix_results"]

    dist_matrix = distance_matrix_results["pcv_matrix"].numpy()
    labels = distance_matrix_results["labels"]
    labels, sort_idxs = sort_labels_manually(labels)

    labels_plot = [ABBREVIATIONS_FOR_LABELS[label] for label in labels]
    dist_matrix = dist_matrix[sort_idxs, :][:, sort_idxs]

    fig, ax = plt.subplots(figsize=(15, 15), layout="compressed")
    im = ax.imshow(dist_matrix, vmin=0, vmax=1)

    cbar = fig.colorbar(im, location="bottom")
    cbar.set_label(label="Percent. Captured Variance", fontsize=fontsize)
    cbar.set_ticks(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        labels=[0.0, 0.25, 0.5, 0.75, 1.0],
        fontsize=fontsize,
    )

    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xticks(
        ticks=np.arange(len(labels)), labels=labels_plot, rotation=90, fontsize=fontsize
    )
    ax.set_yticks(ticks=np.arange(len(labels)), labels=labels_plot, fontsize=fontsize)

    if figure_filename is not None:
        fig.savefig(figure_filename, bbox_inches="tight", dpi=300)
    return fig, ax


def plot_common_embedding(
    svd_pipeline_results: dict,
    *,
    n_cols: int,
    n_rows: int,
    principal_components: Tuple[int, int] = (0, 1),
    fontsize: int = 18,
    figsize: Tuple[int, int] = (18, 14),
    figure_filename: Optional[str] = None,
    flip_y: bool = False,
    flip_x: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot common embedding results from SVD pipeline. This corresponds to Figure 2B in the paper.

    ** Arguments: **
        - svd_pipeline_results: dictionary containing the results of the SVD pipeline
        - n_cols: number of columns in the plot grid
        - n_rows: number of rows in the plot grid
        - principal_components: tuple of principal components to plot (default: (0, 1))
        - fontsize: fontsize for the labels (default: 18)
        - figsize: size of the figure (default: (18, 14))
        - figure_filename: filename to save the figure (default: None)
        - flip_y: whether to flip the y-axis (default: False)
        - flip_x: whether to flip the x-axis (default: False)
    ** Returns: **
        - fig: matplotlib figure object
        - ax: matplotlib axis object array
    """
    _set_default_plotting()
    sns.set_style("whitegrid")

    flip_y = -1 if flip_y else 1
    flip_x = -1 if flip_x else 1

    common_embedding_results = svd_pipeline_results["common_embedding_results"]
    labels = list(common_embedding_results["common_embedding"].keys())

    populations = svd_pipeline_results["populations"]

    # extract labels amd get plot parameters
    plot_parameters = get_plot_parameters_for_labels(labels)

    # Sort labels and common embedding
    labels, _ = sort_labels_manually(labels)

    common_embedding = common_embedding_results["common_embedding"]
    all_embeddings = []
    pc1, pc2 = principal_components
    for label in labels:
        all_embeddings.append(common_embedding[label])
    all_embeddings = torch.cat(all_embeddings, dim=0)

    # Get weights for KDE plot
    weights = []
    for i in range(len(labels)):
        weights += populations[labels[i]].numpy().tolist()
    weights = torch.tensor(weights)

    fig, ax = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )
    if n_rows == 1 and n_cols == 1:
        ax = np.array([ax])

    for i in range(len(labels)):
        sns.kdeplot(
            x=flip_x * all_embeddings[:, pc1],
            y=flip_y * all_embeddings[:, pc2],
            cmap="gray",
            fill=False,
            cbar=False,
            ax=ax.flatten()[i],
            weights=weights,
            alpha=0.8,
            zorder=1,
        )

    for i in range(len(labels)):
        label = labels[i]
        pops = populations[label].numpy()

        ax.flatten()[i].scatter(
            x=flip_x * common_embedding[label][:, pc1],
            y=flip_y * common_embedding[label][:, pc2],
            color=plot_parameters[label]["color"],
            s=pops / pops.max() * 200,
            marker="o",
            linewidth=0.3,
            # edgecolor="black",
            label=f"{FORMATTED_LABELS_FOR_PLOTS[label]}",
            zorder=2,
        )

        for spine in ax.flatten()[i].spines.values():
            spine.set_edgecolor("none")

        ax.flatten()[i].set_xticks([])
        ax.flatten()[i].set_yticks([])
        ax.flatten()[i].set_axis_off()

        ax.flatten()[i].set_xlim((-0.21, 0.21))
        ax.flatten()[i].set_ylim((-0.21, 0.21))

        ax.flatten()[i].set_title(
            f"{FORMATTED_LABELS_FOR_PLOTS[label]}", fontsize=fontsize, y=0.7
        )

    for i in range(len(labels), n_cols * n_rows):
        ax.flatten()[i].axis("off")

    # adjust horizontal space
    plt.subplots_adjust(wspace=-0.1, hspace=0.0)

    if figure_filename is not None:
        fig.savefig(figure_filename, bbox_inches="tight", dpi=300)

    return fig, ax


def plot_projection_to_gt_embedding(
    svd_pipeline_results: dict,
    ref_label_for_populations: str = "Averaged GT 1",
    *,
    fontsize: int = 18,
    n_rows: int = 5,
    n_cols: int = 5,
    figsize: Tuple[int, int] = (25, 20),
    figure_filename: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot projection of each submission to the ground truth embedding, along with the populations.

    ** Arguments: **
        - svd_pipeline_results: dictionary containing the results of the SVD pipeline
        - ref_label_for_populations: label to use for the populations reference (default: "Averaged GT 1")
            This plots a population in the background for reference.
        - fontsize: fontsize for the labels (default: 18)
        - n_rows: number of rows in the plot grid (default: 5)
        - n_cols: number of columns in the plot grid (default: 5)
        - figsize: size of the figure (default: (25, 20))
        - figure_filename: filename to save the figure (default: None)
    ** Returns: **
        - fig: matplotlib figure object
        - ax: matplotlib axis object array
    """

    sns.set_style("white")
    plt.rcParams["axes.titley"] = 0.85

    embedding_gt = svd_pipeline_results["gt_embedding_results"]["gt_embedding"]
    sub_embedding_in_gt = svd_pipeline_results["gt_embedding_results"][
        "submission_embedding"
    ]
    populations = svd_pipeline_results["populations"]

    labels = list(sub_embedding_in_gt.keys())
    labels, _ = sort_labels_manually(labels)
    plot_parameters = get_plot_parameters_for_labels(labels)

    populations_ref = populations[ref_label_for_populations]
    embedding_gt = sub_embedding_in_gt[ref_label_for_populations]

    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)

    for i in range(len(labels)):
        label = labels[i]
        embedding = sub_embedding_in_gt[label]

        pops = populations[label]
        ax.flatten()[i].scatter(
            x=embedding[:, 0],
            y=pops / pops.max(),
            color=plot_parameters[label]["color"],
            marker="o",
            s=100,
            linewidth=0.3,
        )

        ax.flatten()[i].plot(
            embedding_gt[:, 0],
            populations_ref / populations_ref.max(),
            color="black",
            marker="*",
            linewidth=0.3,
            alpha=0.5,
        )

        ax.flatten()[i].set_title(
            f"{FORMATTED_LABELS_FOR_PLOTS[label]}",
            fontsize=fontsize,
            loc="center",
        )

        # set x label only for the last row
        if i >= n_rows:
            ax.flatten()[i].set_xlabel("PC 1", fontsize=fontsize)

        # set y label only for the first column
        if i % n_cols == 0:
            ax.flatten()[i].set_ylabel("Scaled probability", fontsize=fontsize)

        ax.flatten()[i].set_yticks(
            [0.25, 0.5, 0.75, 1.0, 1.30], [0.25, 0.5, 0.75, 1.0, ""], fontsize=fontsize
        )
        ax.flatten()[i].set_xticks([])

        plt.subplots_adjust(wspace=0.0, hspace=0.0)

        # remove the lines around the subplots
        for spine in ax.flatten()[i].spines.values():
            spine.set_edgecolor("none")

    if i < n_cols * n_rows:
        for j in range(i + 1, n_cols * n_rows):
            ax.flatten()[j].axis("off")

    if figure_filename is not None:
        fig.savefig(figure_filename, bbox_inches="tight", dpi=300)

    return fig, ax
