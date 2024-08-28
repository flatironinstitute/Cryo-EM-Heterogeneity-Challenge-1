import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np


def plot_distance_matrix(dist_matrix, labels, title="", save_path=None):
    fig, ax = plt.subplots()
    cax = ax.matshow(dist_matrix, cmap="viridis")
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=90)
    ax.set_yticks(np.arange(len(labels)), labels)

    ax.set_title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    return


def plot_common_embedding(
    submissions_data, embedding_results, title="", pcs=(0, 1), save_path=None
):
    all_embeddings = []
    labels = []
    pc1, pc2 = pcs

    for label, embedding in embedding_results["common_embedding"].items():
        all_embeddings.append(embedding)
        labels.append(label)

    all_embeddings = torch.cat(all_embeddings, dim=0)

    weights = []
    for i in range(len(labels)):
        weights += submissions_data[labels[i]]["populations"].numpy().tolist()

    weights = torch.tensor(weights)
    weights = weights / weights.sum()

    if "gt_embedding" in embedding_results:
        n_cols = min(3, len(labels) + 1)
        n_rows = min((len(labels) + 1) // n_cols, 1)

    else:
        n_cols = min(3, len(labels))
        n_rows = min(len(labels) // n_cols, 1)

    fig, ax = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), sharex=True, sharey=True
    )
    if n_rows == 1 and n_cols == 1:
        ax = np.array([ax])

    for i in range(len(labels)):
        sns.kdeplot(
            x=all_embeddings[:, pc1],
            y=all_embeddings[:, pc2],
            cmap="viridis",
            fill=True,
            cbar=False,
            ax=ax.flatten()[i],
            weights=weights,
        )

    if "gt_embedding" in embedding_results:
        sns.kdeplot(
            x=all_embeddings[:, pc1],
            y=all_embeddings[:, pc2],
            cmap="viridis",
            fill=True,
            cbar=False,
            ax=ax.flatten()[len(labels)],
            weights=weights,
        )

    for i in range(len(labels)):
        pops = submissions_data[labels[i]]["populations"].numpy()
        pops = pops / pops.sum()

        ax.flatten()[i].scatter(
            x=embedding_results["common_embedding"][labels[i]][:, pc1],
            y=embedding_results["common_embedding"][labels[i]][:, pc2],
            color="red",
            s=pops / pops.max() * 200,
            marker="o",
            linewidth=0.3,
            edgecolor="white",
            label=labels[i],
        )

        ax.flatten()[i].set_xticks([])
        ax.flatten()[i].set_yticks([])

        if i >= n_rows:
            ax.flatten()[i].set_xlabel(f"Z{pc1 + 1}", fontsize=12)
        if i % n_cols == 0:
            ax.flatten()[i].set_ylabel(f"Z{pc2 + 1}", fontsize=12)

        i_max = i

    if "gt_embedding" in embedding_results:
        i_max += 1

        ax.flatten()[i_max].scatter(
            x=embedding_results["gt_embedding"][:, pc1],
            y=embedding_results["gt_embedding"][:, pc2],
            color="red",
            s=100,
            marker="o",
            linewidth=0.3,
            edgecolor="white",
            label="Ground Truth",
        )

        ax.flatten()[i_max].set_xlabel(f"Z{pc1 + 1}", fontsize=12)
        ax.flatten()[i_max].set_ylabel(f"Z{pc2 + 1}", fontsize=12)
        ax.flatten()[i_max].set_xticks([])
        ax.flatten()[i_max].set_yticks([])

    if i_max < n_cols * n_rows:
        for j in range(i_max + 1, n_cols * n_rows):
            ax.flatten()[j].axis("off")

    # adjust horizontal space
    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    fig.suptitle(title, fontsize=16)
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(
        lines, labels, loc="center right", fontsize=12, bbox_to_anchor=(1.071, 0.5)
    )

    if save_path is not None:
        plt.savefig(save_path)

    return


def plot_gt_embedding(submissions_data, gt_embedding_results, title="", save_path=None):
    def gauss_pdf(x, mu, var):
        return 1 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * (x - mu) ** 2 / var)

    def compute_gt_dist(z):
        gauss1 = gauss_pdf(z, 150, 750)
        gauss2 = 0.5 * gauss_pdf(z, 0, 500)
        gauss3 = gauss_pdf(z, -150, 750)
        return gauss1 + gauss2 + gauss3

    n_cols = 3
    n_rows = len(list(submissions_data.keys())) // n_cols + 1

    fig, ax = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), sharex=True, sharey=True
    )

    low_gt = -227.927103122416
    high_gt = 214.014930744738
    Z = np.linspace(low_gt, high_gt, gt_embedding_results["gt_embedding"].shape[0])
    x_axis = np.linspace(
        torch.min(gt_embedding_results["gt_embedding"][:, 0]),
        torch.max(gt_embedding_results["gt_embedding"][:, 0]),
        gt_embedding_results["gt_embedding"].shape[0],
    )

    gt_dist = compute_gt_dist(Z)
    gt_dist /= np.max(gt_dist)

    frq, edges = np.histogram(gt_embedding_results["gt_embedding"][:, 0], bins=20)

    i = 0
    for label, embedding in gt_embedding_results["submission_embedding"].items():
        ax.flatten()[i].bar(
            edges[:-1],
            frq / frq.max(),
            width=np.diff(edges),
            # label="Ground Truth",
            alpha=0.8,
            color="#a1c9f4",
        )

        ax.flatten()[i].plot(x_axis, gt_dist)  # , label="True Distribution")

        populations = submissions_data[label]["populations"]
        ax.flatten()[i].scatter(
            x=embedding[:, 0],
            y=populations / populations.max(),
            color="red",
            marker="o",
            s=60,
            linewidth=0.3,
            edgecolor="white",
            label=label,
        )

        # set x label only for the last row
        if i >= n_rows:
            ax.flatten()[i].set_xlabel("SVD 1", fontsize=12)
        # set y label only for the first column
        if i % n_cols == 0:
            ax.flatten()[i].set_ylabel("Scaled probability", fontsize=12)

        # ax.flatten()[i].legend(loc="upper left", fontsize=12)
        ax.flatten()[i].set_ylim(0.0, 1.1)
        ax.flatten()[i].set_xlim(x_axis[0] * 1.3, x_axis[-1] * 1.3)
        # set ticks to be maximum 5 ticks
        ax.flatten()[i].set_yticks(np.arange(0, 1.25, 0.25))
        ax.flatten()[i].set_xticks([])

        plt.subplots_adjust(wspace=0.0, hspace=0.0)

        i += 1

    if i < n_cols * n_rows:
        for j in range(i + 1, n_cols * n_rows):
            ax.flatten()[j].axis("off")

    fig.suptitle(title, fontsize=16)
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(
        lines, labels, loc="center right", fontsize=12, bbox_to_anchor=(1.071, 0.5)
    )

    if save_path is not None:
        plt.savefig(save_path)

    return
