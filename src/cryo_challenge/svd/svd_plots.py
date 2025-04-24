import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

# PLOT_SETUP = {
#     "Ground Truth": {"color": "#e41a1c", "marker": "o"},
#     "Cookie Dough": {"color": "#377eb8", "marker": "v"},
#     "Mango": {"color": "#4daf4a", "marker": "^"},
#     "Vanilla": {"color": "#984ea3", "marker": "<"},
#     "Peanut Butter": {"color": "#ff7f00", "marker": ">"},
#     "Neapolitan": {"color": "#ffff33", "marker": "D"},
#     "Chocolate": {"color": "#a65628", "marker": "x"},
#     "Black Raspberry": {"color": "#f781bf", "marker": "*"},
#     "Cherry": {"color": "#999999", "marker": "s"},
#     "Salted Caramel": {"color": "#e41a1c", "marker": "p"},
#     "Chocolate Chip": {"color": "#377eb8", "marker": "P"},
#     "Rocky Road": {"color": "#4daf4a", "marker": "*"},
# }


# MARKERS = ["o", "v", "^", "<", ">", "D", "X", "*", "s", "p", "P", "*", "h", "H"]
# LABELS = [
#     "Mango",
#     "Cookie Dough",
#     "Vanilla",
#     "Peanut Butter",
#     "Neapolitan",
#     "Chocolate",
#     "Black Raspberry",
#     "Cherry",
#     "Salted Caramel",
#     "Chocolate Chip",
#     "Rocky Road",
#     "Pina Colada",
#     "Ground Truth",
#     "Mint Chocolate Chip",
#     "Bubble Gum",
# ]

COLORS = {
    "1": "#648fff",
    "2": "#785ef0",
    "3": "#dc267f",
    "4": "#fe6100",
    "5": "#ffb000",
}

PLOT_SETUP = {
    "Salted Caramel": {"category": "1", "marker": "o"},
    "Neapolitan": {"category": "1", "marker": "v"},
    "Peanut Butter": {"category": "1", "marker": "^"},
    "Coffee": {"category": "1", "marker": "<"},
    "Cherry": {"category": "2", "marker": "o"},
    "Pina Colada": {"category": "2", "marker": "v"},
    "Cookie Dough": {"category": "2", "marker": "^"},
    "Chocolate Chip": {"category": "2", "marker": "<"},
    "Chocolate": {"category": "2", "marker": ">"},
    "Vanilla": {"category": "3", "marker": "o"},
    "Mango": {"category": "3", "marker": "v"},
    "Rocky Road": {"category": "4", "marker": "o"},
    "Black Raspberry": {"category": "4", "marker": "v"},
    "Ground Truth": {"category": "5", "marker": "o"},
    "Bubble Gum": {"category": "5", "marker": "v"},
    "Mint Chocolate Chip": {"category": "5", "marker": "^"},
}

for key in list(PLOT_SETUP.keys()):
    PLOT_SETUP[key]["color"] = COLORS[PLOT_SETUP[key]["category"]]


def compare_strings(fixed_string, other_string):
    return other_string.startswith(fixed_string)


def sort_labels_category(labels, plot_setup):
    labels_sorted = []
    for i in range(5):  # there are 5 categories
        for label in labels:
            if plot_setup[label]["category"] == str(i + 1):
                labels_sorted.append(label)

    return labels_sorted


def plot_distance_matrix(dist_matrix, labels, title="", save_path=None):
    fig, ax = plt.subplots()
    cax = ax.matshow(dist_matrix, cmap="viridis")
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=90)
    ax.set_yticks(np.arange(len(labels)), labels)

    ax.set_title(title)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    return


# def plot_common_embedding(
#     submissions_data, embedding_results, title="", pcs=(0, 1), save_path=None
# ):
#     all_embeddings = []
#     labels = []
#     pc1, pc2 = pcs

#     for label, embedding in embedding_results["common_embedding"].items():
#         all_embeddings.append(embedding)
#         labels.append(label)

#     plot_setup = {}
#     for i, label in enumerate(labels):
#         for possible_label in PLOT_SETUP.keys():
#             # print(label, possible_label)
#             if compare_strings(possible_label, label):
#                 plot_setup[label] = PLOT_SETUP[possible_label]

#     for label in labels:
#         if label not in plot_setup.keys():
#             raise ValueError(f"Label {label} not found in PLOT_SETUP")

#     if "gt_embedding" in embedding_results:
#         plot_setup["Ground Truth"] = PLOT_SETUP["Ground Truth"]

#     labels = sort_labels_category(labels, plot_setup)
#     all_embeddings = torch.cat(all_embeddings, dim=0)

#     weights = []
#     for i in range(len(labels)):
#         weights += submissions_data[labels[i]]["populations"].numpy().tolist()

#     weights = torch.tensor(weights)
#     weights = weights / weights.sum()

#     if "gt_embedding" in embedding_results:
#         n_rows = np.sqrt(len(labels) + 1)
#         n_rows = np.ceil(n_rows).astype(int)
#         n_cols = np.ceil((len(labels) + 1) / n_rows).astype(int)

#     else:
#         n_rows = np.sqrt(len(labels))
#         n_rows = np.ceil(n_rows).astype(int)
#         n_cols = np.ceil(len(labels) / n_rows).astype(int)

#     fig, ax = plt.subplots(
#         n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3), sharex=True, sharey=True
#     )
#     if n_rows == 1 and n_cols == 1:
#         ax = np.array([ax])

#     # for i in range(len(labels)):
#     #     sns.kdeplot(
#     #         x=all_embeddings[:, pc1],
#     #         y=all_embeddings[:, pc2],
#     #         cmap="gray",
#     #         fill=False,
#     #         cbar=False,
#     #         ax=ax.flatten()[i],
#     #         weights=weights,
#     #         alpha=0.8,
#     #         zorder=1,
#     #     )

#     # if "gt_embedding" in embedding_results:
#     #     sns.kdeplot(
#     #         x=all_embeddings[:, pc1],
#     #         y=all_embeddings[:, pc2],
#     #         cmap="gray",
#     #         fill=False,
#     #         cbar=False,
#     #         ax=ax.flatten()[len(labels)],
#     #         weights=weights,
#     #         # alpha=0.5,
#     #         zorder=1,
#     #     )

#     for i in range(len(labels)):
#         pops = submissions_data[labels[i]]["populations"].numpy()
#         pops = pops / pops.sum()

#         # put a value of i in the top left corner of each plot
#         # ax.flatten()[i].text(
#         #     0.05,
#         #     0.95,
#         #     str(i + 1),
#         #     fontsize=12,
#         #     transform=ax.flatten()[i].transAxes,
#         #     verticalalignment="top",
#         #     bbox=dict(facecolor="white", alpha=0.5),
#         # )
#         ax.flatten()[i].scatter(
#             x=embedding_results["common_embedding"][labels[i]][:, pc1],
#             y=embedding_results["common_embedding"][labels[i]][:, pc2],
#             color=plot_setup[labels[i]]["color"],
#             s=pops / pops.max() * 200,
#             marker=plot_setup[labels[i]]["marker"],
#             linewidth=0.3,
#             edgecolor="black",
#             label=f"{labels[i]} - Group {plot_setup[labels[i]]['category']}.",
#             #label=labels[i],
#             zorder=2,
#         )

#         ax.flatten()[i].set_xticks([])
#         ax.flatten()[i].set_yticks([])

#         if i >= n_rows:
#             ax.flatten()[i].set_xlabel(f"Z{pc1 + 1}", fontsize=12)
#         if i % n_cols == 0:
#             ax.flatten()[i].set_ylabel(f"Z{pc2 + 1}", fontsize=12)

#         ax.flatten()[i].legend(loc="upper left", fontsize=12)

#         i_max = i

#     if "gt_embedding" in embedding_results:
#         i_max += 1
#         # ax.flatten()[i_max].text(
#         #     0.05,
#         #     0.95,
#         #     str(i_max + 1),
#         #     fontsize=12,
#         #     transform=ax.flatten()[i_max].transAxes,
#         #     verticalalignment="top",
#         #     bbox=dict(facecolor="white", alpha=0.5),
#         # )
#         ax.flatten()[i_max].scatter(
#             x=embedding_results["gt_embedding"][:, pc1],
#             y=embedding_results["gt_embedding"][:, pc2],
#             color=plot_setup["Ground Truth"]["color"],
#             s=100,
#             marker=plot_setup["Ground Truth"]["marker"],
#             linewidth=0.3,
#             edgecolor="black",
#             #label=f"{i_max + 1}. Ground Truth",
#             label="Ground Truth - Group 5",
#             zorder=2,
#         )

#         ax.flatten()[i_max].set_xlabel(f"Z{pc1 + 1}", fontsize=12)
#         ax.flatten()[i_max].set_ylabel(f"Z{pc2 + 1}", fontsize=12)
#         ax.flatten()[i_max].set_xticks([])
#         ax.flatten()[i_max].set_yticks([])
#         ax.flatten()[i_max].legend(loc="upper left", fontsize=12)

#     if i_max < n_cols * n_rows:
#         for j in range(i_max + 1, n_cols * n_rows):
#             ax.flatten()[j].axis("off")

#     # adjust horizontal space
#     plt.subplots_adjust(wspace=0.0, hspace=0.0)

#     fig.suptitle(title, fontsize=16)
#     # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
#     # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
#     # fig.legend(
#     #     lines, labels, loc="center right", fontsize=12, bbox_to_anchor=(1.071, 0.5)
#     # )

#     # Generate a box of test on the left side with "category = definition"
#     group_definitions = {
#         "Group 1": "Physics-informed",
#         "Group 2": "Neural network (no physics)",
#         "Group 3": "Linear Method",
#         "Group 4": "Non-linear Method",
#     }

#     if "gt_embedding" in embedding_results:
#         group_definitions["Group 5"] = "Based on GT"

#     text = "\n".join([f"{group}: {definition}" for group, definition in group_definitions.items()])
#     fig.text(
#         0.915, 0.825, text,
#         va='center', ha='left', fontsize=12,
#         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgrey', alpha=0.3, edgecolor='black'),
#         multialignment='left', linespacing=1.8  # Increased vertical spacing
#     )

#     if save_path is not None:
#         plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

#     return


def plot_common_embedding(
    submissions_data, embedding_results, title="", pcs=(0, 1), save_path=None
):
    all_embeddings = []
    labels = []
    pc1, pc2 = pcs

    for label, embedding in embedding_results["common_embedding"].items():
        all_embeddings.append(embedding)
        labels.append(label)

    if "gt_embedding" in embedding_results:
        labels.append("Ground Truth")

    plot_setup = {}
    for i, label in enumerate(labels):
        for possible_label in PLOT_SETUP.keys():
            # print(label, possible_label)
            if compare_strings(possible_label, label):
                plot_setup[label] = PLOT_SETUP[possible_label]

    for label in labels:
        if label not in plot_setup.keys():
            raise ValueError(f"Label {label} not found in PLOT_SETUP")

    labels = sort_labels_category(labels, plot_setup)
    all_embeddings = torch.cat(all_embeddings, dim=0)

    weights = []
    for i in range(len(labels)):
        if labels[i] != "Ground Truth":
            weights += submissions_data[labels[i]]["populations"].numpy().tolist()

    weights = torch.tensor(weights)
    weights = weights / weights.sum()

    n_cols = 3

    if n_cols > len(labels):
        n_cols = len(labels)
        n_rows = 1
    else:
        n_rows = len(labels) // n_cols + 1

    fig, ax = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3), sharex=True, sharey=True
    )
    if n_rows == 1 and n_cols == 1:
        ax = np.array([ax])

    for i in range(len(labels)):
        sns.kdeplot(
            x=all_embeddings[:, pc1],
            y=all_embeddings[:, pc2],
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
        if label != "Ground Truth":
            pops = submissions_data[label]["populations"].numpy()
            pops = pops / pops.sum()

            ax.flatten()[i].scatter(
                x=embedding_results["common_embedding"][label][:, pc1],
                y=embedding_results["common_embedding"][label][:, pc2],
                color=plot_setup[label]["color"],
                s=pops / pops.max() * 200,
                marker=plot_setup[label]["marker"],
                linewidth=0.3,
                edgecolor="black",
                label=f"{label} - Group {plot_setup[label]['category']}.",
                zorder=2,
            )
        else:
            ax.flatten()[i].scatter(
                x=embedding_results["gt_embedding"][:, pc1],
                y=embedding_results["gt_embedding"][:, pc2],
                color=plot_setup["Ground Truth"]["color"],
                s=100,
                marker=plot_setup["Ground Truth"]["marker"],
                linewidth=0.3,
                edgecolor="black",
                # label=f"{i_max + 1}. Ground Truth",
                label="Ground Truth - Group 5",
                zorder=2,
            )

        ax.flatten()[i].set_xticks([])
        ax.flatten()[i].set_yticks([])
        ax.flatten()[i].legend(loc="upper left", fontsize=12)
        if i >= n_rows:
            ax.flatten()[i].set_xlabel(f"Z{pc1 + 1}", fontsize=12)
        if i % n_cols == 0:
            ax.flatten()[i].set_ylabel(f"Z{pc2 + 1}", fontsize=12)

    for i in range(len(labels), n_cols * n_rows):
        ax.flatten()[i].axis("off")

    # adjust horizontal space
    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    fig.suptitle(title, fontsize=16)
    # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # fig.legend(
    #     lines, labels, loc="center right", fontsize=12, bbox_to_anchor=(1.071, 0.5)
    # )

    # Generate a box of test on the left side with "category = definition"
    group_definitions = {
        "Group 1": "Physics-informed",
        "Group 2": "Neural network (no physics)",
        "Group 3": "Linear Method",
        "Group 4": "Non-linear Method",
    }

    if "gt_embedding" in embedding_results:
        group_definitions["Group 5"] = "Based on GT"

    text = "\n".join(
        [f"{group}: {definition}" for group, definition in group_definitions.items()]
    )
    fig.text(
        0.915,
        0.825,
        text,
        va="center",
        ha="left",
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="lightgrey",
            alpha=0.3,
            edgecolor="black",
        ),
        multialignment="left",
        linespacing=1.8,  # Increased vertical spacing
    )

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    return


def plot_gt_embedding(submissions_data, gt_embedding_results, title="", save_path=None):
    def gauss_pdf(x, mu, var):
        return 1 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * (x - mu) ** 2 / var)

    def compute_gt_dist(z):
        gauss1 = gauss_pdf(z, 150, 750)
        gauss2 = 0.5 * gauss_pdf(z, 0, 500)
        gauss3 = gauss_pdf(z, -150, 750)
        return gauss1 + gauss2 + gauss3

    labels = list(submissions_data.keys())

    plot_setup = {}
    for i, label in enumerate(submissions_data.keys()):
        for possible_label in PLOT_SETUP.keys():
            if compare_strings(possible_label, label):
                plot_setup[label] = PLOT_SETUP[possible_label]

    for label in submissions_data.keys():
        if label not in plot_setup.keys():
            raise ValueError(f"Label {label} not found in PLOT_SETUP")

    labels = sort_labels_category(labels, plot_setup)

    # low_gt = -231.62100638454024
    # high_gt = 243.32448171011487
    # Z = np.linspace(low_gt, high_gt, gt_embedding_results["gt_embedding"].shape[0])
    # x_axis = np.linspace(
    #     torch.min(gt_embedding_results["gt_embedding"][:, 0]),
    #     torch.max(gt_embedding_results["gt_embedding"][:, 0]),
    #     gt_embedding_results["gt_embedding"].shape[0],
    # )

    # gt_dist = compute_gt_dist(Z)
    # gt_dist /= np.max(gt_dist)

    # frq, edges = np.histogram(gt_embedding_results["gt_embedding"][:, 0], bins=20)
    label_ref = "Mint Chocolate Chip 1"
    populations_ref = submissions_data[label_ref]["populations"]
    embedding_ref = gt_embedding_results["submission_embedding"][label_ref]

    labels.pop(labels.index(label_ref))

    n_cols = 3

    if n_cols > len(labels):
        n_cols = len(labels)
        n_rows = 1
    else:
        n_rows = len(labels) // n_cols + 1
    fig, ax = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), sharex=True, sharey=True
    )
    if n_rows == 1 and n_cols == 1:
        ax = np.array([ax])

    for i in range(len(labels)):
        label = labels[i]
        embedding = gt_embedding_results["submission_embedding"][label]
        # ax.flatten()[i].text(
        #     0.05,
        #     0.95,
        #     str(i + 1),
        #     fontsize=12,
        #     transform=ax.flatten()[i].transAxes,
        #     verticalalignment="top",
        #     bbox=dict(facecolor="white", alpha=0.5),
        # )

        # ax.flatten()[i].bar(
        #     edges[:-1],
        #     frq / frq.max(),
        #     width=np.diff(edges),
        #     # label="Ground Truth",
        #     alpha=0.8,
        #     color="#a1c9f4",
        # )

        # ax.flatten()[i].plot(x_axis, gt_dist)  # , label="True Distribution")

        populations = submissions_data[label]["populations"]
        ax.flatten()[i].scatter(
            x=embedding[:, 0],
            y=populations / populations.max(),
            color=plot_setup[label]["color"],
            marker=plot_setup[label]["marker"],
            s=100,
            linewidth=0.3,
            edgecolor="black",
            label=f"{i+1}. {label}",
        )

        ax.flatten()[i].plot(
            embedding_ref[:, 0],
            populations_ref / populations_ref.max(),
            # color=plot_setup[label]["color"],
            color="black",
            marker="o",
            # marker=plot_setup[label]["marker"],
            # s=100,
            linewidth=0.3,
            # edgecolor="black",
            # label=f"{i+1}. {label}",
            alpha=0.7,
        )

        # set x label only for the last row
        if i >= n_rows:
            ax.flatten()[i].set_xlabel("SVD 1", fontsize=12)
        # set y label only for the first column
        if i % n_cols == 0:
            ax.flatten()[i].set_ylabel("Scaled probability", fontsize=12)

        ax.flatten()[i].legend(loc="upper left", fontsize=12)
        ax.flatten()[i].set_ylim(0.0, 1.25)
        # ax.flatten()[i].set_xlim(x_axis[0] * 1.3, x_axis[-1] * 1.3)
        # set ticks to be maximum 5 ticks
        ax.flatten()[i].set_yticks(np.arange(0.25, 1.25, 0.25))
        ax.flatten()[i].set_xticks([])

        plt.subplots_adjust(wspace=0.0, hspace=0.0)

    if i < n_cols * n_rows:
        for j in range(i + 1, n_cols * n_rows):
            ax.flatten()[j].axis("off")

    fig.suptitle(title, fontsize=16)
    # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # fig.legend(
    #     lines, labels, loc="center right", fontsize=12, bbox_to_anchor=(1.071, 0.5)
    # )

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    return


def plot_common_eigenvectors(
    common_eigenvectors, n_eig_to_plot=None, title="", save_path=None
):
    n_eig_to_plot = min(10, len(common_eigenvectors))
    n_cols = 5
    n_rows = int(np.ceil(n_eig_to_plot / n_cols))

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 5))

    box_size = int(round((common_eigenvectors[0].shape[-1]) ** (1 / 3)))
    for i in range(n_eig_to_plot):
        eigvol = common_eigenvectors[i].reshape(box_size, box_size, box_size)

        mask_small = torch.where(torch.abs(eigvol) < 1e-3)
        mask_pos = torch.where(eigvol > 0)
        mask_neg = torch.where(eigvol < 0)

        eigvol_pos = torch.zeros_like(eigvol)
        eigvol_neg = torch.zeros_like(eigvol)

        eigvol_pos[mask_pos] = 1.0
        eigvol_neg[mask_neg] = -1.0

        eigvol_for_img = eigvol_neg + eigvol_pos
        eigvol_for_img[mask_small] = 0.0

        ax.flatten()[i].imshow(
            eigvol_for_img.sum(0), cmap="coolwarm", label=f"Eigenvector {i}"
        )
        ax.flatten()[i].set_title(f"Eigenvector {i}")
        ax.flatten()[i].axis("off")
        i_max = i

    if i_max < n_cols * n_rows:
        for j in range(i_max + 1, n_cols * n_rows):
            ax.flatten()[j].axis("off")

    plt.subplots_adjust(wspace=0.0)

    # add a colorbar for the whole figure
    fig.colorbar(
        ax.flatten()[i].imshow(eigvol_for_img.sum(1), cmap="coolwarm"),
        ax=ax,
        orientation="horizontal",
        label="Eigenvector value (neg or pos)",
    )

    fig.suptitle(title, fontsize=16)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    return
