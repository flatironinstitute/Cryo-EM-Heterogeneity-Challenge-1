import torch
import os

from .svd_utils import (
    compute_distance_matrix,
    compute_common_embedding,
    project_to_gt_embedding,
)
from .svd_plots import plot_distance_matrix, plot_common_embedding, plot_gt_embedding
from ..data._io.svd_io_utils import load_submissions_svd, load_gt_svd


def run_svd_with_ref(config: dict):
    outputs_path = os.path.dirname(config["output_params"]["output_file"])

    submissions_data = load_submissions_svd(config)
    gt_data = load_gt_svd(config)

    dist_mtx_results = compute_distance_matrix(submissions_data, gt_data)
    common_embedding_results = compute_common_embedding(submissions_data, gt_data)
    gt_embedding_results = project_to_gt_embedding(submissions_data, gt_data)

    results = {
        "distance_matrix_results": dist_mtx_results,
        "common_embedding_results": common_embedding_results,
        "gt_embedding_results": gt_embedding_results,
    }

    if config["output_params"]["save_svd_data"]:
        results["submissions_data"] = submissions_data
        results["gt_data"] = gt_data

    torch.save(results, config["output_params"]["output_file"])

    if config["output_params"]["generate_plots"]:
        plot_distance_matrix(
            dist_mtx_results["dist_matrix"],
            dist_mtx_results["labels"],
            "SVD Distance Matrix",
            save_path=os.path.join(outputs_path, "svd_distance_matrix.png"),
        )

        plot_common_embedding(
            submissions_data,
            common_embedding_results,
            "Common Embedding between submissions",
            save_path=os.path.join(outputs_path, "common_embedding.png"),
        )

        plot_gt_embedding(
            submissions_data,
            gt_embedding_results,
            "",
            save_path=os.path.join(outputs_path, "gt_embedding.png"),
        )

    return


def run_svd_noref(config: dict):
    outputs_path = os.path.dirname(config["output_params"]["output_file"])

    submissions_data = load_submissions_svd(config)
    dist_mtx_results = compute_distance_matrix(submissions_data)
    common_embeddings_results = compute_common_embedding(submissions_data)[
        "common_embedding"
    ]

    results = {
        "distance_matrix_results": dist_mtx_results,
        "common_embedding_results": common_embeddings_results,
    }

    if config["output_params"]["save_svd_data"]:
        results["submissions_data"] = submissions_data

    torch.save(results, config["output_params"]["output_file"])

    if config["output_params"]["generate_plots"]:
        plot_distance_matrix(
            dist_mtx_results["dist_matrix"],
            dist_mtx_results["labels"],
            "SVD Distance Matrix",
            save_path=os.path.join(outputs_path, "svd_distance_matrix.png"),
        )

        plot_common_embedding(
            submissions_data,
            common_embeddings_results,
            "Common Embedding between submissions",
            save_path=os.path.join(outputs_path, "common_embedding.png"),
        )

    return
