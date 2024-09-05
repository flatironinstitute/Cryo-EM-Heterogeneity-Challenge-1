import torch
import os

from .svd_utils import (
    compute_distance_matrix,
    compute_common_embedding,
    project_to_gt_embedding,
)
from .svd_plots import (
    plot_distance_matrix,
    plot_common_embedding,
    plot_gt_embedding,
    plot_common_eigenvectors,
)
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
        outputs_fname_nopath_noext = os.path.basename(
            config["output_params"]["output_file"]
        )
        outputs_fname_nopath_noext = os.path.splitext(outputs_fname_nopath_noext)[0]
        path_plots = os.path.join(outputs_path, f"plots_{outputs_fname_nopath_noext}")

        os.makedirs(path_plots, exist_ok=True)

        print("Plotting distance matrix")
        plot_distance_matrix(
            dist_mtx_results["dist_matrix"],
            dist_mtx_results["labels"],
            title="SVD Distance Matrix",
            save_path=os.path.join(path_plots, "svd_distance_matrix.png"),
        )

        print("Plotting common embedding")
        plot_common_embedding(
            submissions_data,
            common_embedding_results,
            title="Common Embedding between submissions",
            save_path=os.path.join(path_plots, "common_embedding.png"),
        )

        print("Plotting gt embedding")
        plot_gt_embedding(
            submissions_data,
            gt_embedding_results,
            title="",
            save_path=os.path.join(path_plots, "gt_embedding.png"),
        )

        print("Plotting common eigenvectors")
        plot_common_eigenvectors(
            common_embedding_results["common_eigenvectors"],
            title="Common Eigenvectors between submissions",
            save_path=os.path.join(path_plots, "common_eigenvectors.png"),
        )

    return


def run_svd_noref(config: dict):
    outputs_path = os.path.dirname(config["output_params"]["output_file"])

    submissions_data = load_submissions_svd(config)
    dist_mtx_results = compute_distance_matrix(submissions_data)
    common_embedding_results = compute_common_embedding(submissions_data)

    results = {
        "distance_matrix_results": dist_mtx_results,
        "common_embedding_results": common_embedding_results,
    }

    if config["output_params"]["save_svd_data"]:
        results["submissions_data"] = submissions_data

    torch.save(results, config["output_params"]["output_file"])

    if config["output_params"]["generate_plots"]:
        outputs_fname_nopath_noext = os.path.basename(
            config["output_params"]["output_file"]
        )
        outputs_fname_nopath_noext = os.path.splitext(outputs_fname_nopath_noext)[0]
        path_plots = os.path.join(outputs_path, f"plots_{outputs_fname_nopath_noext}")
        os.makedirs(path_plots, exist_ok=True)

        print("Plotting distance matrix")

        plot_distance_matrix(
            dist_mtx_results["dist_matrix"],
            dist_mtx_results["labels"],
            "SVD Distance Matrix",
            save_path=os.path.join(path_plots, "svd_distance_matrix.png"),
        )

        print("Plotting common embedding")
        plot_common_embedding(
            submissions_data,
            common_embedding_results,
            "Common Embedding between submissions",
            save_path=os.path.join(path_plots, "common_embedding.png"),
        )

        print("Plotting common eigenvectors")
        plot_common_eigenvectors(
            common_embedding_results["common_eigenvectors"],
            title="Common Eigenvectors between submissions",
            save_path=os.path.join(path_plots, "common_eigenvectors.png"),
        )

    return
