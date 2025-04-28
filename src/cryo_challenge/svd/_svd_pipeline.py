import torch

from .svd_utils import (
    compute_cap_var_distance_matrix,
    compute_common_embedding,
    project_to_gt_embedding,
)
from ._load_submisisons_and_gt import load_submissions, load_gt
from ..config_validation._svd_analysis_validation import SVDInputConfig


def run_svd_with_ref(config: SVDInputConfig):
    submissions_svd = load_submissions(config)
    gt_svd = load_gt(config)

    cap_var_dist_mtx_results = compute_cap_var_distance_matrix(submissions_svd, gt_svd)
    # fsc_dist_mtx_results = compute_fsc_matrix_first_eigvecs(
    #     submissions_svd, gt_svd
    # )

    common_embedding_results = compute_common_embedding(submissions_svd, gt_svd)
    gt_embedding_results = project_to_gt_embedding(submissions_svd, gt_svd)

    results = {
        "capvar_distance_matrix_results": cap_var_dist_mtx_results,
        # "fsc_distance_matrix_results": fsc_dist_mtx_results,
        "common_embedding_results": common_embedding_results,
        "gt_embedding_results": gt_embedding_results,
    }

    if config.output_params["save_svd_data"]:
        results["submissions_svd"] = submissions_svd
        results["gt_svd"] = gt_svd

    torch.save(results, config.output_params["output_file"])

    if config.output_params["generate_plots"]:
        raise NotImplementedError(
            "Plots are currently turned off due to incompatibilities. Your results were saved right before this error triggered."
        )

    return


def run_svd_noref(config: dict):
    submissions_svd = load_submissions(config)
    cap_var_dist_mtx_results = compute_cap_var_distance_matrix(submissions_svd)
    # fsc_dist_mtx_results = compute_fsc_matrix_first_eigvecs(submissions_svd)
    common_embedding_results = compute_common_embedding(submissions_svd)

    results = {
        "distance_matrix_results": cap_var_dist_mtx_results,
        # "fsc_distance_matrix_results": fsc_dist_mtx_results,
        "common_embedding_results": common_embedding_results,
    }

    if config.output_params["save_svd_data"]:
        results["submissions_svd"] = submissions_svd

    torch.save(results, config.output_params["output_file"])

    if config.output_params["generate_plots"]:
        raise NotImplementedError(
            "Plots are currently turned off due to incompatibilities. Your results were saved right before this error triggered."
        )

    return
