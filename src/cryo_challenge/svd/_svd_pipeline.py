import torch
import os
import shutil
import logging

from ._svd_metrics import (
    compute_pcv_matrix,
    compute_fsc_matrix_first_eigvecs,
    compute_common_embedding,
    project_to_gt_embedding,
)
from ._load_submisisons_and_gt import load_submissions, load_gt
from ._svd_utils import (
    compute_svd_for_all_submission,
    compute_common_power_spectrum_on_grid,
    compute_svd_of_submission,
)
from ..config_validation._svd_analysis_validation import SVDInputConfig


def run_svd_with_ref(config: SVDInputConfig):
    """
    Run SVD analysis when a reference is provided.
    This function computes the SVD for all submissions and the ground truth (GT) data,
    computes the Captured Variance distance matrix,and the computes a common embedding
    using the eigenvectors of the submissions.

    In addition, the submissions are projected to the GT embedding.


    **Arguments:**
        config: SVDInputConfig
            Configuration object containing the parameters for the SVD analysis.
            See `cryo_challenge.config_validators.SVDInputConfig` for more details.

    **Returns:**
        None

    **Raises:**
        ValueError: If the results file already exists and `overwrite` is set to False.
        NotImplementedError: If the `generate_plots` parameter is set to True.
    """
    path_to_results = os.path.join(
        config.output_params["path_to_output_dir"], "svd_results.pt"
    )
    if os.path.exists(path_to_results) and not config.output_params["overwrite"]:
        raise ValueError(
            f"Results already exist at {path_to_results}. "
            + "If you want to overwrite them, set `overwrite` to True in the config."
        )
    elif os.path.exists(path_to_results) and config.output_params["overwrite"]:
        logging.info(f"Results already exist at {path_to_results}. Overwriting them.")
    else:
        logging.info(f"Results will be saved at {path_to_results}. ")

    logging.info("Loading Submissions")
    submissions_data = load_submissions(config)
    logging.info("... done")

    logging.info("Loading GT")
    gt_data = load_gt(config)
    logging.info("... done")

    if config.normalize_params["normalize_power_spectrum"]:
        logging.info("Computing common power spectrum")
        common_power_spectrum_on_grid = compute_common_power_spectrum_on_grid(
            dataset_for_svd=submissions_data
        )
        logging.info("... done")
    else:
        common_power_spectrum_on_grid = None

    logging.info("Computing SVD for all submissions")
    submissions_svd = compute_svd_for_all_submission(
        submissions_data=submissions_data,
        power_spectrum_on_grid=common_power_spectrum_on_grid,
        svd_max_rank=config.svd_max_rank,
    )
    logging.info("... done")

    logging.info("Computing SVD for GT")
    gt_svd = compute_svd_of_submission(
        submission=gt_data[0],
        power_spectrum_on_grid=common_power_spectrum_on_grid,
        svd_max_rank=config.svd_max_rank,
    )
    logging.info("... done")

    logging.info("Computing Capured Variance Distance Matrix")
    cap_var_dist_mtx_results = compute_pcv_matrix(
        submissions_svd=submissions_svd,
        gt_svd=gt_svd,
    )
    logging.info("... done")

    logging.info("Computing FSC Distance Matrix")
    fsc_dist_mtx_results = compute_fsc_matrix_first_eigvecs(submissions_svd, gt_svd)
    logging.info("... done")

    logging.info("Computing Common Embedding")
    common_embedding_results = compute_common_embedding(submissions_svd, gt_svd)
    logging.info("... done")

    logging.info("Projecting submissions to GT Embedding")
    gt_embedding_results = project_to_gt_embedding(submissions_svd, gt_svd)
    logging.info("... done")

    results = {
        "capvar_distance_matrix_results": cap_var_dist_mtx_results,
        "fsc_distance_matrix_results": fsc_dist_mtx_results,
        "common_embedding_results": common_embedding_results,
        "gt_embedding_results": gt_embedding_results,
    }

    if not config.output_params["keep_prep_submissions_for_svd"]:
        logging.info("Removing temporary directory for prepared submissions for SVD")
        tmp_dir = os.path.join(
            config.output_params["path_to_output_dir"], "prepared_submissions_for_svd/"
        )
        shutil.rmtree(tmp_dir)
        logging.info("... done")
        logging.info("Temporary directory for prepared submissions for SVD was removed")
    else:
        logging.info(
            "Temporary directory for prepared submissions for SVD will be kept"
        )

    logging.info("Saving results")
    torch.save(results, path_to_results)
    logging.info("... done")

    if config.output_params["generate_plots"]:
        raise NotImplementedError(
            "Plots are currently turned off due to incompatibilities. Your results were saved right before this error triggered."
        )

    return


def run_svd_noref(config: dict):
    """
    Run SVD analysis when a reference is NOT provided.
    This function computes the SVD for all submissions,
    computes the Captured Variance distance matrix, and computes
    the common embedding using the eigenvectors of the submissions.

    **Arguments:**
        config: SVDInputConfig
            Configuration object containing the parameters for the SVD analysis.
            See `cryo_challenge.config_validators.SVDInputConfig` for more details.

    **Returns:**
        None

    **Raises:**
        ValueError: If the results file already exists and `overwrite` is set to False.
        NotImplementedError: If the `generate_plots` parameter is set to True.
    """
    path_to_results = os.path.join(
        config.output_params["path_to_output_dir"], "svd_results.pt"
    )
    if os.path.exists(path_to_results) and not config.output_params["overwrite"]:
        raise ValueError(
            f"Results already exist at {path_to_results}. "
            + "If you want to overwrite them, set `overwrite` to True in the config."
        )
    elif os.path.exists(path_to_results) and config.output_params["overwrite"]:
        logging.info(f"Results already exist at {path_to_results}. Overwriting them.")
    else:
        logging.info(f"Results will be saved at {path_to_results}. ")

    logging.info("Loading Submissions")
    submissions_data = load_submissions(config)
    logging.info("... done")

    if config.normalize_params["normalize_power_spectrum"]:
        logging.info("Computing common power spectrum")
        common_power_spectrum_on_grid = compute_common_power_spectrum_on_grid(
            dataset_for_svd=submissions_data
        )
        logging.info("... done")
    else:
        common_power_spectrum_on_grid = None

    logging.info("Computing SVD for all submissions")
    submissions_svd = compute_svd_for_all_submission(
        submissions_data=submissions_data,
        power_spectrum_on_grid=common_power_spectrum_on_grid,
        svd_max_rank=config.svd_max_rank,
    )
    logging.info("... done")

    logging.info("Computing Capured Variance Distance Matrix")
    cap_var_dist_mtx_results = compute_pcv_matrix(
        submissions_svd=submissions_svd,
        gt_svd=None,
    )
    logging.info("... done")

    logging.info("Computing FSC Distance Matrix")
    fsc_dist_mtx_results = compute_fsc_matrix_first_eigvecs(
        submissions_svd, gt_svd=None
    )
    logging.info("... done")

    logging.info("Computing Common Embedding")
    common_embedding_results = compute_common_embedding(
        submissions_svd=submissions_svd, gt_svd=None
    )
    logging.info("... done")

    results = {
        "capvar_distance_matrix_results": cap_var_dist_mtx_results,
        "fsc_distance_matrix_results": fsc_dist_mtx_results,
        "common_embedding_results": common_embedding_results,
    }

    if not config.output_params["keep_prep_submissions_for_svd"]:
        logging.info("Removing temporary directory for prepared submissions for SVD")
        tmp_dir = os.path.join(
            config.output_params["path_to_output_dir"], "prepared_submissions_for_svd/"
        )
        shutil.rmtree(tmp_dir)
        logging.info("... done")
        logging.info("Temporary directory for prepared submissions for SVD was removed")
    else:
        logging.info(
            "Temporary directory for prepared submissions for SVD will be kept"
        )

    logging.info("Saving results")
    torch.save(results, path_to_results)
    logging.info("... done")

    if config.output_params["generate_plots"]:
        raise NotImplementedError(
            "Plots are currently turned off due to incompatibilities. Your results were saved right before this error triggered."
        )

    return
