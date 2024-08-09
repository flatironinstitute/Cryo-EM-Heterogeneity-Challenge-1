from numbers import Number
import pandas as pd
import os


def validate_generic_config(config: dict, reference: dict) -> None:
    """
    Validate a config dictionary against a reference dictionary.

    Parameters
    ----------
    config : dict
        The dictionary to validate.
    reference : dict
        The reference dictionary to validate against.

    Raises
    ------
    ValueError
        If a key in reference is not present in config.
    ValueError
        If the type of a key in config does not match the type of the corresponding key in reference.

    Returns
    -------
    None
    """  # noqa: E501
    for key in reference:
        if key not in config:
            raise ValueError(f"Missing key in config: {key}")
        if not isinstance(config[key], reference[key]):
            raise ValueError(
                f"Invalid type for key {key} in config: {type(config[key])}"
            )
    return


# Preprocessing
def validate_config_preprocessing(config_data: dict) -> None:
    """
    Validate the config dictionary for the preprocessing pipeline.
    """
    keys_and_types = {
        "submission_config_file": str,
        "output_path": str,
        "thresh_percentile": Number,
        "BOT_box_size": int,
        "BOT_loss": str,
        "BOT_iter": Number,
        "BOT_refine": bool,
    }
    validate_generic_config(config_data, keys_and_types)
    return


# MapToMap Distances
def validate_config_mtm_data_submission(config_data_submission: dict) -> None:
    """
    Validate the submission part of the config dictionary for the MapToMap config.

    fname: str, is the path to the submission file (submission_*.pt).
    volume_key: str, is the key in the submission file that contains the volume.
    metadata_key: str, is the key in the submission file that contains the metadata.
    label_key: str, is the key in the submission file that contains the (anonymizing) label.
    """
    keys_and_types = {
        "fname": str,
        "volume_key": str,
        "metadata_key": str,
        "label_key": str,
    }
    validate_generic_config(config_data_submission, keys_and_types)
    return


def validate_config_mtm_data_ground_truth(config_data_ground_truth: dict) -> None:
    """
    Validate the ground truth part of the config dictionary for the MapToMap config.

    volumes: str, is the path to the ground truth volume (.pt) file.
    metadata: str, is the path to the ground truth metadata (.csv) file.
    """
    keys_and_types = {
        "volumes": str,
        "metadata": str,
    }
    validate_generic_config(config_data_ground_truth, keys_and_types)
    return


def validate_config_mtm_data_mask(config_data_mask: dict) -> None:
    """
    Validate the mask part of the config dictionary for the MapToMap config.

    do: bool, is a flag to indicate whether to use a mask.
    volume: str, is the path to the mask volume (.mrc) file.
    """
    keys_and_types = {
        "do": bool,
        "volume": str,
    }
    validate_generic_config(config_data_mask, keys_and_types)
    return


def validate_config_mtm_data(config_data: dict) -> None:
    """
    Validate the data part of the config dictionary for the MapToMap config.

    n_pix: int, is the number of pixels in each dimension of the volume.
    psize: float, is the pixel size of the volume in Angstroms.
    submission: dict, is the submission part of the config.
    ground_truth: dict, is the ground truth part of the config.
    mask: dict, is the mask part of the config.
    """
    keys_and_types = {
        "n_pix": Number,
        "psize": Number,
        "submission": dict,
        "ground_truth": dict,
        "mask": dict,
    }

    validate_generic_config(config_data, keys_and_types)
    validate_config_mtm_data_submission(config_data["submission"])
    validate_config_mtm_data_ground_truth(config_data["ground_truth"])
    validate_config_mtm_data_mask(config_data["mask"])
    return


def validate_config_mtm_analysis_normalize(config_analysis_normalize: dict) -> None:
    """
    Validate the normalize part of the analysis part of the config dictionary for the MapToMap config.

    do: bool, is a flag to indicate whether to normalize the volumes.
    method: str, is the method to use for normalization.
    """  # noqa: E501
    keys_and_types = {
        "do": bool,
        "method": str,
    }
    validate_generic_config(config_analysis_normalize, keys_and_types)
    return


def validate_config_mtm_analysis(config_analysis: dict) -> None:
    """
    Validate the analysis part of the config dictionary for the MapToMap config.

    metrics: list, is a list of metrics to compute.
    chunk_size_submission: int, is the chunk size for the submission volume.
    chunk_size_gt: int, is the chunk size for the ground truth volume.
    normalize: dict, is the normalize part of the analysis part of the config.

    """  # noqa: E501
    keys_and_types = {
        "metrics": list,
        "chunk_size_submission": Number,
        "chunk_size_gt": Number,
        "normalize": dict,
    }

    validate_generic_config(config_analysis, keys_and_types)
    validate_config_mtm_analysis_normalize(config_analysis["normalize"])
    return


def validate_input_config_mtm(config: dict) -> None:
    """
    Validate the config dictionary for the MapToMap config.

    data: dict, is the data part of the config.
    analysis: dict, is the analysis part of the config.
    output: str, is the path to the output file.
    """  # noqa: E501
    keys_and_types = {
        "data": dict,
        "analysis": dict,
        "output": str,
    }

    validate_generic_config(config, keys_and_types)
    validate_config_mtm_data(config["data"])
    validate_config_mtm_analysis(config["analysis"])

    return


def validate_maptomap_result(output_dict: dict) -> None:
    """
    Validate the output dictionary of the map-to-map distance matrix computation.

    cost_matrix: pd.DataFrame, is the cost matrix, with ground truth rows and submission columns.
    user_submission_label: str, is the label of the submission.
    computed_assets: dict, is a dictionary of computed assets, which can be re-used in other analyses.
    """  # noqa: E501
    keys_and_types = {
        "cost_matrix": pd.DataFrame,
        "user_submission_label": str,
        "computed_assets": dict,
    }
    validate_generic_config(output_dict, keys_and_types)
    return


# DistributionToDistribution distances
def validate_config_dtd_optimal_q_kl(config_optimal_q_kl: dict) -> None:
    """
    Validate the optimal_q_kl part of the config dictionary for the DistributionToDistribution config.

    n_iter: int, is the number of iterations for the optimization.
    break_atol: float, is the absolute tolerance for the optimization.
    """  # noqa: E501
    keys_and_types = {
        "n_iter": Number,
        "break_atol": Number,
    }
    validate_generic_config(config_optimal_q_kl, keys_and_types)
    return


def validate_input_config_disttodist(config: dict) -> None:
    """
    Validate the config dictionary.

    input_fname: str, is the path to the map to map distance matrix (output from map2map_pipeline).
    metrics: list, is a list of metrics to compute.
    gt_metadata_fname: str, is the path to the ground truth metadata (.csv) file.
    n_replicates: int, is the number of replicates to compute.
    n_pool_microstate: int, is the number of microstates to pool (low values less than 3-5 can cause problems for optimization convergence in CVXPY numerical solvers).
    replicate_fraction: float, is the fraction of the data to use for replicates.
    cvxpy_solver: str, is the solver to use for CVXPY optimization.
    optimal_q_kl: dict, is the optimal_q_kl part of the config.
    """  # noqa: E501
    keys_and_types = {
        "input_fname": str,
        "metrics": list,
        "gt_metadata_fname": str,
        "n_replicates": Number,
        "n_pool_microstate": Number,
        "replicate_fraction": Number,
        "cvxpy_solver": str,
        "optimal_q_kl": dict,
        "output_fname": str,
    }

    validate_generic_config(config, keys_and_types)
    validate_config_dtd_optimal_q_kl(config["optimal_q_kl"])

    return


# SVD
def validate_config_svd_output(config_output: dict) -> None:
    """
    Validate the output part of the config dictionary for the SVD pipeline.
    """  # noqa: E501
    keys_and_types = {
        "output_path": str,
        "save_volumes": bool,
        "save_svd_matrices": bool,
    }
    validate_generic_config(config_output, keys_and_types)
    return


def validate_config_svd(config: dict) -> None:
    """
    Validate the config dictionary for the SVD pipeline.
    """  # noqa: E501
    keys_and_types = {
        "path_to_volumes": str,
        "box_size_ds": Number,
        "submission_list": list,
        "experiment_mode": str,
        "dtype": str,
        "output_options": dict,
    }

    validate_generic_config(config, keys_and_types)
    validate_config_svd_output(config["output_options"])

    if config["experiment_mode"] == "all_vs_ref":
        if "path_to_reference" not in config.keys():
            raise ValueError(
                "Reference path is required for experiment mode 'all_vs_ref'"
            )

        else:
            assert isinstance(config["path_to_reference"], str)
            os.path.exists(config["path_to_reference"])
            assert (
                "pt" in config["path_to_reference"]
            ), "Reference path point to a .pt file"

    os.path.exists(config["path_to_volumes"])
    for submission in config["submission_list"]:
        sub_path = os.path.join(
            config["path_to_volumes"] + f"submission_{submission}.pt"
        )
        os.path.exists(sub_path)

    assert config["dtype"] in [
        "float32",
        "float64",
    ], "dtype must be either 'float32' or 'float64'"
    assert config["box_size_ds"] > 0, "box_size_ds must be greater than 0"
    assert config["submission_list"] != [], "submission_list must not be empty"

    return
