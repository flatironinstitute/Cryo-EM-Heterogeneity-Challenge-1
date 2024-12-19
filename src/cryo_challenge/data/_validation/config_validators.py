from numbers import Number
import numpy as np
import pandas as pd
import os
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional, List


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
    low_memory: dict, is the low memory part of the analysis part of the config. # TODO: add validation for low_memory

    """  # noqa: E501
    keys_and_types = {
        "metrics": list,
        "chunk_size_submission": Number,
        "chunk_size_gt": Number,
        "normalize": dict,
        "low_memory": dict,
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
class SVDNormalizeParams(BaseModel):
    mask_path: Optional[str] = None
    bfactor: float = None
    box_size_ds: Optional[int] = None

    @field_validator("mask_path")
    def check_mask_path_exists(cls, value):
        if value is not None:
            if not os.path.exists(value):
                raise ValueError(f"Mask file {value} does not exist.")
        return value

    @field_validator("bfactor")
    def check_bfactor(cls, value):
        if value is not None:
            if value < 0:
                raise ValueError("B-factor must be non-negative.")
        return value

    @field_validator("box_size_ds")
    def check_box_size_ds(cls, value):
        if value is not None:
            if value < 0:
                raise ValueError("Downsampled box size must be non-negative.")
        return value


class SVDGtParams(BaseModel):
    gt_vols_file: str
    skip_vols: int = 1

    @field_validator("gt_vols_file")
    def check_mask_path_exists(cls, value):
        if not os.path.exists(value):
            raise ValueError(f"Could not find file {value}.")

        assert value.endswith(".npy"), "Ground truth volumes file must be a .npy file."

        vols_gt = np.load(value, mmap_mode="r")

        if len(vols_gt.shape) not in [2, 4]:
            raise ValueError(
                "Invalid number of dimensions for the ground truth volumes"
            )
        return value

    @field_validator("skip_vols")
    def check_skip_vols(cls, value):
        if value is not None:
            if value < 0:
                raise ValueError("Number of volumes to skip must be non-negative.")
        return value


class SVDOutputParams(BaseModel):
    output_file: str
    save_svd_data: bool = False
    generate_plots: bool = False


class SVDConfig(BaseModel):
    # Main configuration fields
    path_to_submissions: str
    voxel_size: float
    excluded_submissions: List[str] = []
    dtype: str = "float32"
    svd_max_rank: Optional[int] = None

    # Subdictionaries
    normalize_params: SVDNormalizeParams = SVDNormalizeParams()
    gt_params: Optional[SVDGtParams] = None
    output_params: SVDOutputParams

    @model_validator(mode="after")
    def check_path_to_submissions(self):
        path_to_submissions = self.path_to_submissions
        excluded_submissions = self.excluded_submissions

        if not os.path.exists(path_to_submissions):
            raise ValueError(f"Could not find path {path_to_submissions}.")

        submission_files = []
        for file in os.listdir(path_to_submissions):
            if file.endswith(".pt") and "submission" in file:
                submission_files.append(file)
        if len(submission_files) == 0:
            raise ValueError(f"No submission files found in {path_to_submissions}.")

        submission_files = []
        for file in os.listdir(path_to_submissions):
            if file.endswith(".pt") and "submission" in file:
                if file in excluded_submissions:
                    continue
                submission_files.append(file)

        if len(submission_files) == 0:
            raise ValueError(
                f"No submission files found after excluding {excluded_submissions}."
            )

        return self

    @field_validator("dtype")
    def check_dtype(cls, value):
        if value not in ["float32", "float64"]:
            raise ValueError(f"Invalid dtype {value}.")
        return value

    @field_validator("svd_max_rank")
    def check_svd_max_rank(cls, value):
        if value < 1 and value is not None:
            raise ValueError("Max rank must be at least 1.")
        return value

    @field_validator("voxel_size")
    def check_voxel_size(cls, value):
        if value <= 0:
            raise ValueError("Voxel size must be positive.")
        return value
