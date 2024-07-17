import torch
from typing import Tuple
import yaml
import argparse
import os

from .svd_utils import get_vols_svd, project_vols_to_svd
from ..data._io.svd_io_utils import load_volumes, load_ref_vols
from ..data._validation.config_validators import validate_config_svd


def run_svd_with_ref(
    volumes: torch.tensor, ref_volumes: torch.tensor
) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """
    Compute the singular value decomposition of the reference volumes and project the input volumes onto the right singular vectors of the reference volumes.

    Parameters
    ----------
    volumes: torch.tensor
        Tensor of shape (n_volumes, n_x, n_y, n_z) containing the volumes to be projected.
    ref_volumes: torch.tensor
        Tensor of shape (n_volumes_ref, n_x, n_y, n_z) containing the reference volumes.

    Returns
    -------
    U: torch.tensor
        Left singular vectors of the reference volumes.
    S: torch.tensor
        Singular values of the reference volumes.
    V: torch.tensor
        Right singular vectors of the reference volumes.
    coeffs: torch.tensor
        Coefficients of the input volumes projected onto the right singular vectors of the reference volumes.

    Examples
    --------
    >>> volumes = torch.randn(10, 32, 32, 32)
    >>> ref_volumes = torch.randn(10, 32, 32, 32)
    >>> U, S, V, coeffs = run_svd_with_ref(volumes, ref_volumes)
    """  # noqa: E501

    assert volumes.ndim == 4, "Input volumes must have shape (n_volumes, n_x, n_y, n_z)"
    assert volumes.shape[0] > 0, "Input volumes must have at least one volume"
    assert (
        ref_volumes.ndim == 4
    ), "Reference volumes must have shape (n_volumes, n_x, n_y, n_z)"
    assert ref_volumes.shape[0] > 0, "Reference volumes must have at least one volume"
    assert (
        volumes.shape[1:] == ref_volumes.shape[1:]
    ), "Input volumes and reference volumes must have the same shape"

    U, S, V = get_vols_svd(ref_volumes)
    coeffs = project_vols_to_svd(volumes, V)
    coeffs_ref = U @ torch.diag(S)

    return U, S, V, coeffs, coeffs_ref


def run_svd_all_vs_all(volumes: torch.tensor):
    """
    Compute the singular value decomposition of the input volumes.

    Parameters
    ----------
    volumes: torch.tensor
        Tensor of shape (n_volumes, n_x, n_y, n_z) containing the volumes to be decomposed.

    Returns
    -------
    U: torch.tensor
        Left singular vectors of the input volumes.
    S: torch.tensor
        Singular values of the input volumes.
    V: torch.tensor
        Right singular vectors of the input volumes.
    coeffs: torch.tensor
        Coefficients of the input volumes projected onto the right singular vectors.

    Examples
    --------
    >>> volumes = torch.randn(10, 32, 32, 32)
    >>> U, S, V, coeffs = run_svd_all_vs_all(volumes)
    """  # noqa: E501
    U, S, V = get_vols_svd(volumes)
    coeffs = U @ torch.diag(S)
    return U, S, V, coeffs


def run_all_vs_all_pipeline(config: dict):
    """
    Run the all-vs-all SVD pipeline. Load the volumes, compute the SVD, and save the results.

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration options for the pipeline.

    The results are saved in a dictionary with the following
    keys:
    - coeffs: Coefficients of the input volumes projected onto the right singular vectors.
    - metadata: Dictionary containing the populations and indices of each submission (see Tutorial for details).
    - vols_per_submission: Dictionary containing the number of volumes per submission.

    if save_volumes set to True
    - volumes: Tensor of shape (n_volumes, n_x, n_y, n_z) containing the volumes.
    - mean_volumes: Tensor of shape (n_submissions, n_x, n_y, n_z) containing the mean volume of each submission.
    * Note: the volumes will be downsampled, normalized and mean-removed

    if save_svd_matrices set to True
    - U: Left singular vectors of the input volumes.
    - S: Singular values of the input volumes.
    - V: Right singular vectors of the input volumes.

    """  # noqa: E501

    dtype = torch.float32 if config["dtype"] == "float32" else torch.float64
    volumes, mean_volumes, metadata = load_volumes(
        box_size_ds=config["box_size_ds"],
        submission_list=config["submission_list"],
        path_to_submissions=config["path_to_volumes"],
        dtype=dtype,
    )

    U, S, V, coeffs = run_svd_all_vs_all(volumes=volumes)

    output_dict = {
        "coeffs": coeffs,
        "metadata": metadata,
        "config": config,
    }

    if config["output_options"]["save_volumes"]:
        output_dict["volumes"] = volumes
        output_dict["mean_volumes"] = mean_volumes

    if config["output_options"]["save_svd_matrices"]:
        output_dict["U"] = U
        output_dict["S"] = S
        output_dict["V"] = V

    output_file = os.path.join(
        config["output_options"]["output_path"], "svd_results.pt"
    )
    torch.save(output_dict, output_file)

    return output_dict


def run_all_vs_ref_pipeline(config: dict):
    """
    Run the all-vs-ref SVD pipeline. Load the volumes, compute the SVD, and save the results.

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration options for the pipeline.

    The results are saved in a dictionary with the following
    keys:
    - coeffs: Coefficients of the input volumes projected onto the right singular vectors.
    - populations: Dictionary containing the populations of each submission.
    - vols_per_submission: Dictionary containing the number of volumes per submission.

    if save_volumes set to True
    - volumes: Tensor of shape (n_volumes, n_x, n_y, n_z) containing the volumes.
    - mean_volumes: Tensor of shape (n_submissions, n_x, n_y, n_z) containing the mean volume of each submission.
    - ref_volumes: Tensor of shape (n_volumes_ref, n_x, n_y, n_z) containing the reference volumes.
    - mean_ref_volumes: Tensor of shape (n_x, n_y, n_z) containing the mean reference volume.
    * Note: volumes and ref_volumes will be downsampled, normalized and mean-removed

    if save_svd_matrices set to True
    - U: Left singular vectors of the reference volumes.
    - S: Singular values of the reference volumes.
    - V: Right singular vectors of the reference volumes.

    """  # noqa: E501

    dtype = torch.float32 if config["dtype"] == "float32" else torch.float64

    ref_volumes, mean_volume = load_ref_vols(
        config,
        dtype=dtype,
    )

    volumes, mean_volumes, metadata = load_volumes(
        box_size_ds=config["box_size_ds"],
        submission_list=config["submission_list"],
        path_to_submissions=config["path_to_volumes"],
        dtype=dtype,
    )

    U, S, V, coeffs, coeffs_ref = run_svd_with_ref(
        volumes=volumes, ref_volumes=ref_volumes
    )

    output_dict = {
        "coeffs": coeffs,
        "coeffs_ref": coeffs_ref,
        "metadata": metadata,
        "config": config,
    }

    if config["output_options"]["save_volumes"]:
        output_dict["volumes"] = volumes
        output_dict["mean_volumes"] = mean_volumes
        output_dict["ref_volumes"] = ref_volumes
        output_dict["mean_ref_volume"] = mean_volume

    if config["output_options"]["save_svd_matrices"]:
        output_dict["U"] = U
        output_dict["S"] = S
        output_dict["V"] = V

    output_file = os.path.join(
        config["output_options"]["output_path"], "svd_results.pt"
    )
    torch.save(output_dict, output_file)

    return output_dict


def run_svd_pipeline():
    parser = argparse.ArgumentParser(description="Run SVD on volumes")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the config (yaml) file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    validate_config_svd(config)

    return


if __name__ == "__main__":
    run_svd_pipeline()
