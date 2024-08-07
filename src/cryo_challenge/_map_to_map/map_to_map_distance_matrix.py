import mrcfile
import pandas as pd
import pickle
import torch

from .map_to_map_distance import (
    compute_bioem3d_cost,
    compute_cost_corr,
    compute_cost_fsc_chunk,
    compute_cost_l2,
)
from ..data._validation.output_validators import MapToMapResultsValidator


def vmap_distance(
    maps_gt,
    maps_submission,
    map_to_map_distance,
    chunk_size_gt=None,
    chunk_size_submission=None,
):
    return torch.vmap(
        lambda maps_gt: torch.vmap(
            lambda maps_submission: map_to_map_distance(maps_gt, maps_submission),
            chunk_size=chunk_size_submission,
        )(maps_submission),
        chunk_size=chunk_size_gt,
    )(maps_gt)


def run(config):
    """
    Compare a submission to ground truth.
    """

    n_pix = config["data"]["n_pix"]

    submission = torch.load(config["data"]["submission"]["fname"])
    submission_volume_key = config["data"]["submission"]["volume_key"]
    submission_metadata_key = config["data"]["submission"]["metadata_key"]
    label_key = config["data"]["submission"]["label_key"]
    user_submission_label = submission[label_key]

    # n_trunc = 10
    metadata_gt = pd.read_csv(config["data"]["ground_truth"]["metadata"])#[:n_trunc]

    results_dict = {}
    results_dict["config"] = config
    results_dict["user_submitted_populations"] = (
        submission[submission_metadata_key] / submission[submission_metadata_key].sum()
    )

    cost_funcs_d = {
        "fsc": compute_cost_fsc_chunk,
        "corr": compute_cost_corr,
        "l2": compute_cost_l2,
        "bioem": compute_bioem3d_cost,
    }

    maps_user_flat = submission[submission_volume_key].reshape(
        len(submission["volumes"]), -1
    )
    maps_gt_flat = torch.load(config["data"]["ground_truth"]["volumes"]).reshape(
        -1, n_pix**3
    )
    # maps_gt_flat = torch.randn(n_trunc, n_pix**3)

    if config["data"]["mask"]["do"]:
        mask = (
            mrcfile.open(config["data"]["mask"]["volume"]).data.astype(bool).flatten()
        )
        maps_gt_flat = maps_gt_flat[:, mask]
        maps_user_flat = maps_user_flat[:, mask]
    else:
        maps_gt_flat.reshape(len(maps_gt_flat), -1, inplace=True)
        maps_user_flat.reshape(len(maps_gt_flat), -1, inplace=True)

    if config["analysis"]["normalize"]["do"]:
        if config["analysis"]["normalize"]["method"] == "median_zscore":
            maps_gt_flat -= maps_gt_flat.median(dim=1, keepdim=True).values
            maps_gt_flat /= maps_gt_flat.std(dim=1, keepdim=True)
            maps_user_flat -= maps_user_flat.median(dim=1, keepdim=True).values
            maps_user_flat /= maps_user_flat.std(dim=1, keepdim=True)

    computed_assets = {}
    for cost_label, cost_func in cost_funcs_d.items():
        if cost_label in config["analysis"]["metrics"]:  # TODO: can remove
            print("cost matrix", cost_label)

            if (
                cost_label == "fsc"
            ):  # TODO: make pydantic (include base class). type hint inputs to this (what it needs like gt volumes and populations) # noqa: E501
                maps_gt_flat_cube = torch.zeros(len(maps_gt_flat), n_pix**3)
                maps_gt_flat_cube[:, mask] = maps_gt_flat
                maps_user_flat_cube = torch.zeros(len(maps_user_flat), n_pix**3)
                maps_user_flat_cube[:, mask] = maps_user_flat
                cost_matrix, fsc_matrix = cost_func(
                    maps_gt_flat_cube, maps_user_flat_cube, n_pix
                )
                cost_matrix = cost_matrix.numpy()
                computed_assets["fsc_matrix"] = fsc_matrix
            else:
                cost_matrix = vmap_distance(
                    maps_gt_flat,
                    maps_user_flat,
                    cost_func,
                    chunk_size_gt=config["analysis"]["chunk_size_gt"],
                    chunk_size_submission=config["analysis"]["chunk_size_submission"],
                ).numpy()

            cost_matrix_df = pd.DataFrame(
                cost_matrix, columns=None, index=metadata_gt.populations.tolist()
            )

            # output results
            single_distance_results_dict = {
                "cost_matrix": cost_matrix_df,
                "user_submission_label": user_submission_label,
                "computed_assets": computed_assets,
            }

            results_dict[cost_label] = single_distance_results_dict

    # Validate before saving
    _ = MapToMapResultsValidator.from_dict(results_dict)

    with open(config["output"], "wb") as f:
        pickle.dump(results_dict, f)

    return results_dict
