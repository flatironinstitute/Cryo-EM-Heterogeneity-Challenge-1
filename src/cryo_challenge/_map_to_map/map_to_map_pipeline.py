import mrcfile
import pandas as pd
import pickle
import torch

from ..data._validation.output_validators import MapToMapResultsValidator
from .._map_to_map.map_to_map_distance import (
    GT_Dataset,
    FSCDistance,
    FSCDistanceLowMemory,
    Correlation,
    CorrelationLowMemory,
    L2DistanceNorm,
    L2DistanceNormLowMemory,
    BioEM3dDistance,
    BioEM3dDistanceLowMemory,
    FSCResDistance,
    FSCResDistanceLowMemory,
)


AVAILABLE_MAP2MAP_DISTANCES = {
    "fsc": FSCDistance,
    "corr": Correlation,
    "l2": L2DistanceNorm,
    "bioem": BioEM3dDistance,
    "res": FSCResDistance,
}

AVAILABLE_MAP2MAP_DISTANCES_LOW_MEMORY = {
    "corr_low_memory": CorrelationLowMemory,
    "l2_low_memory": L2DistanceNormLowMemory,
    "bioem_low_memory": BioEM3dDistanceLowMemory,
    "fsc_low_memory": FSCDistanceLowMemory,
    "res_low_memory": FSCResDistanceLowMemory,
}


def run(config):
    """
    Compare a submission to ground truth.
    """

    map_to_map_distances = {
        distance_label: distance_class(config)
        for distance_label, distance_class in AVAILABLE_MAP2MAP_DISTANCES.items()
        if distance_label in config["analysis"]["metrics"]
    }

    map_to_map_distances_low_memory = {
        distance_label: distance_class(config)
        for distance_label, distance_class in AVAILABLE_MAP2MAP_DISTANCES_LOW_MEMORY.items()
        if distance_label in config["analysis"]["metrics"]
    }

    assert len(map_to_map_distances_low_memory) == 0 or len(map_to_map_distances) == 0

    if len(map_to_map_distances_low_memory) > 0:
        map_to_map_distances = map_to_map_distances_low_memory
        low_memory_mode = True
    else:
        low_memory_mode = False

    if not low_memory_mode:
        n_pix = config["data"]["n_pix"]

    submission = torch.load(config["data"]["submission"]["fname"])
    submission_volume_key = config["data"]["submission"]["volume_key"]
    submission_metadata_key = config["data"]["submission"]["metadata_key"]
    label_key = config["data"]["submission"]["label_key"]
    user_submission_label = submission[label_key]

    metadata_gt = pd.read_csv(config["data"]["ground_truth"]["metadata"])

    results_dict = {}
    results_dict["config"] = config
    results_dict["user_submitted_populations"] = (
        submission[submission_metadata_key] / submission[submission_metadata_key].sum()
    )

    maps_user_flat = submission[submission_volume_key].reshape(
        len(submission["volumes"]), -1
    )
    if low_memory_mode:
        maps_gt_flat = GT_Dataset(config["data"]["ground_truth"]["volumes"])
    else:
        maps_gt_flat = torch.load(config["data"]["ground_truth"]["volumes"]).reshape(
            -1, n_pix**3
        )

    if config["data"]["mask"]["do"]:
        mask = (
            mrcfile.open(config["data"]["mask"]["volume"]).data.astype(bool).flatten()
        )
        if not low_memory_mode:
            maps_gt_flat = maps_gt_flat[:, mask]
        maps_user_flat = maps_user_flat[:, mask]
    else:
        if not low_memory_mode:
            maps_gt_flat.reshape(len(maps_gt_flat), -1, inplace=True)
        maps_user_flat.reshape(len(maps_gt_flat), -1, inplace=True)

    if config["analysis"]["normalize"]["do"]:
        if config["analysis"]["normalize"]["method"] == "median_zscore":
            if not low_memory_mode:
                maps_gt_flat -= maps_gt_flat.median(dim=1, keepdim=True).values
            if not low_memory_mode:
                maps_gt_flat /= maps_gt_flat.std(dim=1, keepdim=True)
            maps_user_flat -= maps_user_flat.median(dim=1, keepdim=True).values
            maps_user_flat /= maps_user_flat.std(dim=1, keepdim=True)
        else:
            raise NotImplementedError(
                f"Normalization method {config['analysis']['normalize']['method']} not implemented."
            )

    computed_assets = {}
    results_dict["mask"] = mask
    for distance_label, map_to_map_distance in map_to_map_distances.items():
        if distance_label in config["analysis"]["metrics"]:  # TODO: can remove
            print("cost matrix", distance_label)
            print("maps_gt_flat", maps_gt_flat.shape)
            print("maps_user_flat", maps_user_flat.shape)

            map_to_map_distance.distance_matrix_precomputation(
                maps_gt_flat, maps_user_flat
            )
            cost_matrix = map_to_map_distance.get_distance_matrix(
                maps_gt_flat,
                maps_user_flat,
                global_store_of_running_results=results_dict,
            )
            computed_assets = map_to_map_distance.get_computed_assets(
                maps_gt_flat,
                maps_user_flat,
                global_store_of_running_results=results_dict,
            )
            computed_assets.update(computed_assets)

            cost_matrix_df = pd.DataFrame(
                cost_matrix, columns=None, index=metadata_gt.populations.tolist()
            )

            # output results
            single_distance_results_dict = {
                "cost_matrix": cost_matrix_df,
                "user_submission_label": user_submission_label,
                "computed_assets": computed_assets,
            }

            results_dict[distance_label] = single_distance_results_dict

    # Validate before saving
    _ = MapToMapResultsValidator.from_dict(results_dict)

    with open(config["output"], "wb") as f:
        pickle.dump(results_dict, f)

    return results_dict
