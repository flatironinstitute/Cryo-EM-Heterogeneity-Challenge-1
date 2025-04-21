import pandas as pd
import pickle
import torch
import logging

from ..data._validation.output_validators import MapToMapResultsValidator
from .._map_to_map.map_to_map_distance import (
    FSCDistance,
    Correlation,
    L2DistanceNorm,
    BioEM3dDistance,
    FSCResDistance,
    Zernike3DDistance,
    GromovWassersteinDistance,
    ProcrustesWassersteinDistance,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AVAILABLE_MAP2MAP_DISTANCES = {
    "fsc": FSCDistance,
    "corr": Correlation,
    "l2": L2DistanceNorm,
    "bioem": BioEM3dDistance,
    "res": FSCResDistance,
    "zernike3d": Zernike3DDistance,
    "gromov_wasserstein": GromovWassersteinDistance,
    "procrustes_wasserstein": ProcrustesWassersteinDistance,
}


def run(config):
    """
    Compare a submission to ground truth.
    """

    logger.info("Running map-to-map analysis")
    map_to_map_distances = {
        distance_label: distance_class(config)
        for distance_label, distance_class in AVAILABLE_MAP2MAP_DISTANCES.items()
        if distance_label in config["analysis"]["metrics"]
    }

    do_low_memory_mode = config["analysis"]["low_memory"]["do"]

    logger.info("Loading submission")
    submission = torch.load(config["data"]["submission"]["fname"], weights_only=False)
    submission_volume_key = config["data"]["submission"]["volume_key"]
    submission_metadata_key = config["data"]["submission"]["metadata_key"]
    label_key = config["data"]["submission"]["label_key"]
    user_submission_label = submission[label_key]

    metadata_gt = pd.read_csv(config["data"]["ground_truth"]["metadata"])

    self_results_dict, results_dict = {}, {}
    results_dict["config"] = config
    self_results_dict["config"] = config

    results_dict["user_submitted_populations"] = (
        submission[submission_metadata_key] / submission[submission_metadata_key].sum()
    )

    maps_user_flat = submission[submission_volume_key].reshape(
        len(submission["volumes"]), -1
    )

    logger.info("Loading ground truth")
    maps_gt_flat = torch.load(
        config["data"]["ground_truth"]["volumes"],
        mmap=do_low_memory_mode,
        weights_only=False,
    )

    computed_assets = {}
    self_computed_assets = {}
    for distance_label, map_to_map_distance in map_to_map_distances.items():
        single_distance_results_dict = {}
        if distance_label in config["analysis"]["metrics"]:  # TODO: can remove
            logger.info(f"cost matrix: {distance_label}")

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
                "computed_assets": computed_assets,
            }

        if distance_label in config["analysis"]["self_metrics"]:
            map_to_map_distance.distance_matrix_precomputation(
                maps_user_flat, maps_user_flat
            )
            cost_matrix = map_to_map_distance.get_distance_matrix(
                maps_user_flat,
                maps_user_flat,
                global_store_of_running_results=self_results_dict,
            )
            self_computed_assets = map_to_map_distance.get_computed_assets(
                maps_user_flat,
                maps_user_flat,
                global_store_of_running_results=self_results_dict,
            )

            print("self_results_dict.keys()", self_results_dict.keys())

            self_computed_assets.update(self_computed_assets)

            print(
                distance_label,
                cost_matrix.shape,
                results_dict["user_submitted_populations"].shape,
            )
            cost_matrix_df = pd.DataFrame(
                cost_matrix,
                columns=None,
                index=results_dict["user_submitted_populations"].tolist(),
            )

            self_single_distance_results_dict = {
                "self_cost_matrix": cost_matrix_df,
                "computed_assets": self_computed_assets,
            }

            self_single_distance_results_dict_nooverwrite = {
                "self_cost_matrix": cost_matrix_df,
                "self_computed_assets": self_computed_assets,
            }

            self_results_dict[distance_label] = self_single_distance_results_dict

            single_distance_results_dict.update(
                self_single_distance_results_dict_nooverwrite
            )

        if (
            distance_label in config["analysis"]["metrics"]
            or distance_label not in config["analysis"]["self_metrics"]
        ):
            single_distance_results_dict.update(
                {"user_submission_label": user_submission_label}
            )

            results_dict[distance_label] = single_distance_results_dict
            print("results_dict.keys()", results_dict.keys())

    # Validate before saving
    _ = MapToMapResultsValidator.from_dict(results_dict)

    with open(config["output"], "wb") as f:
        pickle.dump(results_dict, f)

    return results_dict
