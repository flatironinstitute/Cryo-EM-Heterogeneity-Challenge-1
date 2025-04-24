import pandas as pd
import pickle
import torch
import logging
from collections import OrderedDict

from ..config_validation._map_to_map_validation import MapToMapResultsValidator
from .map_to_map_distance import (
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

ORDER_TO_ENDURE_FSC_BEFORE_RES = [
    "corr",
    "l2",
    "bioem",
    "fsc",
    "res",
    "zernike3d",
    "gromov_wasserstein",
    "procrustes_wasserstein",
]

AVAILABLE_MAP2MAP_DISTANCES = {
    "corr": Correlation,
    "l2": L2DistanceNorm,
    "bioem": BioEM3dDistance,
    "fsc": FSCDistance,
    "res": FSCResDistance,
    "zernike3d": Zernike3DDistance,
    "gromov_wasserstein": GromovWassersteinDistance,
    "procrustes_wasserstein": ProcrustesWassersteinDistance,
}

AVAILABLE_MAP2MAP_DISTANCES = OrderedDict(
    (k, AVAILABLE_MAP2MAP_DISTANCES[k]) for k in ORDER_TO_ENDURE_FSC_BEFORE_RES
)


def run(config):
    """
    Compare a submission to ground truth.
    """

    for key, value in config["metrics"].copy().items():
        if len(value) == 0:
            del config["metrics"][key]
            print(f"Removing {key} from config['metrics'] because it is empty")

    logger.info("Running map-to-map analysis")
    map_to_map_distances = {
        distance_label: distance_class(config)
        for distance_label, distance_class in AVAILABLE_MAP2MAP_DISTANCES.items()
        if distance_label in config["metrics"]
    }

    do_low_memory_mode = config["metrics"]["shared_params"]["low_memory"] is not None

    logger.info("Loading submission")
    submission = torch.load(
        config["data_params"]["submission_params"]["path_to_submission_file"],
        weights_only=False,
    )
    submission_volume_key = config["data_params"]["submission_params"]["volume_key"]
    submission_metadata_key = config["data_params"]["submission_params"]["metadata_key"]
    label_key = config["data_params"]["submission_params"]["label_key"]
    user_submission_label = submission[label_key]

    metadata_gt = pd.read_csv(
        config["data_params"]["ground_truth_params"]["path_to_metadata"]
    )

    results_dict = {}
    results_dict["config"] = config

    results_dict["user_submitted_populations"] = torch.tensor(
        submission[submission_metadata_key] / submission[submission_metadata_key].sum()
    )

    maps_user_flat = submission[submission_volume_key].reshape(
        len(submission["volumes"]), -1
    )

    logger.info("Loading ground truth")
    maps_gt_flat = torch.load(
        config["data_params"]["ground_truth_params"]["path_to_volumes"],
        mmap=do_low_memory_mode,
        weights_only=False,
    )

    computed_assets = {}
    for distance_label, map_to_map_distance in map_to_map_distances.items():
        print(f"Computing {distance_label} distance")
        if distance_label in config["metrics"].keys():
            print(f"Computing {distance_label} distance")
            config_metric = config["metrics"][distance_label]
            print(f"with config {config_metric}")
        else:
            del config["metrics"][distance_label]

        if (
            distance_label in config["metrics"].keys()
            and len(config["metrics"][distance_label]) != 0
        ):  # TODO: can remove
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
                "user_submission_label": user_submission_label,
                "computed_assets": computed_assets,
            }

            results_dict[distance_label] = single_distance_results_dict
    print("results_dict.keys()", results_dict.keys())
    print("results_dict['config']", results_dict["config"])

    # Validate before saving
    results_dict = dict(MapToMapResultsValidator(**results_dict).model_dump())

    with open(config["path_to_output_file"], "wb") as f:
        pickle.dump(results_dict, f)

    return results_dict
