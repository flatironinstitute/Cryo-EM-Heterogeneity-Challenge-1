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


class MapToMapDistance:
    def __init__(self, config):
        self.config = config

    def get_distance(self, map1, map2):
        raise NotImplementedError()

    def get_distance_matrix(self, maps1, maps2):
        chunk_size_submission = self.config["analysis"]["chunk_size_submission"]
        chunk_size_gt = self.config["analysis"]["chunk_size_gt"]
        distance_matrix = torch.vmap(
            lambda maps1: torch.vmap(
                lambda maps2: self.get_distance(maps1, maps2),
                chunk_size=chunk_size_submission,
            )(maps2),
            chunk_size=chunk_size_gt,
        )(maps1)

        return distance_matrix
    
    def get_computed_assets(self, maps1, maps2):
        return {}

class L2DistanceNorm(MapToMapDistance):
    def __init__(self, config):
        super().__init__(config)

    def get_distance(self, map1, map2):
        return torch.norm(map1 - map2)**2
    
class L2DistanceSum(MapToMapDistance):
    def __init__(self, config):
        super().__init__(config)

    def get_distance(self, map1, map2):
        return compute_cost_l2(map1, map2)
    
class Correlation(MapToMapDistance):
    def __init__(self, config):
        super().__init__(config)

    def get_distance(self, map1, map2):
        return compute_cost_corr(map1, map2) 

class BioEM3dDistance(MapToMapDistance):
    def __init__(self, config):
        super().__init__(config)

    def get_distance(self, map1, map2):
        return compute_bioem3d_cost(map1, map2) 
    
class FSCDistance(MapToMapDistance):
    def __init__(self, config):
        super().__init__(config)
        
    def get_distance_matrix(self, maps1, maps2): # custom method
        maps_gt_flat = maps1
        maps_user_flat = maps2
        n_pix = self.config["data"]["n_pix"]
        maps_gt_flat_cube = torch.zeros(len(maps_gt_flat), n_pix**3)
        mask = (
            mrcfile.open(self.config["data"]["mask"]["volume"]).data.astype(bool).flatten()
        )
        maps_gt_flat_cube[:, mask] = maps_gt_flat
        maps_user_flat_cube = torch.zeros(len(maps_user_flat), n_pix**3)
        maps_user_flat_cube[:, mask] = maps_user_flat
        
        cost_matrix, fsc_matrix =  compute_cost_fsc_chunk(maps_gt_flat_cube, maps_user_flat_cube, n_pix)
        self.stored_computed_assets = {'fsc_matrix': fsc_matrix}
        return cost_matrix
    
    def get_computed_assets(self, maps1, maps2):
        return self.stored_computed_assets # must run get_distance_matrix first
    
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

    metadata_gt = pd.read_csv(config["data"]["ground_truth"]["metadata"])

    results_dict = {}
    results_dict["config"] = config
    results_dict["user_submitted_populations"] = (
        submission[submission_metadata_key] / submission[submission_metadata_key].sum()
    )

    cost_funcs_d = {
        "fsc": FSCDistance(config),
        "corr": Correlation(config),
        "l2": L2DistanceSum(config),
        "bioem": BioEM3dDistance(config),
    }

    maps_user_flat = submission[submission_volume_key].reshape(
        len(submission["volumes"]), -1
    )
    maps_gt_flat = torch.load(config["data"]["ground_truth"]["volumes"]).reshape(
        -1, n_pix**3
    )

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
    for cost_label, map_to_map_distance in cost_funcs_d.items():
        if cost_label in config["analysis"]["metrics"]:  # TODO: can remove
            print("cost matrix", cost_label)

            cost_matrix = map_to_map_distance.get_distance_matrix(
                maps_gt_flat, maps_user_flat
            ).numpy()
            computed_assets = map_to_map_distance.get_computed_assets(
                maps_gt_flat, maps_user_flat
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

            results_dict[cost_label] = single_distance_results_dict

    # Validate before saving
    _ = MapToMapResultsValidator.from_dict(results_dict)

    with open(config["output"], "wb") as f:
        pickle.dump(results_dict, f)

    return results_dict
