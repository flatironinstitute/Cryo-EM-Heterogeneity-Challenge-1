import torch
import os
import pickle
from dataclasses import dataclass
from glob import glob
import pandas as pd
from tqdm import tqdm

from cryo_challenge._metric_comparison.information_imbalance import (
    return_information_imbalace,
)


@dataclass
class Config:
    submission_fnames: list
    metrics: list
    output: str
    number_of_nearest_neighbors: int


def pearson_correlation(distance_matrix_i, distance_matrix_j):
    flattened_i = distance_matrix_i.flatten()
    flattened_j = distance_matrix_j.flatten()
    correlation_matrix = torch.corrcoef(torch.stack([flattened_i, flattened_j]))
    pearson_r = correlation_matrix[0, 1].item()
    return pearson_r


def run(config):
    metrics = config.metrics

    submissions_data = {}
    for fname in config.submission_fnames:
        with open(fname, "rb") as f:
            data = pickle.load(f)
            assert os.path.basename(fname) not in data, f"{fname}: {data.keys()}"
            submissions_data[os.path.basename(fname)] = data

    output = []

    for submission_basename, data in tqdm(submissions_data.items()):
        for metric in metrics:
            if metric not in data:
                raise ValueError(f"Metric {metric} not found in data")

        for idx_i, distance_method_i in enumerate(metrics):
            distance_matrix_i = torch.from_numpy(
                data[distance_method_i]["cost_matrix"].values
            )
            if distance_matrix_i == "corr":
                distance_matrix_i *= -1

            for idx_j, distance_method_j in enumerate(metrics):
                if idx_i <= idx_j:
                    continue
                distance_matrix_j = torch.from_numpy(
                    data[distance_method_j]["cost_matrix"].values
                )
                if distance_matrix_j == "corr":
                    distance_matrix_j *= -1

                ii = return_information_imbalace(
                    distance_matrix_i,
                    distance_matrix_j,
                    config.number_of_nearest_neighbors,
                )
                pearson_r = pearson_correlation(distance_matrix_i, distance_matrix_j)

                output.append(
                    {
                        "submission": submission_basename,
                        "distance_method_i": distance_method_i,
                        "distance_method_j": distance_method_j,
                        "ii_ij": ii[0].item(),
                        "ii_ji": ii[1].item(),
                        "pearson_r": pearson_r,
                    }
                )

    return output


if __name__ == "__main__":
    number_of_nearest_neighbors = 1
    config = Config(
        submission_fnames=glob(
            "/mnt/home/smbp/ceph/smbpchallenge/round2/set2/map_to_map/map_to_map_??.pkl"
        ),
        metrics=["fsc", "l2", "corr", "bioem", "res"],
        output=f"/mnt/home/smbp/ceph/smbpchallenge/round2/set2/metric_comparison/information_imbalance_k={number_of_nearest_neighbors}.csv",
        number_of_nearest_neighbors=number_of_nearest_neighbors,
    )

    pd.DataFrame(run(config)).to_csv(config.output, index=False)
