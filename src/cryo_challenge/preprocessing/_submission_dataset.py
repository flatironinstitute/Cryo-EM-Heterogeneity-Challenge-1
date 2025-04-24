import os
import glob
from natsort import natsorted
import numpy as np
import torch
from torch.utils.data import Dataset
import mrcfile
import pathlib

from typing import List
from ..config_validation._preprocessing_validators import (
    PreprocessingDatasetReferenceConfig,
    PreprocessingDatasetSubmissionConfig,
)


class SubmissionPreprocessingDataset(Dataset):
    """
    A dataset class for loading the raw volumes and populations from the submission dataset.
    """

    config_for_ref_vol: PreprocessingDatasetReferenceConfig
    configs_for_vol_sets: List[PreprocessingDatasetSubmissionConfig]

    def __init__(
        self,
        config_for_ref_vol: PreprocessingDatasetReferenceConfig,
        configs_for_vol_sets: List[PreprocessingDatasetSubmissionConfig],
    ):
        """
        Config arguments must be validate by our Pydantic validators.

        **Arguments**
        ----------
        config_for_ref_vol : PreprocessingDatasetReferenceConfig
            Configuration for the reference volume set.
        configs_for_vol_sets : List[PreprocessingDatasetSubmissionConfig]
            List of configurations for the volume sets.
        """

        self.config_for_ref_vol = config_for_ref_vol
        self.configs_for_vol_sets = configs_for_vol_sets

    def __len__(self):
        return len(self.configs_for_vol_sets)

    def __getitem__(self, idx):
        populations = torch.from_numpy(
            np.loadtxt(self.configs_for_vol_sets[idx].path_to_populations_file)
        )

        volumes = _load_all_volumes_in_directory(
            self.configs_for_vol_sets[idx].path_to_volumes,
            self.configs_for_vol_sets[idx].box_size,
        )

        data_dict = {
            "volumes": volumes,
            "config": self.configs_for_vol_sets[idx],
            "populations": populations,
        }

        return data_dict


def _load_all_volumes_in_directory(
    path_to_volumes: List[pathlib.Path], box_size: int
) -> torch.Tensor:
    """
    Load volumes from a list of paths.

    Parameters
    ----------
    path_to_volumes : List[pathlib.Path]
        Path to the directory containing the volumes.
    box_size : int
        Size of the box for the volumes.

    Returns
    -------
    torch.Tensor
        Tensor containing the loaded volumes.
    """
    list_of_vol_paths = _get_volumes_paths_from_directory(path_to_volumes)[:3]

    volumes = torch.empty(
        (len(list_of_vol_paths), box_size, box_size, box_size), dtype=torch.float32
    )

    for i, vol_path in enumerate(list_of_vol_paths):
        with mrcfile.open(vol_path, mode="r") as vol:
            volumes[i] = torch.tensor(vol.data.copy())

    return volumes.to(torch.float32)


def _get_volumes_paths_from_directory(
    path_to_volumes: pathlib.Path, exclude_mask=True
) -> List[pathlib.Path]:
    """
    Get the list of paths to all volume files in the directory.

    Parameters
    ----------
    path_to_volumes : pathlib.Path
        Path to the directory containing the volumes.
    exclude_mask : bool, optional
        Whether to exclude files containing "mask" in their name, by default True.

    Returns
    -------
    List[pathlib.Path]
        List of paths to the volumes.
    """
    vol_paths = natsorted(glob.glob(os.path.join(path_to_volumes, "*.mrc")))
    if exclude_mask:
        # Exclude files containing "mask" in their name
        vol_paths = [vol_path for vol_path in vol_paths if "mask" not in vol_path]

    return vol_paths
