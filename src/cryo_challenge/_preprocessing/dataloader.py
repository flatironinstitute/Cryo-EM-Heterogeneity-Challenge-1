import os
import glob
from natsort import natsorted
import numpy as np
import torch
from torch.utils.data import Dataset
import mrcfile


class SubmissionPreprocessingDataLoader(Dataset):
    """
    Dataset class for loading submission data for preprocessing

    Parameters:
    -----------
    submission_config (dict): dictionary containing submission configuration

    For information on submission config, refer to help documentation by calling
    SubmissionPreprocessingDataLoader.help()

    Returns:
    -------
    data_dict (dict): dictionary containing submission data
    """

    def __init__(self, submission_config):
        self.submission_config = submission_config
        self.submission_paths, self.gt_path = self.extract_submission_paths()
        self.subs_index = [int(idx) for idx in list(self.submission_config.keys())[1:]]
        path_to_gt_ref = os.path.join(
            self.gt_path, self.submission_config["gt"]["ref_align_fname"]
        )

        try:
            self.vol_gt_ref = mrcfile.open(path_to_gt_ref, mode="r").data.astype(
                np.float32
            )

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Reference ground truth volume {path_to_gt_ref} not found"
            )

    def validate_submission_config(self):
        if "gt" not in self.submission_config.keys():
            raise ValueError("Ground truth information not found in submission config")

        for key, value in self.submission_config.items():
            if key == "gt":
                if "path" not in value.keys():
                    raise ValueError("Path not found for ground truth")
                if "box_size" not in value.keys():
                    raise ValueError("Box size not found for ground truth")
                if "pixel_size" not in value.keys():
                    raise ValueError("Pixel size not found for ground truth")
                continue
            else:
                if "path" not in value.keys():
                    raise ValueError(f"Path not found for submission {key}")
                if "id" not in value.keys():
                    raise ValueError(f"ID not found for submission {key}")
                if "box_size" not in value.keys():
                    raise ValueError(f"Box size not found for submission {key}")
                if "pixel_size" not in value.keys():
                    raise ValueError(f"Pixel size not found for submission {key}")
                if "align" not in value.keys():
                    raise ValueError(f"Align not found for submission {key}")
                if "populations_file" not in value.keys():
                    raise ValueError(f"Population file not found for submission {key}")

                if not os.path.exists(value["path"]):
                    raise ValueError(f"Path {value['path']} does not exist")

                if not os.path.isdir(value["path"]):
                    raise ValueError(f"Path {value['path']} is not a directory")

        ids = list(self.submission_config.keys())[1:]
        if ids != list(range(len(ids))):
            raise ValueError(
                "Submission IDs should be integers starting from 0 and increasing by 1"
            )

        return

    @classmethod
    def help(cls):
        print("Help documentation for SubmissionPreprocessingDataLoader")
        print(
            "SubmissionPreprocessingDataLoader - Dataset class for loading submission data for preprocessing"  # noqa: E501
        )
        print("Parameters:")
        print(
            "submission_config (dict): dictionary containing submission configuration"
        )  # noqa: E501
        print("Submission config should contain the following")
        print("gt: dictionary containing ground truth information")
        print("gt: {")
        print("    path: path to ground truth volumes")
        print("    id: id of ground truth volume")
        print("    box_size: box size of ground truth volume")
        print("    pixel_size: pixel size of ground truth volume")
        print(
            "    align: 0 or 1, 1 if submission needs to be aligned to ground truth, 0 otherwise"  # noqa: E501
        )
        print("}")
        print("submissions: dictionary containing submission information")
        print("submissions: {")
        print("    id: {")
        print("        path: path to submission volume")
        print("        id: id of submission volume")
        print("        box_size: box size of submission volume")
        print("        pixel_size: pixel size of submission volume")
        print(
            "        align: 0 or 1, 1 if submission needs to be aligned to ground truth, 0 otherwise"  # noqa: E501
        )
        print("    }")
        print("}")
        print("id should be an integer starting from 0 and increasing by 1")
        print("Example usage:")
        print("submission_config = {")
        print("    'gt': {")
        print("        'path': 'path_to_gt_volume',")
        print("        'id': 0,")
        print("        'box_size': 64,")
        print("        'pixel_size': 1.0,")
        print("        'align': 0")
        print("    },")
        print("    1: {")
        print("        'path': 'path_to_submission_volume',")
        print("        'id': 1,")
        print("        'box_size': 64,")
        print("        'pixel_size': 1.0,")
        print("        'align': 0")
        print("    }")
        print("}")
        return

    def extract_submission_paths(self):
        submission_paths = []
        for key, value in self.submission_config.items():
            if key == "gt":
                gt_path = value["path"]

            else:
                submission_paths.append(value["path"])
        return submission_paths, gt_path

    def __len__(self):
        return len(self.submission_paths)

    def __getitem__(self, idx):
        vol_paths = natsorted(
            glob.glob(os.path.join(self.submission_paths[idx], "*.mrc"))
        )
        vol_paths = [vol_path for vol_path in vol_paths if "mask" not in vol_path]

        assert len(vol_paths) > 0, "No volumes found in submission directory"

        populations = np.loadtxt(self.submission_config["populations_file"]).astype(
            float
        )
        populations = torch.from_numpy(populations)

        vol0 = mrcfile.open(vol_paths[0], mode="r")
        volumes = torch.zeros(
            (
                len(vol_paths),
                vol0.data.shape[0],
                vol0.data.shape[1],
                vol0.data.shape[2],
            ),
            dtype=torch.float32,
        )
        headers = []
        voxel_sizes = []

        for i, vol_path in enumerate(vol_paths):
            vol = mrcfile.open(vol_path, mode="r")
            volumes[i] = torch.tensor(vol.data.copy())
            headers.append(vol.header)
            voxel_sizes.append(vol.voxel_size)

        data_dict = {
            "volumes": volumes,
            "headers": headers,
            "vol_paths": vol_paths,
            "voxel_sizes": voxel_sizes,
            "config": self.submission_config[str(self.subs_index[idx])],
            "populations": populations,
        }

        return data_dict
