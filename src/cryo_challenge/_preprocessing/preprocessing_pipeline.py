import torch
import json
import os

from .align_utils import align_submission, threshold_submissions
from .crop_pad_utils import crop_pad_submission
from .fourier_utils import downsample_submission


def save_submission(volumes, populations, submission_id, submission_index, config):
    """
    Save preprocessed submission volumes

    Parameters:
    -----------
    volumes (torch.Tensor): submission volumes
        shape: (n_volumes, im_x, im_y, im_z)
    populations (list): populations
    submission_id (int): submission id
    submission_index (int): submission index

    Returns:
    --------
    submission_dict (dict): dictionary containing submission data
    """

    submission_dict = {
        "volumes": volumes,
        "populations": populations,
        "id": submission_id,
    }

    submission_path = os.path.join(
        config["output_path"], f"submission_{submission_index}.pt"
    )
    torch.save(submission_dict, submission_path)

    return submission_dict


def update_hash_table(hash_table_path, hash_table):
    if os.path.exists(hash_table_path):
        with open(hash_table_path, "r") as f:
            hash_table_old = json.load(f)
        hash_table_old.update(hash_table)

        with open(hash_table_path, "w") as f:
            json.dump(hash_table_old, f, indent=4)

    else:
        with open(hash_table_path, "w") as f:
            json.dump(hash_table, f, indent=4)

    return


def preprocess_submissions(submission_dataset, config):
    hash_table = {}
    box_size_gt = submission_dataset.submission_config["gt"]["box_size"]
    pixel_size_gt = submission_dataset.submission_config["gt"]["pixel_size"]
    vol_gt_ref = submission_dataset.vol_gt_ref

    for i in range(len(submission_dataset)):
        idx = submission_dataset.subs_index[i]

        sub_flavor = submission_dataset.submission_config[str(idx)]["flavor_name"]
        sub_name = submission_dataset.submission_config[str(idx)]["name"]
        hash_table[sub_flavor] = {
            "name": sub_name,
            "filename": f"submission_{idx}.pt",
        }

        print(f"Preprocessing submission {idx}...")

        pixel_size_sub = submission_dataset.submission_config[str(idx)]["pixel_size"]
        volumes = submission_dataset[i]["volumes"]

        print("    Cropping and padding submission")
        # pad/crop (prep for downsampling)
        volumes, success = crop_pad_submission(
            volumes, box_size_gt, pixel_size_sub, pixel_size_gt
        )
        if success == 0:
            print(
                "Submission is not suited for cropping-padding, pixel size less than ground truth pixel size"  # noqa: E501
            )
            continue

        # downsampling making sure pixel sizes and box sizes match
        print("    Downsampling submission")
        volumes = downsample_submission(volumes, box_size_gt)

        # thresholding
        print("    Thresholding submission")
        volumes = threshold_submissions(volumes, config["thresh_percentile"])

        # center submission
        # print("    Centering submission")
        # volumes = center_submission(volumes, pixel_size=pixel_size_gt)

        # flip handedness
        if submission_dataset.submission_config[str(idx)]["flip"] == 1:
            print("    Flipping handedness of submission")
            volumes = volumes.flip(-1)

        # align to GT
        if submission_dataset.submission_config[str(idx)]["align"] == 1:
            print("    Aligning submission to ground truth")
            volumes = align_submission(volumes, vol_gt_ref, config)

        # save preprocessed volumes
        print("    Saving preprocessed submission")
        submission_version = submission_dataset.submission_config[str(idx)][
            "submission_version"
        ]
        if str(submission_version) == "0":
            submission_version = ""
        else:
            submission_version = f" {submission_version}"
        print(f" SUBMISSION VERSION {submission_version}")
        submission_id = (
            submission_dataset.submission_config[str(idx)]["flavor_name"]
            + submission_version
        )
        print(f"SUBMISSION ID {submission_id}")

        save_submission(
            volumes,
            submission_dataset[i]["populations"],
            submission_id,
            idx,
            config,
        )
        print(f"   submission saved as submission_{idx}.pt")
        print(f"Preprocessing submission {idx} complete")

    hash_table_path = os.path.join(
        config["output_path"], "submission_to_icecream_table.json"
    )

    update_hash_table(hash_table_path, hash_table)

    return
