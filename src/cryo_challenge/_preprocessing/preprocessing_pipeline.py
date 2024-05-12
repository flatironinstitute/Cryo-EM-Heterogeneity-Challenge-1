import torch
import numpy as np
import json

from .align_utils import align_submission, center_submission, threshold_submissions
from .crop_pad_utils import crop_pad_submission
from .fourier_utils import downsample_submission


def save_submission(volumes, populations, submission_id, submission_index):
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

    torch.save(submission_dict, f"submission_{submission_index}.pt")

    return submission_dict


def preprocess_submission_parse_kwargs(kwargs):
    req_kwargs = {
        "thresh_percentile": 93.0,
        "BOT_box_size": 32,
        "BOT_loss": "wemd",
        "BOT_iter": 200,
        "BOT_reflect": False,
        "BOT_refine": True,
    }

    for key, value in req_kwargs.items():
        if key not in kwargs.keys():
            kwargs[key] = value

    for key, value in kwargs.items():
        if key not in req_kwargs.keys():
            raise ValueError(f"Invalid keyword argument {key}")

        if key == "thresh_percentile":
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"thresh_percentile should be a number, got {type(value)}"
                )

            if value < 0 or value > 100:
                raise ValueError("thresh_percentile should be between 0 and 100")

        if key == "BOT_box_size":
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"BOT_box_size should be an integer, got {type(value)}"
                )

            if value < 0:
                raise ValueError("BOT_box_size should be a positive integer")

        if key == "BOT_loss":
            if not isinstance(value, str):
                raise ValueError(f"BOT_loss should be a string, got {type(value)}")

            if value not in ["wemd", "eu"]:
                raise ValueError("BOT_loss should be 'wemd' or 'eu'")

        if key == "BOT_iter":
            if not isinstance(value, (int, float)):
                raise ValueError(f"BOT_iter should be an integer, got {type(value)}")

            if value < 0:
                raise ValueError("BOT_iter should be a positive integer")

        if key == "BOT_reflect":
            if not isinstance(value, bool):
                raise ValueError(f"BOT_reflect should be a boolean, got {type(value)}")

        if key == "BOT_refine":
            if not isinstance(value, bool):
                raise ValueError(f"BOT_refine should be a boolean, got {type(value)}")

    return kwargs


def preprocess_submissions(submission_dataset, seed, **kwargs):
    params = preprocess_submission_parse_kwargs(kwargs)

    np.random.seed(seed)
    ice_cream_flavors = [
        "Chocolate",
        "Vanilla",
        "Cookies N' Cream",
        "Mint Chocolate Chip",
        "Strawberry",
        "Butter Pecan",
        "Salted Caramel",
        "Pistachio",
        "Rocky Road",
        "Coffee",
        "Cookie Dough",
        "Chocolate Chip",
        "Neapolitan",
        "Cherry",
        "Rainbow Sherbet",
        "Peanut Butter",
        "Cotton Candy",
        "Lemon Sorbet",
        "Mango",
        "Black Raspberry",
    ]

    n_subs = max(submission_dataset.subs_index) + 1
    random_mapping = np.random.choice(len(ice_cream_flavors), n_subs, replace=False)
    hash_table = {}

    box_size_gt = submission_dataset.submission_config["gt"]["box_size"]
    pixel_size_gt = submission_dataset.submission_config["gt"]["pixel_size"]
    vol_gt_ref = submission_dataset.vol_gt_ref

    for i in range(len(submission_dataset)):
        idx = submission_dataset.subs_index[i]

        hash_table[submission_dataset.submission_config[str(idx)]["name"]] = (
            ice_cream_flavors[random_mapping[idx]]
        )

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
                "Submission is not suited for cropping-padding, pixel size less than ground truth pixel size"
            )
            continue

        # downsampling making sure pixel sizes and box sizes match
        print("    Downsampling submission")
        volumes = downsample_submission(volumes, box_size_gt)

        # thresholding
        print("    Thresholding submission")
        volumes = threshold_submissions(volumes, params["thresh_percentile"])

        # center submission
        print("    Centering submission")
        volumes = center_submission(volumes)

        # align to GT
        if submission_dataset.submission_config[str(idx)]["align"] == 1:
            print("    Aligning submission to ground truth")
            volumes = align_submission(volumes, vol_gt_ref, params)

        # save preprocessed volumes
        print("    Saving preprocessed submission")
        save_submission(
            volumes,
            submission_dataset[i]["populations"],
            ice_cream_flavors[random_mapping[idx]],
            idx,
        )
        print(f"   submission saved as submission_{idx}.pt")
        print(f"Preprocessing submission {idx} complete")

    with open("hash_table.json", "w") as f:
        json.dump(hash_table, f, indent=4)

    return
