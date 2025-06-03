import torch
import json
import os
import mrcfile
import numpy as np

from ._global_alignment import align_submission_to_reference
from ._cropping_and_padding import crop_or_pad_submission
from ._downsampling import downsample_submission
from ._submission_dataset import SubmissionPreprocessingDataset
from ..config_validation._preprocessing_validators import PreprocessingRunConfig


def run_preprocessing_pipeline(
    submission_dataset: SubmissionPreprocessingDataset,
    run_config: PreprocessingRunConfig,
) -> None:
    metadata_submission_preproc = {}
    box_size_gt = submission_dataset.config_for_ref_vol.box_size
    voxel_size_gt = submission_dataset.config_for_ref_vol.voxel_size

    with mrcfile.open(
        submission_dataset.config_for_ref_vol.path_to_reference_volume, mode="r"
    ) as file:
        reference_volume = file.data.copy()

    metadata_preproc_path = os.path.join(
        run_config.output_path, "submission_to_icecream_table.json"
    )

    for i in range(len(submission_dataset)):
        dataset_item = submission_dataset[i]
        sub_config = dataset_item["config"]

        sub_fname = _flavor_and_version_to_fname(
            sub_config.ice_cream_flavor,
            sub_config.submission_version,
        )
        submission_id = f"{sub_config.ice_cream_flavor} {sub_config.submission_version}"

        metadata_submission_preproc[submission_id] = {
            "submission name": sub_config.submission_name,
            "submission_version": sub_config.submission_version,
            "filename": sub_fname,
        }

        print(f"Preprocessing submission {submission_id}...")

        voxel_size_sub = sub_config.voxel_size
        volumes = dataset_item["volumes"]

        print("    Cropping and padding submission")
        # pad/crop (prep for downsampling)
        volumes = crop_or_pad_submission(
            volumes, box_size_gt, voxel_size_sub, voxel_size_gt
        )

        # downsampling making sure pixel sizes and box sizes match
        print("    Downsampling submission")
        volumes = downsample_submission(volumes, box_size_gt)

        # flip handedness
        if sub_config.flip_handedness:
            print("    Flipping handedness of submission")
            volumes = volumes.flip(-1)

        # align to reference
        if sub_config.align:
            print("    Aligning submission to reference")
            volumes, rotation_mtx = align_submission_to_reference(
                volumes,
                reference_volume,
                sub_config.initial_rotation_guess,
                sub_config.threshold_percentile,
                loss_type=run_config.BOTAlign_params["loss_type"],
                loss_params=run_config.BOTAlign_params["loss_params"],
                downsampled_size=run_config.BOTAlign_params["downsampled_size"],
                refinement_downsampled_size=run_config.BOTAlign_params[
                    "refinement_downsampled_size"
                ],
                max_iters=run_config.BOTAlign_params["max_iters"],
                refine=run_config.BOTAlign_params["refine"],
                tau=run_config.BOTAlign_params["tau"],
                surrogate_max_iter=run_config.BOTAlign_params["surrogate_max_iter"],
                surrogate_min_grad=run_config.BOTAlign_params["surrogate_min_grad"],
                surrogate_min_step=run_config.BOTAlign_params["surrogate_min_step"],
                verbosity=run_config.BOTAlign_params["verbosity"],
                dtype=run_config.BOTAlign_params["dtype"],
            )
        else:
            rotation_mtx = np.eye(3)

        # save preprocessed volumes
        print("    Saving preprocessed submission")

        _save_submission(
            volumes,
            submission_dataset[i]["populations"],
            submission_id,
            sub_fname,
            run_config.output_path,
            rotation_mtx,
        )
        print(f"   submission saved as {sub_fname}")
        print(f"Preprocessing submission {submission_id} complete")

        _update_metadata_file(metadata_preproc_path, metadata_submission_preproc)

    return


def _flavor_and_version_to_fname(flavor, version):
    flavor = flavor.replace(" ", "_").lower()
    version = version.replace(" ", "_")
    version = version.replace(".", "_").lower()
    return f"submission_{flavor}_{version}.pt"


def _save_submission(
    volumes, populations, submission_id, submission_fname, output_path, rotation_mtx
):
    """
    Save preprocessed submission volumes

    **Arguments**

    volumes (torch.Tensor): submission volumes
        shape: (n_volumes, im_x, im_y, im_z)
    populations (list): populations
    submission_id (int): submission id
    submission_fname (str): filename used to save the submission
    output_path (str): path to save the submission
    rotation_mtx (np.ndarray): rotation matrix used to align the submission

    **Returns**

    None
    """

    submission_dict = {
        "volumes": volumes,
        "populations": populations,
        "id": submission_id,
        "rotation_matrix": rotation_mtx,
    }

    submission_path = os.path.join(output_path, submission_fname)
    torch.save(submission_dict, submission_path)

    return


def _update_metadata_file(metadata_preproc_path, metadata_submission_preproc):
    if os.path.exists(metadata_preproc_path):
        with open(metadata_preproc_path, "r") as f:
            old_metadata = json.load(f)
        old_metadata.update(metadata_submission_preproc)

        with open(metadata_preproc_path, "w") as f:
            json.dump(old_metadata, f, indent=4)

    else:
        with open(metadata_preproc_path, "w") as f:
            json.dump(metadata_submission_preproc, f, indent=4)

    return
