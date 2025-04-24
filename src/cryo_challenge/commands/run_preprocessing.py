import argparse
import os
import yaml

from ..config_validation._preprocessing_validators import (
    PreprocessingDatasetReferenceConfig,
    PreprocessingRunConfig,
    PreprocessingDatasetSubmissionConfig,
)
from ..preprocessing._preprocessing_pipeline import run_preprocessing_pipeline
from ..preprocessing._submission_dataset import SubmissionPreprocessingDataset


def _add_args(parser):
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the config (yaml) file"
    )
    return parser


def _mkbasedir(out):
    if not os.path.exists(out):
        try:
            os.makedirs(out)
        except (FileExistsError, PermissionError):
            raise ValueError("Output path does not exist and cannot be created.")
    return


def _warnexists(out):
    if os.path.exists(out):
        Warning("Warning: {} already exists. Overwriting.".format(out))


def _load_submission_configs(path_to_submissions_config):
    with open(path_to_submissions_config, "r") as f:
        submission_config_list = [
            PreprocessingDatasetSubmissionConfig(**sub_config)
            for sub_config in yaml.safe_load(f)
        ]
    return submission_config_list


def _load_reference_config(path_to_reference_config):
    with open(path_to_reference_config, "r") as f:
        reference_config_dict = yaml.safe_load(f)

    reference_config = PreprocessingDatasetReferenceConfig(**reference_config_dict)
    return reference_config


def main(args):
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    config = PreprocessingRunConfig(**config)
    print(config)

    _warnexists(config.output_path)
    _mkbasedir(config.output_path)

    reference_config = _load_reference_config(config.path_to_reference_config)
    submission_configs = _load_submission_configs(config.path_to_submissions_config)

    dataset = SubmissionPreprocessingDataset(
        configs_for_vol_sets=submission_configs,
        config_for_ref_vol=reference_config,
    )

    run_preprocessing_pipeline(dataset, config)

    return 0


def main_as_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    return main(_add_args(parser).parse_args())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(_add_args(parser).parse_args())
