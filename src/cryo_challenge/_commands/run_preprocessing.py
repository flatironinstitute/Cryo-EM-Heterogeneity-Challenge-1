import argparse
import os
import yaml
import json

from ..data._validation.config_validators import validate_config_preprocessing
from .._preprocessing.preprocessing_pipeline import preprocess_submissions
from .._preprocessing.dataloader import SubmissionPreprocessingDataLoader


def add_args(parser):
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the config (yaml) file"
    )
    return parser


def mkbasedir(out):
    if not os.path.exists(out):
        try:
            os.makedirs(out)
        except (FileExistsError, PermissionError):
            raise ValueError("Output path does not exist and cannot be created.")
    return


def warnexists(out):
    if os.path.exists(out):
        Warning("Warning: {} already exists. Overwriting.".format(out))


def main(args):
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    validate_config_preprocessing(config)
    print(config)

    warnexists(config["output_path"])
    mkbasedir(config["output_path"])

    with open(config["submission_config_file"], "r") as f:
        submission_config = json.load(f)
    submission_dataset = SubmissionPreprocessingDataLoader(submission_config)

    preprocess_submissions(submission_dataset, config)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
