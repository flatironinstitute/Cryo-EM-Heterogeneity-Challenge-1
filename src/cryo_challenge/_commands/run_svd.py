"""
Run SVD on submission volumes and, optionally, reference volumes.
"""

import argparse
import os
import yaml

from .._svd.svd_pipeline import run_all_vs_all_pipeline, run_all_vs_ref_pipeline
from ..data._validation import validate_config_svd


def add_args(parser):
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the config (yaml) file"
    )
    return parser


def mkbasedir(out):
    if not os.path.exists(os.path.dirname(out)):
        try:
            os.makedirs(config["output_options"]["output_path"])
        except (FileExistsError, PermissionError):
            raise ValueError("Output path does not exist and cannot be created.")
    return


def warnexists(out):
    if os.path.exists(out):
        Warning("Warning: {} already exists. Overwriting.".format(out))


def main(config):
    mkbasedir(config["output_options"]["output_path"])
    warnexists(config["output_options"]["output_path"])

    if config["experiment_mode"] == "all_vs_all":
        run_all_vs_all_pipeline(config)

    elif config["experiment_mode"] == "all_vs_ref":
        run_all_vs_ref_pipeline(config)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    args = add_args(parser).parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    validate_config_svd(config)
    main(config)
