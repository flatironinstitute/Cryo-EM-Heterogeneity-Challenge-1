"""
Quantify the similarity between ground truth distribution and user submitted distribution, taking into account the population weights and a map to map distance.
"""

import argparse
import os
import yaml

from ..distribution_to_distribution.distribution_to_distribution import run
from ..config_validation._distribution_to_distribution_validation import (
    DistToDistInputConfig,
)


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

    config = dict(DistToDistInputConfig(**config).model_dump())
    warnexists(config["path_to_output_file"])
    mkbasedir(os.path.dirname(config["path_to_output_file"]))

    return run(config)


def main_as_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    return main(add_args(parser).parse_args())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _ = main(add_args(parser).parse_args())
    print("Done", 0)
