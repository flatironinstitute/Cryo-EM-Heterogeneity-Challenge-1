"""
Compute map to map distances on ground truth versus submission volumes.
"""

import argparse
import os
import yaml

from .._map_to_map.map_to_map_pipeline import run
from ..data._validation.config_validators import validate_input_config_mtm


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

    validate_input_config_mtm(config)
    warnexists(config["output"])
    mkbasedir(os.path.dirname(config["output"]))

    return run(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # args = parser.parse_args()
    main(add_args(parser).parse_args())
