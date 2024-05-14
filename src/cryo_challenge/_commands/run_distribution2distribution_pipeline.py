"""
Quantify the similarity between ground truth distribution and user submitted distribution, taking into account the population weights and a map to map distance.
"""

import argparse
import os
import yaml

from .._distribution_to_distribution.distribution_to_distribution import run
from ..data._validation.config_validators import validate_input_config_disttodist


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

    validate_input_config_disttodist(config)
    warnexists(config["output_fname"])
    mkbasedir(os.path.dirname(config["output_fname"]))

    run(config)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    main(add_args(parser).parse_args())
