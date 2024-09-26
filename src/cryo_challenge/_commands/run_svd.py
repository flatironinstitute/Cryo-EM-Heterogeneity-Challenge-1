"""
Run SVD on submission volumes and, optionally, reference volumes.
"""

import argparse
import os
import yaml

from .._svd.svd_pipeline import run_svd_noref, run_svd_with_ref
from ..data._validation.config_validators import SVDConfig


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

    config = SVDConfig(**config).model_dump()

    warnexists(config["output_params"]["output_file"])
    mkbasedir(os.path.dirname(config["output_params"]["output_file"]))

    output_path = os.path.dirname(config["output_params"]["output_file"])

    with open(os.path.join(output_path, "config.yaml"), "w") as file:
        yaml.dump(config, file)

    if config["gt_params"] is None:
        run_svd_noref(config)

    else:
        run_svd_with_ref(config)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
