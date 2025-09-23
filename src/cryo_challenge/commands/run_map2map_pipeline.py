"""
Compute map to map distances on ground truth versus submission volumes.
"""

import argparse
import os
import yaml

from ..map_to_map.map_to_map_pipeline import run
from ..config_validation._map_to_map_validation import MapToMapInputConfig


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


def run_map2map_from_config(config: MapToMapInputConfig):
    config_as_dict = dict(config.model_dump(exclude_none=True))
    warnexists(config_as_dict["path_to_output_file"])
    mkbasedir(os.path.dirname(config_as_dict["path_to_output_file"]))
    return run(config_as_dict)


def main(args):
    with open(args.config, "r") as file:
        config_file = yaml.safe_load(file)

    config = MapToMapInputConfig(**config_file)
    run_output = run_map2map_from_config(config)
    return run_output


def main_as_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
