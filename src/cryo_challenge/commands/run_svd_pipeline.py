"""
Run SVD on submission volumes and, optionally, reference volumes.
"""

import argparse
import os
import yaml
import logging
import datetime
import torch

from cryo_challenge.svd import (
    run_svd_noref,
    run_svd_with_ref,
    compute_common_power_spectrum_on_grid,
    load_submissions,
)
from cryo_challenge.config_validation import SVDInputConfig


def add_args(parser):
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the config (yaml) file"
    )
    parser.add_argument(
        "--precompute_power_spectrum",
        action="store_true",
        help="Whether to precompute the power spectrum used for normalization."
        " Only this step of the pipeline will be run.",
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


def run_svd_from_config(config: SVDInputConfig):
    warnexists(config.output_params["path_to_output_dir"])
    mkbasedir(config.output_params["path_to_output_dir"])

    # Directory for saving intermediate results
    tmp_dir = os.path.join(
        config.output_params["path_to_output_dir"], "prepared_submissions_for_svd/"
    )
    mkbasedir(tmp_dir)

    # set up logger
    logger = logging.getLogger()
    logger.handlers.clear()

    logger_fname = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    logger_fname = os.path.join(
        config.output_params["path_to_output_dir"], logger_fname + ".log"
    )
    fhandler = logging.FileHandler(filename=logger_fname, mode="a")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)

    config_as_dict = dict(config.model_dump())
    with open(
        os.path.join(config.output_params["path_to_output_dir"], "config.yaml"), "w"
    ) as file:
        yaml.dump(config_as_dict, file)

    if config.gt_params is None:
        logging.info("Running SVD without reference volumes")
        output = run_svd_noref(config)

    else:
        logging.info("Running SVD with reference volumes")
        output = run_svd_with_ref(config)

    return output


def main(args):
    with open(args.config, "r") as file:
        config_dict = yaml.safe_load(file)

    config = SVDInputConfig(**config_dict)
    if args.precompute_power_spectrum:
        logging.info("Precomputing power spectrum for normalization")
        common_power_spectrum_on_grid = compute_common_power_spectrum_on_grid(
            dataset_for_svd=load_submissions(config)
        )
        logging.info("... done")
        torch.save(
            common_power_spectrum_on_grid,
            os.path.join(
                config.output_params["path_to_output_dir"], "common_power_spectrum.pt"
            ),
        )
        logging.info(
            "Power spectrum volume was saved at "
            f"{os.path.join(config.output_params['path_to_output_dir'], 'common_power_spectrum.pt')}"
        )
        logging.info(
            "Power spectrum precomputation is done. Exiting as `precompute_power_spectrum` flag is set."
        )

        return
    return run_svd_from_config(config)


def main_as_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    return main(add_args(parser).parse_args())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
