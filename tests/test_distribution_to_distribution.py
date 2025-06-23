from omegaconf import OmegaConf
from cryo_challenge.commands import run_distribution_to_distribution_pipeline


def test_run_distribution_to_distribution_no_regularization():
    args = OmegaConf.create(
        {
            "config": "tests/config_files/test_config_distribution_to_distribution_no_regularization.yaml"
        }
    )
    run_distribution_to_distribution_pipeline.main(args)


def test_run_distribution_to_distribution_regularization():
    args = OmegaConf.create(
        {
            "config": "tests/config_files/test_config_distribution_to_distribution_regularization.yaml"
        }
    )
    run_distribution_to_distribution_pipeline.main(args)
