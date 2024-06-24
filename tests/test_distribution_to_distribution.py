from omegaconf import OmegaConf
from cryo_challenge._commands import run_distribution2distribution_pipeline


def test_run_distribution2distribution_pipeline():   
    args = OmegaConf.create({'config': 'tests/config_files/test_config_distribution_to_distribution.yaml'})
    run_distribution2distribution_pipeline.main(args)