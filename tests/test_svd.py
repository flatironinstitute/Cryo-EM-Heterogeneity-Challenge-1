from omegaconf import OmegaConf
from cryo_challenge.commands import run_svd_pipeline


def test_run_svd():
    args = OmegaConf.create({"config": "tests/config_files/test_config_svd.yaml"})
    run_svd_pipeline.main(args)
