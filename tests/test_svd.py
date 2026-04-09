from omegaconf import OmegaConf
from cryo_challenge.commands import run_svd_pipeline


def test_run_svd():
    args = OmegaConf.create(
        {
            "config": "tests/config_files/test_config_svd.yaml",
            "precompute_power_spectrum": False,
        }
    )
    run_svd_pipeline.main(args)


def test_precompute_ps():
    args = OmegaConf.create(
        {
            "config": "tests/config_files/test_config_svd.yaml",
            "precompute_power_spectrum": True,
        }
    )
    run_svd_pipeline.main(args)
