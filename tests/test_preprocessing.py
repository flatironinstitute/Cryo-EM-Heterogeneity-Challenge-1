from omegaconf import OmegaConf
from cryo_challenge.commands import run_preprocessing


def test_run_preprocessing():
    args = OmegaConf.create({"config": "tests/config_files/test_config_preproc.yaml"})
    run_preprocessing.main(args)
