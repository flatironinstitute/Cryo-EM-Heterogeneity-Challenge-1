from omegaconf import OmegaConf
from cryo_challenge._commands import run_svd


def test_run_preprocessing():    
    args = OmegaConf.create({'config': 'tests/config_files/test_config_svd.yaml'})
    run_svd.main(args)