from omegaconf import OmegaConf
from cryo_challenge._commands import run_map2map_pipeline


def test_run_map2map_pipeline():
    args = OmegaConf.create(
        {"config": "tests/config_files/test_config_map_to_map.yaml"}
    )
    run_map2map_pipeline.main(args)
