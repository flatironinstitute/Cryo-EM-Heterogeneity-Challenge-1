from omegaconf import OmegaConf
from cryo_challenge._commands import run_distribution2distribution_pipeline

args = OmegaConf.create({'config': 'tests/config_files/test_config_map_to_map_distance_matrix.yaml'})

def test_run_distribution2distribution_pipeline(args):    
    run_distribution2distribution_pipeline.main(args)