from omegaconf import OmegaConf
from cryo_challenge._commands import run_map2map_pipeline

args = OmegaConf.create({'config': 'config_files/test_config_map_to_map_distance_matrix.yaml'})

def test_run_map2map_pipeline(args):    
    run_map2map_pipeline.main(args)