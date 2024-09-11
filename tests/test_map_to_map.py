from omegaconf import OmegaConf
from cryo_challenge._commands import run_map2map_pipeline
import numpy as np


def test_run_map2map_pipeline():
    args = OmegaConf.create(
        {"config": "tests/config_files/test_config_map_to_map.yaml"}
    )
    results_dict = run_map2map_pipeline.main(args)

    args_low_memory = OmegaConf.create(
        {"config": "tests/config_files/test_config_map_to_map_low_memory.yaml"}
    )
    results_dict_low_memory = run_map2map_pipeline.main(args_low_memory)
    for metric in ["fsc", "corr", "l2", "bioem"]:
        if metric == "fsc":
            np.allclose(
                results_dict[metric]["computed_assets"]["fsc_matrix"],
                results_dict_low_memory[metric + "_low_memory"]["computed_assets"][
                    "fsc_matrix"
                ],
            )
        np.allclose(
            results_dict[metric]["cost_matrix"].values,
            results_dict_low_memory[metric + "_low_memory"]["cost_matrix"].values,
        )
