from omegaconf import OmegaConf
from cryo_challenge.commands import run_map2map_pipeline
import numpy as np


def test_run_map2map_pipeline_self():
    args = OmegaConf.create(
        {"config": "tests/config_files/test_config_map_to_map_self.yaml"}
    )
    results_dict = run_map2map_pipeline.main(args)
    for metric in ["fsc", "res", "corr", "l2", "bioem"]:
        assert metric in results_dict.keys()
        n_row_self, n_col_self = results_dict[metric]["cost_matrix_self"].values.shape
        assert n_row_self == n_col_self


def test_run_map2map_pipeline_procrustes():
    args = OmegaConf.create(
        {
            "config": "tests/config_files/test_config_map_to_map_procrustes_wasserstein.yaml"
        }
    )
    _ = run_map2map_pipeline.main(args)
    return


def test_run_map2map_pipeline_gw():
    args = OmegaConf.create(
        {"config": "tests/config_files/test_config_map_to_map_gw.yaml"}
    )
    results_dict = run_map2map_pipeline.main(args)
    assert "gromov_wasserstein" in results_dict.keys()
    return


# def test_run_map2map_pipeline_zernike():
#     try:
#         args = OmegaConf.create(
#             {"config": "tests/config_files/test_config_map_to_map_external.yaml"}
#         )
#         results_dict = run_map2map_pipeline.main(args)
#         assert "zernike3d" in results_dict.keys()
#     except Exception as e:
#         print(e)
#         print(
#             "External test failed. Skipping test. Fails when running in CI if external dependencies are not installed."
#         )
#     return


def test_run_map2map_pipeline_low_memory():
    for config_fname, config_fname_low_memory in zip(
        [
            "tests/config_files/test_config_map_to_map.yaml",
            "tests/config_files/test_config_map_to_map_nomask_nonormalize.yaml",
        ],
        [
            "tests/config_files/test_config_map_to_map_low_memory_subbatch.yaml",
            "tests/config_files/test_config_map_to_map_low_memory_subbatch_nomask_nonormalize.yaml",
        ],
    ):
        args = OmegaConf.create({"config": config_fname})
        results_dict = run_map2map_pipeline.main(args)

        args_low_memory = OmegaConf.create({"config": config_fname_low_memory})
        results_dict_low_memory = run_map2map_pipeline.main(args_low_memory)
        for metric in ["fsc", "corr", "l2", "bioem"]:
            if metric == "fsc":
                np.allclose(
                    results_dict[metric]["computed_assets"]["fsc_matrix"],
                    results_dict_low_memory[metric]["computed_assets"]["fsc_matrix"],
                )
            elif metric == "res":
                np.allclose(
                    results_dict[metric]["computed_assets"]["fraction_nyquist"],
                    results_dict_low_memory[metric]["computed_assets"][
                        "fraction_nyquist"
                    ],
                )
            np.allclose(
                results_dict[metric]["cost_matrix"].values,
                results_dict_low_memory[metric]["cost_matrix"].values,
            )
    return


if __name__ == "__main__":
    test_run_map2map_pipeline_procrustes()
