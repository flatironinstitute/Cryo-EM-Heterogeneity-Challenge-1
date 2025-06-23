from omegaconf import OmegaConf
from cryo_challenge.commands import run_distribution_to_distribution_pipeline


def test_run_distribution_to_distribution_no_regularization():
    args = OmegaConf.create(
        {
            "config": "tests/config_files/test_config_distribution_to_distribution_no_regularization.yaml"
        }
    )
    run_distribution_to_distribution_pipeline.main(args)


def test_run_distribution_to_distribution_regularization():
    args = OmegaConf.create(
        {
            # "config": "tests/config_files/test_config_distribution_to_distribution_regularization.yaml"
            "config": "/mnt/home/smbp/ceph/smbpchallenge/distribution_to_distribution/config_files/config_distribution_to_distribution_mango_1_l2_corr_bioem_ranknormalizationfalse_nmicrostates40.yaml"
        }
    )
    run_distribution_to_distribution_pipeline.main(args)
