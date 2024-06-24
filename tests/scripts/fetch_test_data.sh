mkdir -p data/dataset_2_submissions tests/data/dataset_2_submissions tests/results
wget https://files.osf.io/v1/resources/8h6fz/providers/dropbox/dataset_2_submissions/test_submission_0_n8.pt?download=true -O tests/data/dataset_2_submissions/test_submission_0_n8.pt
wget https://files.osf.io/v1/resources/8h6fz/providers/dropbox/Ground_truth/test_maps_gt_flat_10.pt?download=true -O tests/data/test_maps_gt_flat_10.pt
wget https://files.osf.io/v1/resources/8h6fz/providers/dropbox/Ground_truth/test_metadata_10.csv?download=true -O tests/data/test_metadata_10.csv 
wget https://files.osf.io/v1/resources/8h6fz/providers/dropbox/Ground_truth/mask_dilated_wide_224x224.mrc?download=true -O data/mask_dilated_wide_224x224.mrc 