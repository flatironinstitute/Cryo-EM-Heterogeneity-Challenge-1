mkdir -p data/dataset_2_submissions tests/results tests/data/unprocessed_dataset_2_submissions/submission_x
wget https://files.osf.io/v1/resources/8h6fz/providers/dropbox/tests/dataset_2_submissions/test_submission_0_n8.pt?download=true -O tests/data/dataset_2_submissions/test_submission_0_n8.pt
wget https://files.osf.io/v1/resources/8h6fz/providers/dropbox/tests/Ground_truth/test_maps_gt_flat_10.pt?download=true -O tests/data/Ground_truth/test_maps_gt_flat_10.pt
wget https://files.osf.io/v1/resources/8h6fz/providers/dropbox/tests/Ground_truth/test_metadata_10.csv?download=true -O tests/data/Ground_truth/test_metadata_10.csv 
wget https://files.osf.io/v1/resources/8h6fz/providers/dropbox/tests/Ground_truth/1.mrc?download=true -O tests/data/Ground_truth/1.mrc 
wget https://files.osf.io/v1/resources/8h6fz/providers/dropbox/tests/Ground_truth/mask_dilated_wide_224x224.mrc?download=true -O tests/data/Ground_truth/mask_dilated_wide_224x224.mrc 
for FILE in 1.mrc 2.mrc 3.mrc 4.mrc populations.txt
do
    wget https://files.osf.io/v1/resources/8h6fz/providers/dropbox/tests/unprocessed_dataset_2_submissions/submission_x/${FILE}?download=true -O tests/data/unprocessed_dataset_2_submissions/submission_x/${FILE}
done
