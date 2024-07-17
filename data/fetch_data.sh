mkdir -p data/dataset_2_submissions data/dataset_1_submissions data/dataset_2_ground_truth

# dataset 1 submissions
for i in {0..10}
do
    wget https://files.osf.io/v1/resources/8h6fz/providers/dropbox/dataset_1_submissions/submission_${i}.pt?download=true -O data/dataset_1_submissions/submission_${i}.pt
done

# dataset 2 submissions
for i in {0..11}
do
    wget https://files.osf.io/v1/resources/8h6fz/providers/dropbox/dataset_2_submissions/submission_${i}.pt?download=true -O data/dataset_2_submissions/submission_${i}.pt
done

# ground truth

wget https://files.osf.io/v1/resources/8h6fz/providers/dropbox/Ground_truth/maps_gt_flat.pt?download=true -O data/dataset_2_ground_truth/maps_gt_flat.pt

wget https://files.osf.io/v1/resources/8h6fz/providers/dropbox/Ground_truth/metadata.csv?download=true -O data/dataset_2_ground_truth/metadata.csv

wget https://files.osf.io/v1/resources/8h6fz/providers/dropbox/Ground_truth/mask_dilated_wide_224x224.mrc?download=true -O data/dataset_2_ground_truth/mask_dilated_wide_224x224.mrc
