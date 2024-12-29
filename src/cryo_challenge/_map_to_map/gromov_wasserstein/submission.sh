#!/bin/bash
#SBATCH --job-name=row_wise
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err
#SBATCH --partition=ccb
#SBATCH -n 40
#SBATCH --time=99:00:00

for N_DOWNSAMPLE_PIX in 20
do
    for TOP_K in 500
    do
        for EXPONENT in 1.0
        do
            srun python /mnt/home/gwoollard/ceph/repos/Cryo-EM-Heterogeneity-Challenge-1/src/cryo_challenge/_map_to_map/gromov_wasserstein/gw_weighted_voxels.py --n_downsample_pix  ${N_DOWNSAMPLE_PIX} --top_k ${TOP_K} --exponent $EXPONENT
        done
    done
done
