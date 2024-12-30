#!/bin/bash
#SBATCH --job-name=exponent
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err
#SBATCH --partition=ccb
#SBATCH -n 40
#SBATCH --time=99:00:00

for N_DOWNSAMPLE_PIX in 20
do
    for TOP_K in 250 500 750
    do
        for EXPONENT in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.5 2.0
        do
            srun python /mnt/home/gwoollard/ceph/repos/Cryo-EM-Heterogeneity-Challenge-1/src/cryo_challenge/_map_to_map/gromov_wasserstein/gw_weighted_voxels.py --n_downsample_pix  ${N_DOWNSAMPLE_PIX} --top_k ${TOP_K} --exponent $EXPONENT --slurm
        done
    done
done
