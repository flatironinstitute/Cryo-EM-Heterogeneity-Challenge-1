#!/bin/bash
#SBATCH --job-name=daskmockup
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err
#SBATCH --partition=ccb
#SBATCH -n 3
#SBATCH --time=1:00:00



export SLURM_NTASKS=4
cd /mnt/home/gwoollard/ceph/repos/Cryo-EM-Heterogeneity-Challenge-1/src/cryo_challenge/_map_to_map/gromov_wasserstein/
srun python slurm_runner.py
