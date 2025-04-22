#!/bin/bash
#SBATCH --job-name=torchmp_benchmark
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err
#SBATCH --partition=ccb
#SBATCH -n 1
#SBATCH -c 128
#SBATCH --time=99:00:00

# Define parameter ranges
n_pix_values=(244)
nn_values=(1)
num_workers_values=(16 32 64 128)

# Loop through all combinations of k, nn, and num_workers
for n_pix in "${n_pix_values[@]}"; do
  for nn in "${nn_values[@]}"; do
    for num_workers in "${num_workers_values[@]}"; do
      echo "Running with n_pix=$n_pix, nn=$nn, num_workers=$num_workers"
      python torch_mp.py +n_pix=$n_pix +nn=$nn +num_workers=$num_workers > torchmp_benchmark_npix_${n_pix}_nn_${nn}_num_workers_${num_workers}.log
    done
  done
done
