#!/bin/bash
#SBATCH --job-name=olmo-baseline
#SBATCH --account=bikd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:15:00
#SBATCH --output=results/baseline/slurm-%j.out

# submit from the repo root: sbatch slurm/baseline.sbatch
source ~/classes/598/env.sh
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd $PROJ/598-project

echo "host $(hostname)  job $SLURM_JOB_ID  $(date)"
nvidia-smi
# correctness runs first inside the same process, so the model loads once
python -m olmo_bench.benchmark --device cuda --dtype bfloat16
echo "done $(date)"
