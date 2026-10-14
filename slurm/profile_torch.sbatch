#!/bin/bash
#SBATCH --job-name=olmo-prof-torch
#SBATCH --account=bikd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:15:00
#SBATCH --output=results/baseline_profile/slurm-%x-%j.out

# submit from the repo root: TAG=<run_tag> sbatch slurm/profile_torch.sbatch
source ~/classes/598/env.sh
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd $PROJ/598-project
TAG=${TAG:-$(date +%Y%m%d_%H%M%S)_$SLURM_JOB_ID}
OUT=results/baseline_profile/$TAG/torch
RAW=$RUNS/baseline_profile/$TAG/torch
mkdir -p $OUT $RAW

echo "host $(hostname)  job $SLURM_JOB_ID  tag $TAG  $(date)"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,temperature.gpu --format=csv
python -m profiling.run --tool torch --out $OUT --raw $RAW
echo "done $(date)"
