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

# submit from the repo root: TAG=<run_tag> [OPT=fused,static,graph] sbatch slurm/profile_torch.sbatch
source ~/classes/598/env.sh
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd $PROJ/598-APE
TAG=${TAG:-$(date +%Y%m%d_%H%M%S)_$SLURM_JOB_ID}
DIR=baseline_profile
[ -n "$OPT" ] && DIR=optimized_profile
OUT=results/$DIR/$TAG/torch
RAW=$RUNS/$DIR/$TAG/torch
mkdir -p $OUT $RAW

echo "host $(hostname)  job $SLURM_JOB_ID  tag $TAG  $(date)"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,temperature.gpu --format=csv
python -m profiling.run --tool torch --opt "$OPT" --out $OUT --raw $RAW
echo "done $(date)"
