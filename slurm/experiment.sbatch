#!/bin/bash
#SBATCH --job-name=olmo-exp
#SBATCH --account=bikd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:15:00
#SBATCH --output=results/experiments/slurm-%x-%j.out

# submit from the repo root:
#   TAG=e1 VARIANTS="baseline fused" sbatch slurm/experiment.sbatch
source ~/classes/598/env.sh
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=/work/nvme/bikd/$USER/598/inductor_cache
cd $PROJ/598-project
TAG=${TAG:-exp}_$SLURM_JOB_ID
OUT=results/experiments/$TAG
mkdir -p $OUT

echo "host $(hostname)  job $SLURM_JOB_ID  tag $TAG  variants: $VARIANTS  $(date)"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,temperature.gpu --format=csv
if [ -n "$UNIT_TESTS" ]; then
    python -m pytest -q $UNIT_TESTS || exit 1
fi
if [ -n "$KERNEL_BENCH" ]; then
    python -m olmo_bench.kernels --bench
fi
for v in $VARIANTS; do
    echo "=== variant $v  $(date)"
    opt=$v
    [ "$v" = baseline ] && opt=""
    python -m olmo_bench.benchmark --opt "$opt" --out-root $OUT --run-tag $v
done
echo "done $(date)"
