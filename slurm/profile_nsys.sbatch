#!/bin/bash
#SBATCH --job-name=olmo-prof-nsys
#SBATCH --account=bikd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:15:00
#SBATCH --output=results/baseline_profile/slurm-%x-%j.out

# submit from the repo root: TAG=<run_tag> sbatch slurm/profile_nsys.sbatch
source ~/classes/598/env.sh
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd $PROJ/598-project
TAG=${TAG:-$(date +%Y%m%d_%H%M%S)_$SLURM_JOB_ID}
OUT=results/baseline_profile/$TAG/nsys
RAW=$RUNS/baseline_profile/$TAG/nsys
mkdir -p $OUT $RAW

echo "host $(hostname)  job $SLURM_JOB_ID  tag $TAG  $(date)"
nsys --version
# recording starts/stops at cudaProfilerStart/Stop in profiling/run.py, after model load and warmup
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output=$RAW/olmo --force-overwrite=true \
    python -m profiling.run --tool nsys

nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,nvtx_kern_sum,cuda_kern_exec_sum \
    --format csv --timeunit usec --force-overwrite true --output $OUT/nsys $RAW/olmo.nsys-rep
# per-kernel traces are large, keep them with the raw data
nsys stats --report cuda_gpu_trace,nvtx_gpu_proj_trace \
    --format csv --timeunit nsec --force-overwrite true --output $RAW/nsys $RAW/olmo.nsys-rep
python -m profiling.nsys_gaps $RAW/nsys_cuda_gpu_trace.csv $RAW/nsys_nvtx_gpu_proj_trace.csv --out $OUT
echo "done $(date)"
