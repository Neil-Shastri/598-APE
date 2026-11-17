#!/bin/bash
#SBATCH --job-name=olmo-prof-ncu
#SBATCH --account=bikd-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:25:00
#SBATCH --output=results/baseline_profile/slurm-%x-%j.out

# submit from the repo root: TAG=<run_tag> sbatch slurm/profile_ncu.sbatch
source ~/classes/598/env.sh
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd $PROJ/598-APE
TAG=${TAG:-$(date +%Y%m%d_%H%M%S)_$SLURM_JOB_ID}
OUT=results/baseline_profile/$TAG/ncu
RAW=$RUNS/baseline_profile/$TAG/ncu
mkdir -p $OUT $RAW

echo "host $(hostname)  job $SLURM_JOB_ID  tag $TAG  $(date)"
ncu --version | tail -1
# only kernels between cudaProfilerStart/Stop are profiled: decoder layer 3 of one
# prefill (batch 1, seq 512) and of the 5th decode step (batch 1)
ncu --profile-from-start off --nvtx \
    --section SpeedOfLight \
    --section SpeedOfLight_RooflineChart \
    --section SpeedOfLight_HierarchicalSingleRooflineChart \
    --section SpeedOfLight_HierarchicalHalfRooflineChart \
    --section SpeedOfLight_HierarchicalTensorRooflineChart \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Chart \
    --section MemoryWorkloadAnalysis_Tables \
    --section ComputeWorkloadAnalysis \
    --section Occupancy \
    --section LaunchStats \
    --section WarpStateStats \
    --section SchedulerStats \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,lts__t_sector_hit_rate.pct \
    --export $RAW/olmo --force-overwrite \
    python -m profiling.run --tool ncu --batches 1 --warmup 2 --ncu-layer 3 --ncu-step 4

ncu --import $RAW/olmo.ncu-rep --csv --page raw --print-units base > $RAW/ncu_raw.csv
ncu --import $RAW/olmo.ncu-rep --csv --page raw --print-units base --print-nvtx-rename kernel > $RAW/ncu_raw_nvtx.csv
ncu --import $RAW/olmo.ncu-rep --page details > $RAW/ncu_details.txt
python -m profiling.ncu_table $RAW/ncu_raw.csv $RAW/ncu_raw_nvtx.csv --out $OUT
echo "done $(date)"
