#!/bin/bash
source ~/classes/598/env.sh
cd $PROJ/gpucheck
OUT=$PROJ/gpucheck/results_$(date +%Y%m%d_%H%M%S).txt
{
  echo "== host: $(hostname)  date: $(date)"
  echo "== nvidia-smi"; nvidia-smi
  echo "== gpu query"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit,ecc.mode.current,pcie.link.gen.max,pcie.link.width.max --format=csv
  nvidia-smi -q > nvidia-smi-q.txt
  echo "== nvcc"; nvcc --version | tail -2
  echo "== tools"; which ncu nsys compute-sanitizer; ncu --version | tail -1; nsys --version
  echo "== vecadd"; ./vecadd
  echo "== compute-sanitizer"; compute-sanitizer ./vecadd
  echo "== ncu"
  ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
      --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum ./vecadd
  echo "== nsys"
  nsys profile --stats=true --force-overwrite=true -o $RUNS/vecadd_nsys ./vecadd
  echo "== torch"
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.version.cuda)"
  echo "== done: $(date)"
} 2>&1 | tee $OUT
echo "saved to $OUT"
