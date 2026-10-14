#!/bin/bash
source ~/classes/598/env.sh
cd $PROJ/gpucheck
OUT=$PROJ/gpucheck/ncu_$(date +%Y%m%d_%H%M%S).txt
{
  echo "== host: $(hostname)"; which ncu; ncu --version | tail -1
  ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
      --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum ./vecadd
} 2>&1 | tee $OUT
echo "saved to $OUT"
