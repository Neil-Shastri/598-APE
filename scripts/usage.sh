#!/bin/bash
# GPU-hours charged to bikd, from sacct (elapsed * billing / 1000)
sacct -u $USER -A bikd-delta-gpu -S 2026-09-01 -X -P --format=JobID,Partition,Start,Elapsed,ElapsedRaw,AllocTRES,State |
awk -F"|" 'NR==1 {print "JobID|Partition|Start|Elapsed|State"; next}
  { b=0; if (match($6, /billing=[0-9]+/)) b=substr($6, RSTART+8, RLENGTH-8); s+=$5*b/1000; print $1"|"$2"|"$3"|"$4"|"$7 }
  END { printf "TOTAL charged: %.4f hours (%.0f s); remaining of 10: %.4f hours\n", s/3600, s, 10-s/3600 }' | column -t -s"|"
