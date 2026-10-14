module load python/3.13.5-gcc13.3.1 cudatoolkit/26.5_13.2
export PROJ=~/classes/598
export HF_HOME=/work/nvme/bikd/$USER/598/hf
export PIP_CACHE_DIR=/work/hdd/bikd/$USER/598/pip-cache
export RUNS=/work/hdd/bikd/$USER/598/runs
export PATH=/opt/nvidia/nsight-compute/2026.1.1:$PATH
if [ -f $PROJ/venv/bin/activate ]; then source $PROJ/venv/bin/activate; fi
