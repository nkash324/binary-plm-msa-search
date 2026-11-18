#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; rm -f results/full_quick.flag
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
conda activate fold
python 32_fulllength_quick.py 2>&1 | grep -v -i -E "warn|unexpected|Loading weights|LOAD REPORT|lm_head|Notes|can be ignored|^$|---|Key " | tail -40
if grep -q "MSAS DONE" results/full_quick.log; then
  conda activate cf; export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
  python 33_fulllength_fold.py 2>&1 | grep -v -i warn | tail -25
fi
echo "ALL DONE" > results/full_quick.flag
