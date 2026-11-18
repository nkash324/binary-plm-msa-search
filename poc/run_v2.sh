#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; rm -f results/cutoff_v2.flag
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
conda activate fold
python 29_cutoff_v2.py 2>&1 | grep -v -i warn | grep -v -i unexpected | tail -45
if grep -q "MSAS DONE" results/cutoff_v2.log; then
  conda activate cf; export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
  python 30_fold_v2.py 2>&1 | grep -v -i warn | tail -45
fi
echo "ALL DONE" > results/cutoff_v2.flag
