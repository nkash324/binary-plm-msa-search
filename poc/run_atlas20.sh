#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; rm -f results/atlas20.flag
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
conda activate fold
python 25_atlas_build.py 2>&1 | grep -v -i warn | tail -25
python 26_atlas20_msa.py 2>&1 | grep -v -i warn | tail -30
conda activate cf; export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
python 27_atlas20_fold.py 2>&1 | grep -v -i warn | tail -45
echo "ALL DONE" > results/atlas20.flag
