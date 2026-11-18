#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
conda activate fold
python 31_context_test.py 2>&1 | grep -v -i warn | grep -v -i unexpected | tail -20
