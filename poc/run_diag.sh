#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
conda activate fold
python 28_diagnose.py 2>&1 | grep -v -i warn | tail -40
