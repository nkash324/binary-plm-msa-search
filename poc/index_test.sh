#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate fold
cd /mnt/c/Folding/poc
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
nice -n 10 python 08_index_test.py
