#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate cf; cd /mnt/c/Folding/poc
python 19_e2e_fold.py 2>&1 | grep -A8 "Time per protein"
