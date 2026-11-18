#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate fold; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc && python 21_atlas_test.py 2>&1 | grep -v -i -E "warn|^\s*$" | tail -45
