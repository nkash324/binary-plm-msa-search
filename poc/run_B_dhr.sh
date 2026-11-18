#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; conda activate fold
python 42_B_dhr.py 2>&1 | grep -v -i warn | tail -5 | sed "s/^/$(date +%H:%M) B: /" >> results/overnight.log
