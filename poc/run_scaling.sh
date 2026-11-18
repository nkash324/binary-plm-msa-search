#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate fold; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc && python 20_scaling.py 2>&1 | grep -v -i warn | tail -25; python drop_file_cache.py
