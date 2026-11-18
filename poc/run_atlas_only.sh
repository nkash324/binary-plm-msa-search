#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; rm -f results/atlas_only.flag
conda activate cf; export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
python 24_atlas_only.py 2>&1 | grep -v -i warn | tail -45
echo "ALL DONE" > results/atlas_only.flag
