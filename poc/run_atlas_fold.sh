#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; rm -f results/atlas_fold.log
conda activate fold
python 22_atlas_msa.py 2>&1 | grep -v -i -E "warn|UNEXPECTED|^lm_head|^---|^Key|Notes|^\s*$" | tail -30 || exit 1
python drop_file_cache.py
conda activate cf; export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
python 23_atlas_fold.py 2>&1 | grep -v -i warn | tail -45
echo "ALL DONE" > results/atlas_fold.flag
