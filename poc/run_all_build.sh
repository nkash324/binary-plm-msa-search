#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 ATLAS_LIST=atlas_all_shards.txt
conda activate fold
python 25_atlas_build.py 2>&1 | grep -v -i warn | tail -5
echo "ALL DONE" > results/atlas_all.flag
