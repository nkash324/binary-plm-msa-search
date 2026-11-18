#!/usr/bin/env bash
cd /mnt/c/Folding/poc
until grep -q "DHR MSAS DONE" results/dhr.log 2>/dev/null; do sleep 30; done
while pgrep -f "python 25_atlas_build" >/dev/null || pgrep -f "python 36_dhr" >/dev/null; do sleep 30; done
echo "starting folds $(date +%H:%M)"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
conda activate cf; export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
python 37_dhr_bigdb_fold.py 2>&1 | grep -v -i warn | tail -45
echo "ALL DONE" > results/dhr_bigdb.flag
