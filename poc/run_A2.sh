#!/usr/bin/env bash
# Test A stages 3-4 (unattended). Waits for the server screen + Atlas index; GPU steps hold /tmp/gpu.lock.
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; L=results/overnight.log
until grep -q "A1 DONE" $L; do sleep 60; done
while pgrep -f "python 25_atlas_build" >/dev/null; do sleep 60; done
echo "$(date +%H:%M) A: Atlas index $(ls ~/folding-data/atlas/idx20/*.ids.txt | wc -l) shards; starting rows" >> $L
conda activate fold
flock /tmp/gpu.lock python 46_dark_rows.py > results/A_rows.out 2>&1 || echo "$(date +%H:%M) A: rows step FAILED (see results/A_rows.out)" >> $L
conda activate cf; export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
grep -q "ROWS DONE" $L && flock /tmp/gpu.lock python 47_dark_fold.py > results/A_fold.out 2>&1
echo "$(date +%H:%M) A: retrying the 3 big-db folds that crashed earlier" >> $L
flock /tmp/gpu.lock python 37_dhr_bigdb_fold.py > results/A_retry37.out 2>&1
echo "$(date +%H:%M) A ALL DONE" >> $L
