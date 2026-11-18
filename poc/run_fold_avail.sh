#!/usr/bin/env bash
# Resume test A folds on the 37 available proteins (rows already built; finished folds reused).
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; L=results/overnight.log
export A_SHALLOW=avail.txt A_ROWS=rows_avail.json A_REPORT=A_dark_avail.md
conda activate cf; export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
echo "$(date +%H:%M) A-avail: folds resumed (WSL capped at 4 GB)" >> $L
python 47_dark_fold.py > results/A_avail_fold.out 2>&1 || echo "$(date +%H:%M) A-avail: fold FAILED" >> $L
echo "$(date +%H:%M) A-avail DONE" >> $L
