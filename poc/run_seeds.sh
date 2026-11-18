#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
cd /mnt/c/Folding/poc; conda activate cf
echo "$(date +%H:%M) A-preview: seed check started" >> results/overnight.log
python A_seeds.py > results/A_seeds.out 2>&1 && echo "$(date +%H:%M) A-preview: seed check DONE" >> results/overnight.log || echo "$(date +%H:%M) A-preview: seed check FAILED" >> results/overnight.log
