#!/usr/bin/env bash
# End-to-end test. Log: C:\Folding\poc\results\e2e.log
set -uo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc
conda activate fold
echo "== $(date '+%F %T') pick proteins + build MSAs (4 pipelines)" >> results/e2e.log
python 18_e2e.py > /dev/null 2>> results/e2e.err || { echo "!! FAILED (see e2e.err)" >> results/e2e.log; echo "ALL DONE" > results/e2e.flag; exit 1; }
python drop_file_cache.py
echo "== $(date '+%F %T') fold + score" >> results/e2e.log
conda activate cf
export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
python 19_e2e_fold.py > /dev/null 2>> results/e2e.err || echo "!! FAILED fold (see e2e.err)" >> results/e2e.log
echo "== $(date '+%F %T') ALL DONE" >> results/e2e.log
echo "ALL DONE" > results/e2e.flag
