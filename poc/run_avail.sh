#!/usr/bin/env bash
# Test A on every server MSA retrieved so far (rerun as more arrive; finished folds are reused).
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; L=results/overnight.log; D=$HOME/folding-data/poc/e2e/dark
say() { echo "$(date +%H:%M) $*" >> $L; }
conda activate fold
python 41_dark_depth.py 2>&1 | grep -v -i warn >> $L
cut -f1 $D/depth.tsv | tail -n +2 > $D/avail.txt
export A_SHALLOW=avail.txt A_ROWS=rows_avail.json A_REPORT=A_dark_avail.md
say "A-avail: $(wc -l < $D/avail.txt) proteins: building rows"
python 46_dark_rows.py > results/A_avail_rows.out 2>&1 || { say "A-avail: rows FAILED"; exit 1; }
conda activate cf; export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
python 47_dark_fold.py > results/A_avail_fold.out 2>&1 || say "A-avail: fold FAILED"
say "A-avail DONE"
