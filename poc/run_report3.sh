#!/usr/bin/env bash
# Stop the 'both' folds and report on alone / + BinPLM / + Atlas.
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; L=results/overnight.log
until grep -q "folded bigdb_atlas" $L; do sleep 20; done
pkill -f "[r]un_fold_avail.sh"; sleep 1; pkill -f "[4]7_dark_fold.py"; pkill -f "[c]olabfold_batch.*out3"; sleep 3
export A_SHALLOW=avail.txt A_ROWS=rows_avail.json A_REPORT=A_dark_avail.md A_SETS=bigdb,bigdb_yours,bigdb_atlas
conda activate cf
python 47_dark_fold.py > results/A_avail_report.out 2>&1 || echo "$(date +%H:%M) A-avail: report FAILED" >> $L
echo "$(date +%H:%M) A-avail report DONE" >> $L
