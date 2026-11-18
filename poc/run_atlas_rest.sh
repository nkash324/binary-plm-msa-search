#!/usr/bin/env bash
# Finish the Atlas folds (only proteins that folded in both other sets), then report alone / + BinPLM / + Atlas.
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; L=results/overnight.log; D=$HOME/folding-data/poc/e2e/dark
for q in $(cat $D/avail.txt); do
  a=$(ls $D/out3/bigdb/${q}_unrelaxed_rank_*.pdb 2>/dev/null | wc -l); b=$(ls $D/out3/bigdb_yours/${q}_unrelaxed_rank_*.pdb 2>/dev/null | wc -l)
  [ $a -ge 3 ] && [ $b -ge 3 ] && echo $q
done > $D/avail_ok.txt
echo "$(date +%H:%M) A-avail: Atlas folds resumed for $(wc -l < $D/avail_ok.txt) comparable proteins" >> $L
export A_SHALLOW=avail_ok.txt A_ROWS=rows_avail.json A_REPORT=A_dark_avail.md A_SETS=bigdb,bigdb_yours,bigdb_atlas
conda activate cf; export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
python 47_dark_fold.py > results/A_avail_report.out 2>&1 || echo "$(date +%H:%M) A-avail: report FAILED" >> $L
echo "$(date +%H:%M) A-avail report DONE" >> $L
