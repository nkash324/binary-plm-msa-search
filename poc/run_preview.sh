#!/usr/bin/env bash
# Test A PREVIEW: the 7 shallow proteins (Neff<30) among the first 37 server MSAs. Separate shallow/rows/report files,
# so the full run (run_A.sh) is untouched; identical MSAs/folds get reused by it.
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; L=results/overnight.log; D=$HOME/folding-data/poc/e2e/dark
say() { echo "$(date +%H:%M) $*" >> $L; }
printf "d3w2wa2\nd2dlba1\nd1xaka_\nd1ezga_\nd1bmlc3\nd1l0sa_\nd2x9ba_\n" > $D/shallow_preview.txt
export A_SHALLOW=shallow_preview.txt A_ROWS=rows_preview.json A_REPORT=A_dark_preview.md
say "A-preview: started (7 proteins)"
conda activate fold
python 46_dark_rows.py > results/A_prev_rows.out 2>&1 || { say "A-preview: rows FAILED"; echo "ALL DONE" > results/preview.flag; exit 1; }
conda activate cf; export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
python 47_dark_fold.py > results/A_prev_fold.out 2>&1 || say "A-preview: fold FAILED"
say "A-preview DONE"; echo "ALL DONE" > results/preview.flag
