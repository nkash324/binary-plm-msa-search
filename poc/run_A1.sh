#!/usr/bin/env bash
# Test A, stage 1+2: candidates -> ColabFold server screen -> depth.  Log: results/overnight.log
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; L=results/overnight.log
echo "$(date +%H:%M) A1 start" >> $L
conda activate fold
python 40_dark_select.py 2>&1 | grep -v -i warn >> $L
D=$HOME/folding-data/poc/e2e/dark
conda activate cf
s=$(date +%s)
colabfold_batch --msa-only --msa-mode mmseqs2_uniref_env "$D/candidates.fasta" "$D/msa" > "$D/server.log" 2>&1
echo "$(date +%H:%M) server screen done: $(ls $D/msa/*.a3m 2>/dev/null | wc -l) MSAs in $(( $(date +%s) - s ))s" >> $L
conda activate fold
python 41_dark_depth.py 2>&1 | grep -v -i warn >> $L
echo "$(date +%H:%M) A1 DONE" >> $L
