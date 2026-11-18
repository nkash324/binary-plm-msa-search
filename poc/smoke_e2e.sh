#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; rm -f results/e2e.log results/e2e.err
conda activate fold
N_RUN=2 python 18_e2e.py 2>&1 | grep -v -i warn | tail -12
D=$HOME/folding-data/poc/e2e
for p in jackhmmer mmseqs2 BinPLM hybrid; do f=$(ls $D/a3m/$p/*.a3m | head -1); echo "-- $p: $(grep -c '^>' $f) seqs"; sed -n 3,4p $f | cut -c1-110; done
python drop_file_cache.py
conda activate cf; export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
python 19_e2e_fold.py 2>&1 | grep -v -i warn | head -30
