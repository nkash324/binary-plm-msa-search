#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; conda activate fold
python 43_B_plmsearch.py /nonexistent/plmsearch.sav > /tmp/esm1b.log 2>&1
ls -la ~/folding-data/phase1/esm1b_mean.npy && echo "$(date +%H:%M) B: ESM-1b vectors for SCOPe ready (PLMSearch scoring waits for its archive)" >> results/overnight.log
