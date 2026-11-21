#!/usr/bin/env bash
# Option A, step 1: SCOPe decoys + Swiss-Prot-mean codes, then leak-free retune. Low priority, 3 threads.
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate fold; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc
nice -n 10 python 51_scope_decoys.py 2>&1 | grep -v -i "warn\|UNEXPECTED\|^Notes\|^- UNEX"
nice -n 10 python 52_retune.py 2>&1 | grep -v -i "warn"
echo "STEP1 DONE"
