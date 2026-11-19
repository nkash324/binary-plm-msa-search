#!/usr/bin/env bash
# Step 1 (alignment correctness 2x2) then step 2 (MSAs for the 4 cells). Low priority, 3 CPU threads.
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate fold; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc
for s in "$@"; do nice -n 10 python "$s" 2>&1 | grep -v Warning; done
