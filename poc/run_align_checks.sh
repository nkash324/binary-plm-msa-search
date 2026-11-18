#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate fold; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/phase1
for b in 256 128; do BITS=$b python 09_align_check.py | tail -5; done
cd /mnt/c/Folding/poc && N_HARD=2 python 16_ablation.py 2>&1 | tail -12
