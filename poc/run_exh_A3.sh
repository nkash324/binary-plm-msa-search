#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate fold; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc
nice -n 10 python 55_twostage_scope.py 2>&1 | grep -v -i warn
nice -n 10 python 45_B_eval.py cb-cb20 cb-cb256 cb-cb1024 cb-pq4 2s-cb20-1 2s-cb20-5 2s-cb256-1 2s-cb256-5 2s-cb1024-1 2s-cb1024-5 2s-pq4-1 2s-pq4-5 2>&1 | grep -v -i warn | sed -n 5,40p
