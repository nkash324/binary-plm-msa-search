#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate fold; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc
for P in sprot old; do PARAMS=$P nice -n 10 python 53_exh_scope.py 2>&1 | grep -v -i warn | tail -4; done
