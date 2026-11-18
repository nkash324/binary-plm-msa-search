#!/usr/bin/env bash
# wait until all 19 DHR folds exist, then score and report DHR immediately (big-db folds keep going)
cd /mnt/c/Folding/poc
until grep -q "folded dhr" results/dhr_bigdb.log 2>/dev/null; do sleep 20; done
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate cf
python 38_dhr_report.py 2>&1 | grep -v -i warn
