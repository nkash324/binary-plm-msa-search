#!/usr/bin/env bash
# Extra: exhaustive SW on ALL 15,177 SCOPe40 queries (leak-free params), then the B table.
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate fold; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc
ALL=1 nice -n 10 python 53_exh_scope.py 2>&1 | grep -v -i "warn"
nice -n 10 python 45_B_eval.py exh640-raw-all exh640-z-all 2>&1 | grep -v -i "warn" | tail -20
echo "ALL DONE"
