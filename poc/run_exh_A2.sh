#!/usr/bin/env bash
# Option A, step 2-lite: exhaustive SW on the SCOPe 1000 subset (leak-free params, then old params), then the B table.
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate fold; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc
nice -n 10 python 53_exh_scope.py 2>&1 | grep -v -i "warn"
PARAMS=old nice -n 10 python 53_exh_scope.py 2>&1 | grep -v -i "warn"
nice -n 10 python 45_B_eval.py 2>&1 | grep -v -i "warn" | tail -16
echo "STEP2 DONE"
