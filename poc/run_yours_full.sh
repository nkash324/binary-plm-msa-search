#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; rm -f results/yours_full.flag; export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
conda activate fold
python 34_yours_fulllength.py 2>&1 | grep -v -i -E "warn|unexpected" | tail -15
if grep -q "MSAS DONE" results/yours_full.log; then
  conda activate cf; export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
  python 34_yours_fulllength.py --fold 2>&1 | grep -v -i warn | tail -20
fi
echo "ALL DONE" > results/yours_full.flag
MSYS_NO_PATHCONV=1 true
