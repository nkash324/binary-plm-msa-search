#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate cf
export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
D=$HOME/folding-data/poc/e2e; T=$D/dbg; rm -rf $T; mkdir -p $T/in
cp $D/a3m/mm_atlas/d1xg0a_.a3m $T/in/
colabfold_batch $T/in $T/out --num-models 1 --num-recycle 3 --model-type alphafold2_ptm 2>&1 | grep -v -i warn | tail -15
echo "exit: ${PIPESTATUS[0]}"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
