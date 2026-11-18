#!/usr/bin/env bash
# Wait until the bits128 folds are finished, then stop the fold script (precal/combined folds would be wasted).
L=/mnt/c/Folding/poc/results/ablation.log
until grep -q "bits128: folded" "$L"; do sleep 20; done
pkill -f "17_fold_ablation.py"; pkill -f "bin/colabfold_batch"
echo "$(date '+%H:%M:%S') stopped folding after bits128 (pre-computed chance level accepted 0 relatives; fixing)" >> "$L"
