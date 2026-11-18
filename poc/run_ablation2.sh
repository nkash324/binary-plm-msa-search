#!/usr/bin/env bash
# After the first fold pass is stopped: run the fixed calibration configs, then fold + final report.
set -uo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
L=/mnt/c/Folding/poc/results/ablation.log
until grep -q "stopped folding after bits128" "$L"; do sleep 20; done
while pgrep -f "17_fold_ablation.py" > /dev/null; do sleep 5; done
exec > >(tee -a "$L") 2>&1
export PYTHONWARNINGS=ignore
rm -rf "$HOME/folding-data/poc/fold/out_abl/precal" "$HOME/folding-data/poc/fold/out_abl/combined" \
       "$HOME/folding-data/poc/fold/a3m_abl/precal"* "$HOME/folding-data/poc/fold/a3m_abl/combined"*
python /mnt/c/Folding/poc/drop_file_cache.py
echo "== $(date '+%F %T') 3b: fixed calibration (200 random + fitted tail): cal200, combined"
conda activate fold
(cd /mnt/c/Folding/poc && ONLY=cal200,combined python 16_ablation.py) || { echo "!! FAILED"; echo "== ALL DONE"; exit 1; }
python /mnt/c/Folding/poc/drop_file_cache.py
echo "== $(date '+%F %T') 4b: AlphaFold accuracy (finished folds are reused)"
conda activate cf
export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
(cd /mnt/c/Folding/poc && python 17_fold_ablation.py 2>&1 | grep -v -i warn) || echo "!! FAILED"
echo "== $(date '+%F %T') ALL DONE (final)"
echo "ALL DONE" > /mnt/c/Folding/poc/results/ablation_final.flag
