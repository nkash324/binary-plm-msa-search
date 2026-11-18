#!/usr/bin/env bash
# Steps 4 -> 1-3 -> 5-6. Log: C:\Folding\poc\results\align_fold.log
set -uo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
LOGF=/mnt/c/Folding/poc/results/align_fold.log
exec > >(tee -a "$LOGF") 2>&1
export PYTHONWARNINGS=ignore
echo "== $(date '+%F %T') step 4: alignment check on SCOPe distant pairs"
conda activate fold
cd /mnt/c/Folding/phase1 && python 09_align_check.py || { echo "!! FAILED"; echo "ALL DONE"; exit 1; }
if ! grep -q '"go": true' "$HOME/folding-data/poc/emb_align_params.json"; then
  echo "embedding aligner is not clearly better -> stopping before folding"; echo "ALL DONE"; exit 0
fi
echo "== $(date '+%F %T') steps 1-3: find + align distant relatives"
rm -rf "$HOME/folding-data/poc/embmsa"
cd /mnt/c/Folding/poc && python 11_find_align.py || { echo "!! FAILED"; echo "ALL DONE"; exit 1; }
echo "== $(date '+%F %T') steps 5-6: folding"
conda activate cf
export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
python 12_fold2.py 2>&1 | grep -v -i warn || echo "!! FAILED"
echo "== $(date '+%F %T') ALL DONE"
