#!/usr/bin/env bash
# Speed-up ablation. Log: C:\Folding\poc\results\ablation.log
set -uo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
LOGF=/mnt/c/Folding/poc/results/ablation.log
exec > >(tee -a "$LOGF") 2>&1
export PYTHONWARNINGS=ignore
fail() { echo "!! FAILED: $*"; echo "== ALL DONE"; exit 1; }
conda activate fold
D=$HOME/folding-data/poc
drop() { python /mnt/c/Folding/poc/drop_file_cache.py; }
echo "== $(date '+%F %T') 1/4 build 256- and 128-bit residue codes"
[ -s "$D/res_bits128.npy" ] || (cd /mnt/c/Folding/poc && python 15_build_small_codes.py) || fail codes
drop
echo "== $(date '+%F %T') 2/4 alignment accuracy with smaller codes (SCOPe held-out pairs)"
for b in 256 128; do
  [ -s "$D/emb_align_params_$b.json" ] || (cd /mnt/c/Folding/phase1 && BITS=$b python 09_align_check.py | tail -4) || fail "align check $b"
  drop
done
echo "== $(date '+%F %T') 3/4 speed + accepted relatives, every config"
(cd /mnt/c/Folding/poc && python 16_ablation.py) || fail ablation
drop
echo "== $(date '+%F %T') 4/4 AlphaFold accuracy, every config"
conda activate cf
export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
(cd /mnt/c/Folding/poc && python 17_fold_ablation.py 2>&1 | grep -v -i warn) || fail fold
echo "== $(date '+%F %T') ALL DONE"
