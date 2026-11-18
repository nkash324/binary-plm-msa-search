#!/usr/bin/env bash
# Unattended overnight run. Every step is resumable and failures don't stop later steps.
# Start:  wsl -d Ubuntu -- bash /mnt/c/Folding/poc/overnight.sh      (leave the window open)
# Log:    C:\Folding\poc\results\overnight.log
set -uo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
cd "$(dirname "$0")"
LOGF="$PWD/results/overnight.log"
exec > >(tee -a "$LOGF") 2>&1
D="$HOME/folding-data/poc"
rm -f "$D/PAUSE" "$D/gpu_rest"

# Full speed: nobody is using the PC.
export POC_CPU=6 POC_GPU_REST=0 POC_SUB_TOKENS=12000 POC_GPU_MEM=0.9 PYTHONWARNINGS=ignore
step() { echo "== $(date '+%F %T') $*"; }
run()  { "$@" || echo "!! FAILED (exit $?): $*"; }
quiet() { grep --line-buffered -v -i -E "warn|Loading weights|LOAD REPORT|UNEXPECTED|^Key|^---|^Notes|can be ignored|^$"; }

conda activate fold
step "PART 1: embedding (GPU) + exhaustive HMMER reference (CPU) in parallel"
( POC_CPU=5 run python 03_baselines.py phmmer_max ) > "$D/hmmer.log" 2>&1 &
HM=$!
run python 02_embed.py 2>&1 | quiet
step "embedding finished; waiting for HMMER reference"
wait $HM; cat "$D/hmmer.log"
[ -f "$D/hits/jackhmmer_af2.json" ] || run python 03_baselines.py jackhmmer_af2
step "two-step embedding search (K=5000)"
run python 04_embsearch.py | tail -1
step "MMseqs2 (ColabFold settings)"
run python 03_baselines.py mmseqs_cf

step "PART 2: bigger prefilter (K=20000) + hybrid"
POC_K1=20000 run python 04_embsearch.py | tail -1
run python 06_hybrid.py
step "alignment + scoring of all methods"
run python 05_verify.py

step "PART 3: folding test"
[ -n "$(ls $D/fold/ref/*.ent 2>/dev/null)" ] || run bash get_structures.sh
conda activate cf
export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8   # ColabFold defaults crash under WSL
run python 07_fold.py 40
step "ALL DONE"
