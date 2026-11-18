#!/usr/bin/env bash
# Run Phase 1 steps inside the 'fold' conda env.  Usage: bash run.sh [step ...]  (default: all)
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fold
cd "$(dirname "$0")"
steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(01_download 02_embed 03_mmseqs 04_search 05_evaluate)
for s in "${steps[@]}"; do
  echo "################ $s  $(date +%T)"
  python "$s.py"
done
echo "################ ALL DONE $(date +%T)"
