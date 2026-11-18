#!/usr/bin/env bash
# Quick end-to-end check of the pipeline on 8000 proteins / 10 queries.
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fold
T="$HOME/folding-data/poctest"
rm -rf "$T"; mkdir -p "$T"
awk '/^>/{n++} n<=8000' "$HOME/folding-data/poc/sprot.fasta" > "$T/sprot.fasta"
head -20 "$HOME/folding-data/poc/queries.fasta" > "$T/queries.fasta"
export POC_DATA="$T"
cd "$(dirname "$0")"
python 02_embed.py 2>&1 | grep -v -i warn | tail -6
python 04_embsearch.py 2>&1 | tail -2
python 03_baselines.py 2>&1 | tail -6
python 05_verify.py 2>&1 | tail -14
