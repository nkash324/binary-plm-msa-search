#!/usr/bin/env bash
# Full proof-of-concept pipeline, throttled (lowest CPU/disk priority, 3 CPU workers, GPU rests ~25%).
# The CPU-only HMMER baselines run alongside the GPU embedding step; each query is still timed on its own core.
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fold
cd "$(dirname "$0")"
LOW="nice -n 19 ionice -c3"
[ -f "$HOME/folding-data/poc/sprot.fasta" ] || python 01_prepare.py

echo "== $(date +%T) start: embedding (GPU) + HMMER baselines (CPU) in parallel"
$LOW python 03_baselines.py phmmer_max > "$HOME/folding-data/poc/hmmer.log" 2>&1 &
HMMER_PID=$!
$LOW python 02_embed.py
echo "== $(date +%T) embedding done; waiting for HMMER baselines"
wait $HMMER_PID
cat "$HOME/folding-data/poc/hmmer.log"

echo "== $(date +%T) embedding search"
$LOW python 04_embsearch.py | tail -2
echo "== $(date +%T) mmseqs2"
$LOW python 03_baselines.py mmseqs_cf
echo "== $(date +%T) align + score"
$LOW python 05_verify.py
echo "== $(date +%T) POC DONE"
