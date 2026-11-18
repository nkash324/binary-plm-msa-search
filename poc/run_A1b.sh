#!/usr/bin/env bash
# Test A server screen, resumable: only candidates without an MSA yet. Light on memory (waits on the server).
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; L=results/overnight.log; D=$HOME/folding-data/poc/e2e/dark
python3 - <<'PY'
from pathlib import Path
D = Path.home() / "folding-data/poc/e2e/dark"
L = (D / "candidates.fasta").read_text().split(">")[1:]
todo = [r for r in L if not (D / "msa" / (r.split("\n")[0].strip() + ".a3m")).exists()]
(D / "todo.fasta").write_text("".join(">" + r for r in todo))
print(len(todo), "candidates still to screen")
PY
conda activate cf
s=$(date +%s)
CUDA_VISIBLE_DEVICES="" colabfold_batch --msa-only --msa-mode mmseqs2_uniref_env "$D/todo.fasta" "$D/msa" >> "$D/server.log" 2>&1
echo "$(date +%H:%M) A: server screen done: $(ls $D/msa/*.a3m 2>/dev/null | wc -l)/200 MSAs ($(( $(date +%s) - s ))s)" >> $L
conda activate fold
python 41_dark_depth.py 2>&1 | grep -v -i warn >> $L
echo "$(date +%H:%M) A1 DONE" >> $L
