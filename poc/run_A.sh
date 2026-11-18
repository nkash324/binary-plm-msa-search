#!/usr/bin/env bash
# Test A end to end. Server screen is CPU-only and light; GPU steps wait until pLM-BLAST has finished.
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; L=results/overnight.log; D=$HOME/folding-data/poc/e2e/dark
say() { echo "$(date +%H:%M) $*" >> $L; }
conda activate cf
for pass in 1 2; do                                     # second pass picks up any the server dropped
  python3 - <<'PY'
from pathlib import Path
D = Path.home() / "folding-data/poc/e2e/dark"
L = (D / "candidates.fasta").read_text().split(">")[1:]
todo = [r for r in L if not (D / "msa" / (r.split("\n")[0].strip() + ".a3m")).exists()]
(D / "todo.fasta").write_text("".join(">" + r for r in todo)); print(len(todo))
PY
  [ -s $D/todo.fasta ] || break
  say "A: server screen pass $pass started"
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" colabfold_batch --msa-only --msa-mode mmseqs2_uniref_env "$D/todo.fasta" "$D/msa" >> "$D/server.log" 2>&1
done
n=$(ls $D/msa/*.a3m | wc -l); say "A: server screen: $n/200 MSAs"
[ "$n" -ge 150 ] || { say "A: screen FAILED (only $n MSAs, see server.log)"; echo "ALL DONE" > results/overnight.flag; exit 1; }
conda activate fold
python 41_dark_depth.py 2>&1 | grep -v -i warn >> $L
until [ -f results/plmblast.flag ]; do sleep 60; done     # one GPU job at a time
python 46_dark_rows.py > results/A_rows.out 2>&1 || { say "A: rows step FAILED (results/A_rows.out)"; echo "ALL DONE" > results/overnight.flag; exit 1; }
conda activate cf; export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
python 47_dark_fold.py > results/A_fold.out 2>&1 || say "A: fold step FAILED (results/A_fold.out)"
say "A: retrying the 3 big-db folds that crashed earlier"
python 37_dhr_bigdb_fold.py > results/A_retry37.out 2>&1
conda activate fold; python 45_B_eval.py > results/B_eval.out 2>&1
say "ALL DONE"; echo "ALL DONE" > results/overnight.flag
