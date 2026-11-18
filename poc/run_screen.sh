#!/usr/bin/env bash
# Test A server screen in chunks of 20 (one 150-sequence job sat PENDING for 1 h). CPU only, light.
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; L=results/overnight.log; D=$HOME/folding-data/poc/e2e/dark
say() { echo "$(date +%H:%M) $*" >> $L; }
conda activate cf
for round in $(seq 1 15); do
  python3 - <<'PY'
from pathlib import Path
D = Path.home() / "folding-data/poc/e2e/dark"
L = (D / "candidates.fasta").read_text().split(">")[1:]
todo = [r for r in L if not (D / "msa" / (r.split("\n")[0].strip() + ".a3m")).exists()]
(D / "todo.fasta").write_text("".join(">" + r for r in todo[:20])); print(len(todo))
PY
  [ -s $D/todo.fasta ] || break
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" timeout 3600 colabfold_batch --msa-only --msa-mode mmseqs2_uniref_env \
      "$D/todo.fasta" "$D/msa" >> "$D/server.log" 2>&1
  say "A: server screen: $(ls $D/msa/*.a3m | wc -l)/200 MSAs"
done
say "A: server screen finished: $(ls $D/msa/*.a3m | wc -l)/200 MSAs"
