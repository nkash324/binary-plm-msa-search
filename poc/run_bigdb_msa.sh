#!/usr/bin/env bash
# Big-database baseline: ColabFold's public MSA server (UniRef30 + ColabFoldDB, MMseqs2), the 19 test domains only.
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate cf
cd /mnt/c/Folding/poc
python - <<'PY'
import json
from common import DATA, RESULTS, read_fasta
E = DATA / "e2e"
qs = sorted({k.split("|")[1] for k in json.loads((RESULTS / "e2e.json").read_text())["accuracy"]})
ids, seqs = read_fasta(E / "queries.fasta"); S = dict(zip(ids, seqs))
(E / "bigdb").mkdir(exist_ok=True)
(E / "bigdb" / "queries19.fasta").write_text("".join(f">{q}\n{S[q]}\n" for q in qs))
print(len(qs), "queries written")
PY
E=$HOME/folding-data/poc/e2e/bigdb
/usr/bin/time -f "MSA server wall time %e s" colabfold_batch --msa-only --msa-mode mmseqs2_uniref_env "$E/queries19.fasta" "$E/msa" 2>&1 | grep -v -i warn | tail -8
ls "$E/msa"/*.a3m | wc -l
