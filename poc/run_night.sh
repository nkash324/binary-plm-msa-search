#!/usr/bin/env bash
# Overnight chain: ONE memory-heavy job at a time (the ColabFold server screen runs separately; it only waits on network).
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; L=results/overnight.log
say() { echo "$(date +%H:%M) $*" >> $L; }
P1=$HOME/folding-data/phase1; PB=$HOME/tools/plmblast_data; mkdir -p $PB

conda activate fold   # step 1 (Atlas index) already complete

# 2. ESM-1b vectors for PLMSearch (GPU)
say "chain: 2/6 ESM-1b vectors"
[ -f $P1/esm1b_mean.npy ] || python 43_B_plmsearch.py /nonexistent > results/night_esm1b.out 2>&1
[ -f $P1/esm1b_mean.npy ] && say "B: ESM-1b vectors ready" || say "B: ESM-1b FAILED (results/night_esm1b.out)"

# 3. pLM-BLAST: ProtT5 embeddings (GPU), then search 1000 fixed queries vs all (CPU, 6 workers)
say "chain: 3/6 pLM-BLAST"; conda activate plmblast; cd ~/tools/plmblast
cp $P1/seqs.fasta $PB/scope.fas
python3 - <<'PY'
import numpy as np
from pathlib import Path
P = Path.home() / "folding-data/phase1/seqs.fasta"; L = P.read_text().split(">")[1:]
sub = np.sort(np.random.default_rng(7).choice(len(L), 1000, replace=False))
(Path.home() / "tools/plmblast_data/query.fas").write_text("".join(">" + L[i] for i in sub))
PY
[ -d $PB/scope ] || python embeddings.py start $PB/scope.fas $PB/scope -embedder pt --gpu -bs 0 --asdir > /mnt/c/Folding/poc/results/night_plmblast_emb.out 2>&1
[ -d $PB/query ] || python embeddings.py start $PB/query.fas $PB/query -embedder pt --gpu -bs 0 --asdir >> /mnt/c/Folding/poc/results/night_plmblast_emb.out 2>&1
say "B: pLM-BLAST embeddings done ($(du -sh $PB/scope | cut -f1) for the database)"
s=$(date +%s)
python scripts/plmblast.py $PB/scope $PB/query $PB/hits.csv -cpc 70 -workers 6 > /mnt/c/Folding/poc/results/night_plmblast_search.out 2>&1
secs=$(( $(date +%s) - s )); say "B: pLM-BLAST search finished in ${secs}s"
cd /mnt/c/Folding/poc; conda activate fold
python 44_B_plmblast_rank.py $PB/hits.csv $secs >> $L 2>&1

# 4. PLMSearch scoring (needs the 13 GB archive; extract only the trained layer)
say "chain: 4/6 PLMSearch"; T=/mnt/f/tools/plmsearch
until [ -f $T/plmsearch_data.tar.gz ] && [ "$(stat -c %s $T/plmsearch_data.tar.gz)" -ge 12995086677 ]; do sleep 60; done
mkdir -p ~/tools/plmsearch_model; tar -xzf $T/plmsearch_data.tar.gz -C ~/tools/plmsearch_model --wildcards '*/model/plmsearch.sav' 2>> $L
SAV=$(find ~/tools/plmsearch_model -name plmsearch.sav | head -1)
python 43_B_plmsearch.py "$SAV" 2>&1 | grep -E "saved|DONE|Error" | sed "s/^/$(date +%H:%M) B: /" >> $L

# 5. Test A rows + folds (after the server screen)
say "chain: 5/6 test A (waiting for server screen)"
until grep -q "A1 DONE" $L; do sleep 60; done
python 46_dark_rows.py > results/A_rows.out 2>&1 || say "A: rows step FAILED (results/A_rows.out)"
conda activate cf; export TF_FORCE_UNIFIED_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
grep -q "ROWS DONE" $L && python 47_dark_fold.py > results/A_fold.out 2>&1
say "A: retrying the 3 big-db folds that crashed earlier"
python 37_dhr_bigdb_fold.py > results/A_retry37.out 2>&1

# 6. Test B table
say "chain: 6/6 test B table"; conda activate fold
python 45_B_eval.py > results/B_eval.out 2>&1
say "ALL DONE"; echo "ALL DONE" > results/overnight.flag
