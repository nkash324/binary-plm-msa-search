#!/usr/bin/env bash
# pLM-BLAST (test B) after the main chain is finished (one heavy job at a time), then rebuild the B table.
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; L=results/overnight.log
say() { echo "$(date +%H:%M) $*" >> $L; }
# starts right away: the chain is only waiting on the ColabFold server (light)
PB=$HOME/tools/plmblast_data
say "B2: pLM-BLAST (after chain)"; conda activate plmblast; cd ~/tools/plmblast
[ -d $PB/scope ] || python embeddings.py start $PB/scope.fas $PB/scope -embedder pt --gpu -bs 0 --asdir > /mnt/c/Folding/poc/results/plmblast_emb.out 2>&1
[ -d $PB/query ] || python embeddings.py start $PB/query.fas $PB/query -embedder pt --gpu -bs 0 --asdir >> /mnt/c/Folding/poc/results/plmblast_emb.out 2>&1
[ -f $PB/scope/0.emb ] || { say "B2: pLM-BLAST embeddings FAILED (results/plmblast_emb.out)"; echo "ALL DONE" > results/plmblast.flag; exit 1; }
say "B2: pLM-BLAST embeddings done ($(du -sh $PB/scope | cut -f1))"
[ -f $PB/query/emb.64 ] || python scripts/dbtofile.py $PB/query >> /mnt/c/Folding/poc/results/plmblast_emb.out 2>&1
s=$(date +%s)
python scripts/plmblast.py $PB/scope $PB/query $PB/hits.csv -cpc 70 -workers 5 > /mnt/c/Folding/poc/results/plmblast_search.out 2>&1
secs=$(( $(date +%s) - s )); say "B2: pLM-BLAST search finished in ${secs}s"
cd /mnt/c/Folding/poc; conda activate fold
python 44_B_plmblast_rank.py $PB/hits.csv $secs >> $L 2>&1 || say "B2: ranking FAILED"
python 45_B_eval.py > results/B_eval.out 2>&1
say "B2 DONE (results/B_benchmark.md updated)"
echo "ALL DONE" > results/plmblast.flag
