#!/usr/bin/env bash
# Per-step MMseqs2 timing, ColabFold settings, one protein at a time (6 threads), same 4 proteins as the profile.
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate fold
D=$HOME/folding-data/poc; W=$D/mm_steps; rm -rf $W; mkdir -p $W
for q in d1t0hb_ d4h03a1 d4ooza1 d2kl4a1; do
  awk -v q=">$q" 'BEGIN{p=0} /^>/{p=($1==q)} p' $D/queries.fasta > $W/$q.fa
  mmseqs createdb $W/$q.fa $W/$q.q > /dev/null
  /usr/bin/time -f "TOTAL %e" mmseqs search $W/$q.q $D/mmseqs_work/db $W/$q.res $W/tmp_$q --num-iterations 3 -s 8 -e 0.1 --max-seqs 10000 --threads 6 > $W/$q.log 2> $W/$q.time
  echo "== $q"; cat $W/$q.time | tail -1
  awk '/^(prefilter|align|result2profile|mergedbs|subtractdbs|swapresults|filterdb|concatdbs|expandaln|rescorediagonal|search|createsubdb) /{m=$1} /Time for processing/{print m, $4,$5,$6,$7}' $W/$q.log
done
