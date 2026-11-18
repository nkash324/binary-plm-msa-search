#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate fold
D=$HOME/folding-data/poc; W=$D/mm_steps
/usr/bin/time -f "createindex took %e s, peak RAM %M KB" mmseqs createindex $D/mmseqs_work/db $W/idx_tmp -s 8 --threads 6 > /dev/null 2> $W/createindex.time
tail -1 $W/createindex.time
echo "index size: $(du -ch $D/mmseqs_work/db.idx* | tail -1)"
for q in d1t0hb_ d4h03a1 d4ooza1 d2kl4a1; do
  rm -rf $W/$q.res* $W/tmp_$q
  /usr/bin/time -f "TOTAL %e" mmseqs search $W/$q.q $D/mmseqs_work/db $W/$q.res $W/tmp_$q --num-iterations 3 -s 8 -e 0.1 --max-seqs 10000 --threads 6 > $W/$q.log 2> $W/$q.time
  echo "== $q $(tail -1 $W/$q.time)"
  awk '/^(prefilter|align|result2profile) /{m=$1} /Time for processing/{if(m!=""){print m, $5,$6,$7}; m=""}' $W/$q.log | tr '\n' ';'; echo
done
