E=$HOME/folding-data/poc/e2e
for p in mm_atlas hyb_atlas; do
  n=$(ls $E/out/$p/*_unrelaxed_rank_001*.pdb 2>/dev/null | wc -l); echo "$p done: $n/20"
  ls -lt --time-style=+%H:%M:%S $E/out/$p/*_unrelaxed_rank_001*.pdb 2>/dev/null | head -3 | awk '{print $6, $7}'
done
ls -d $E/retry_atlas/*/* 2>/dev/null
date +%H:%M:%S; ps aux | grep "[c]olabfold_batch" | awk '{print $NF, $(NF-8)}' | head
