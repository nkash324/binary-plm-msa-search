# Binary index test: 50 queries vs Swiss-Prot (576,073 proteins)

Sequences themselves: 209 MB (1 byte per amino acid). Index must be smaller to count as a win.
Recall = share of the true relatives (exhaustive phmmer --max) that make the candidate list.

| index | size MB | bytes/aa | s/query (CPU, 4 threads) | recall@500 | recall@2000 | recall@5000 | recall@20000 |
|---|---|---|---|---|---|---|---|
| protein-640 | 46 | 0.22 | 0.02 | 0.482 | 0.551 | 0.626 | 0.780 |
| protein-pca-128 | 9 | 0.04 | 0.00 | 0.603 | 0.706 | 0.754 | 0.813 |
| protein-pca-256 | 18 | 0.09 | 0.00 | 0.720 | 0.795 | 0.820 | 0.868 |
| window-32-128 | 109 | 0.52 | 0.52 | 0.803 | 0.880 | 0.913 | 0.930 |
| window-32-256 | 218 | 1.04 | 0.97 | 0.879 | 0.934 | 0.947 | 0.950 |
| window-64-128 | 57 | 0.27 | 0.12 | 0.733 | 0.832 | 0.864 | 0.869 |
| window-64-256 | 114 | 0.54 | 0.20 | 0.841 | 0.921 | 0.929 | 0.930 |

## Real alignment of the candidates (phmmer, 1 CPU core per query)

| index | K | mode | align s/query | final recall |
|---|---|---|---|---|
| protein-640 | 500 | filtered | 0.14 | 0.459 |
| protein-640 | 500 | max | 0.40 | 0.482 |
| protein-640 | 2000 | filtered | 0.18 | 0.528 |
| protein-640 | 2000 | max | 1.22 | 0.551 |
| window-64-256 | 500 | filtered | 0.22 | 0.817 |
| window-64-256 | 500 | max | 0.49 | 0.841 |
| window-64-256 | 2000 | filtered | 0.32 | 0.897 |
| window-64-256 | 2000 | max | 1.35 | 0.921 |

For comparison (same 100-query run last night): JackHMMER 1.35 s + 0.23 s align, recall 0.88; MMseqs2 0.47 s (6 threads), recall 0.994.
