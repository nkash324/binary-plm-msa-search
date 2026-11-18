# Binary index test: 3 queries vs Swiss-Prot (20,000 proteins)

Sequences themselves: 8 MB (1 byte per amino acid). Index must be smaller to count as a win.
Recall = share of the true relatives (exhaustive phmmer --max) that make the candidate list.

| index | size MB | bytes/aa | s/query (CPU, 4 threads) | recall@500 | recall@2000 | recall@5000 | recall@20000 |
|---|---|---|---|---|---|---|---|
| protein-640 | 2 | 0.19 | 0.01 | nan | nan | nan | nan |
| protein-pca-128 | 0 | 0.04 | 0.00 | nan | nan | nan | nan |
| protein-pca-256 | 1 | 0.08 | 0.00 | nan | nan | nan | nan |
| window-32-128 | 4 | 0.52 | 0.03 | nan | nan | nan | nan |
| window-32-256 | 9 | 1.04 | 0.03 | nan | nan | nan | nan |
| window-64-128 | 2 | 0.27 | 0.00 | nan | nan | nan | nan |
| window-64-256 | 4 | 0.54 | 0.01 | nan | nan | nan | nan |

## Real alignment of the candidates (phmmer, 1 CPU core per query)

| index | K | mode | align s/query | final recall |
|---|---|---|---|---|
| protein-640 | 500 | filtered | 0.07 | nan |
| protein-640 | 500 | max | 0.19 | nan |
| protein-640 | 2000 | filtered | 0.04 | nan |
| protein-640 | 2000 | max | 0.61 | nan |
| window-32-128 | 500 | filtered | 0.04 | nan |
| window-32-128 | 500 | max | 0.21 | nan |
| window-32-128 | 2000 | filtered | 0.05 | nan |
| window-32-128 | 2000 | max | 0.62 | nan |

For comparison (same 100-query run last night): JackHMMER 1.35 s + 0.23 s align, recall 0.88; MMseqs2 0.47 s (6 threads), recall 0.994.
