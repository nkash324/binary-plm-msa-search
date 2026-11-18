# Test A: proteins that stay shallow even with the big databases (7 folded of 7)

Screen: 200 fresh SCOPe domains (fewest Swiss-Prot relatives out of 1000 random) through the ColabFold server. Neff (80% identity) of their big-database MSAs: <30: 7, 30-100: 3, >=100: 27 (of 37).

| MSA | mean TM | median TM | correct folds (TM>0.5) |
|---|---|---|---|
| big-db MMseqs2 (ColabFold) alone | 0.716 | 0.784 | 5/7 |
| + BinPLM Swiss-Prot rows | 0.608 | 0.492 | 3/7 |
| + Atlas rows (full index) | 0.659 | 0.627 | 5/7 |
| + both | 0.608 | 0.490 | 3/7 |

| comparison | mean TM change | better | worse | sign p | permutation p |
|---|---|---|---|---|---|
| + BinPLM Swiss-Prot rows vs alone | -0.108 | 1 | 3 | 0.62 | 0.183 |
| + Atlas rows (full index) vs alone | -0.058 | 1 | 2 | 1.00 | 0.591 |
| + both vs alone | -0.108 | 1 | 3 | 0.62 | 0.282 |

| protein | big-db rows | Neff | + BinPLM rows | + Atlas rows | big-db MMseqs2 (ColabFold) alone | + BinPLM Swiss-Prot rows | + Atlas rows (full index) | + both |
|---|---|---|---|---|---|---|---|---|
| d3w2wa2 | 3 | 1.0 | 93 | 24 | 0.49 | 0.41 | 0.49 | 0.32 |
| d2dlba1 | 18 | 2.2 | 78 | 55 | 0.31 | 0.38 | 0.54 | 0.48 |
| d1xaka_ | 29 | 3.4 | 6 | 4 | 0.78 | 0.49 | 0.63 | 0.49 |
| d1ezga_ | 72 | 6.3 | 15 | 40 | 0.96 | 0.96 | 0.96 | 0.96 |
| d1bmlc3 | 85 | 9.0 | 51 | 32 | 0.71 | 0.72 | 0.72 | 0.75 |
| d1l0sa_ | 48 | 14.9 | 112 | 85 | 0.80 | 0.34 | 0.34 | 0.34 |
| d2x9ba_ | 122 | 27.4 | 23 | 3 | 0.96 | 0.96 | 0.94 | 0.92 |
