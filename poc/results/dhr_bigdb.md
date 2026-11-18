# DHR and big-database baseline: 16 fresh hard proteins

DHR: official weights, Swiss-Prot, top 2000 -> JackHMMER (DHR flags). Big databases: ColabFold MSA server (UniRef30 + ColabFoldDB). AlphaFold: ColabFold, 1 model, 3 recycles, no templates.

| MSA | mean TM | median TM | correct folds (TM>0.5) | MSA rows (median) |
|---|---|---|---|---|
| JackHMMER (Swiss-Prot) | 0.593 | 0.497 | 8/16 | 3 |
| MMseqs2 (Swiss-Prot) | 0.664 | 0.713 | 10/16 | 7 |
| BinPLM (Swiss-Prot) | 0.783 | 0.905 | 13/16 | 139 |
| DHR (Swiss-Prot) | 0.581 | 0.508 | 8/16 | 4 |
| Atlas 20 shards only | 0.801 | 0.922 | 13/16 | 96 |
| MMseqs2, big databases (ColabFold) | 0.919 | 0.968 | 16/16 | 2698 |
| big-db MMseqs2 + BinPLM rows | 0.921 | 0.967 | 16/16 | 2867 |
| big-db MMseqs2 + Atlas rows | 0.919 | 0.968 | 16/16 | 2794 |

| comparison | mean TM change | better | worse | sign p | permutation p |
|---|---|---|---|---|---|
| BinPLM (Swiss-Prot) vs DHR (Swiss-Prot) | +0.202 | 10 | 1 | 0.01 | 0.002 |
| DHR (Swiss-Prot) vs JackHMMER (Swiss-Prot) | -0.012 | 2 | 1 | 1.00 | 0.847 |
| DHR (Swiss-Prot) vs MMseqs2 (Swiss-Prot) | -0.083 | 1 | 4 | 0.38 | 0.111 |
| MMseqs2, big databases (ColabFold) vs MMseqs2 (Swiss-Prot) | +0.255 | 13 | 0 | 0.00 | 0.000 |
| MMseqs2, big databases (ColabFold) vs BinPLM (Swiss-Prot) | +0.136 | 8 | 0 | 0.01 | 0.000 |
| big-db MMseqs2 + BinPLM rows vs MMseqs2, big databases (ColabFold) | +0.002 | 0 | 0 | 1.00 | 0.552 |
| big-db MMseqs2 + Atlas rows vs MMseqs2, big databases (ColabFold) | -0.000 | 0 | 0 | 1.00 | 0.937 |
| Atlas 20 shards only vs MMseqs2, big databases (ColabFold) | -0.118 | 0 | 6 | 0.03 | 0.001 |

| protein | JackHMMER (Swiss-Prot) | MMseqs2 (Swiss-Prot) | BinPLM (Swiss-Prot) | DHR (Swiss-Prot) | Atlas 20 shards only | MMseqs2, big databases (ColabFold) | big-db MMseqs2 + BinPLM rows | big-db MMseqs2 + Atlas rows | rows: DHR / big-db |
|---|---|---|---|---|---|---|---|---|---|
| d1ei5a1 | 0.41 | 0.95 | 0.93 | 0.41 | 0.93 | 0.93 | 0.93 | 0.93 | 14 / 529 |
| d1guqa1 | 0.70 | 0.70 | 0.73 | 0.70 | 0.75 | 0.82 | 0.85 | 0.84 | 20 / 10843 |
| d1odha_ | 0.31 | 0.31 | 0.38 | 0.31 | 0.39 | 0.97 | 0.97 | 0.97 | 7 / 1824 |
| d1oi0a_ | 0.87 | 0.83 | 0.92 | 0.87 | 0.94 | 0.93 | 0.93 | 0.93 | 1 / 3126 |
| d1twua_ | 0.47 | 0.66 | 0.87 | 0.47 | 0.91 | 0.97 | 0.97 | 0.97 | 1 / 5418 |
| d1xg0a_ | 0.29 | 0.36 | 0.21 | 0.26 | 0.22 | 0.58 | 0.56 | 0.54 | 4 / 420 |
| d2b8ma1 | 0.46 | 0.46 | 0.87 | 0.55 | 0.87 | 0.98 | 0.98 | 0.98 | 3 / 2254 |
| d2cj4a_ | 0.97 | 0.97 | 0.98 | 0.97 | 0.98 | 0.97 | 0.97 | 0.97 | 8 / 947 |
| d2dzqa1 | 0.89 | 0.84 | 0.24 | 0.27 | 0.27 | 0.90 | 0.90 | 0.90 | 0 / 1983 |
| d2fyxa1 | 0.73 | 0.74 | 0.79 | 0.73 | 0.83 | 0.84 | 0.85 | 0.85 | 3 / 8361 |
| d2np5a2 | 0.43 | 0.43 | 0.93 | 0.43 | 0.92 | 0.96 | 0.96 | 0.96 | 0 / 1141 |
| d3l20a1 | 0.42 | 0.46 | 0.92 | 0.45 | 0.95 | 0.97 | 0.97 | 0.97 | 4 / 7628 |
| d3reta_ | 0.53 | 0.90 | 0.91 | 0.89 | 0.95 | 0.97 | 0.97 | 0.98 | 7 / 6867 |
| d4ltta1 | 0.98 | 0.96 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 3 / 7189 |
| d5m17a_ | 0.68 | 0.73 | 0.90 | 0.68 | 0.97 | 0.98 | 0.98 | 0.98 | 8 / 5321 |
| d5muaa1 | 0.33 | 0.32 | 0.96 | 0.31 | 0.96 | 0.97 | 0.98 | 0.97 | 8 / 2270 |

DHR time per protein (median): embed query 0.02s, vector search 0.20s, jackhmmer on retrieved 0.03s
Big-database MSAs came from the public ColabFold server (shared queue), so their wall time is not comparable.
