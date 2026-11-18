# Adding ESM Atlas relatives (one 500k-protein shard) to the MSA: 19 fresh hard proteins

Atlas relatives: binary index of Meta's ESM-2 3B per-protein vectors -> top 100 -> embedding alignment -> only those above the chance cutoff. AlphaFold: ColabFold, 1 model, 3 recycles, no templates.

| MSA | mean TM | median TM | correct folds (TM>0.5) |
|---|---|---|---|
| JackHMMER | 0.590 | 0.477 | 9/19 |
| MMseqs2 | 0.669 | 0.725 | 12/19 |
| BinPLM (Swiss-Prot) | 0.756 | 0.897 | 15/19 |
| Hybrid (MMseqs2 + BinPLM) | 0.767 | 0.873 | 15/19 |
| MMseqs2 + Atlas | 0.810 | 0.884 | 16/19 |
| Hybrid + Atlas | 0.765 | 0.902 | 14/19 |

| comparison | mean TM change | better | worse | sign p | permutation p |
|---|---|---|---|---|---|
| MMseqs2 + Atlas vs MMseqs2 | +0.141 | 10 | 1 | 0.01 | 0.004 |
| Hybrid + Atlas vs Hybrid (MMseqs2 + BinPLM) | -0.002 | 0 | 2 | 0.50 | 0.832 |
| MMseqs2 + Atlas vs Hybrid (MMseqs2 + BinPLM) | +0.043 | 3 | 2 | 1.00 | 0.311 |
| Hybrid + Atlas vs MMseqs2 | +0.095 | 11 | 3 | 0.06 | 0.172 |

| protein | Atlas rows added | JackHMMER | MMseqs2 | BinPLM (Swiss-Prot) | Hybrid (MMseqs2 + BinPLM) | MMseqs2 + Atlas | Hybrid + Atlas |
|---|---|---|---|---|---|---|---|
| d1cr5a1 | 0 | 0.48 | 0.86 | 0.30 | 0.30 | 0.86 | 0.30 |
| d1ei5a1 | 23 | 0.41 | 0.95 | 0.93 | 0.92 | 0.95 | 0.93 |
| d1guqa1 | 98 | 0.70 | 0.70 | 0.73 | 0.74 | 0.75 | 0.77 |
| d1odha_ | 8 | 0.31 | 0.31 | 0.38 | 0.37 | 0.47 | 0.29 |
| d1oi0a_ | 86 | 0.87 | 0.83 | 0.92 | 0.93 | 0.93 | 0.94 |
| d1pbya4 | 8 | 0.28 | 0.28 | 0.55 | 0.55 | 0.28 | 0.42 |
| d1twua_ | 99 | 0.47 | 0.66 | 0.87 | 0.87 | 0.88 | 0.89 |
| d1xg0a_ | 2 | 0.29 | 0.36 | 0.21 | 0.27 | 0.26 | 0.26 |
| d2b8ma1 | 100 | 0.46 | 0.46 | 0.87 | 0.87 | 0.84 | 0.89 |
| d2cj4a_ | 1 | 0.97 | 0.97 | 0.98 | 0.98 | 0.97 | 0.98 |
| d2dzqa1 | 0 | 0.89 | 0.84 | 0.24 | 0.41 | 0.84 | 0.41 |
| d2fyxa1 | 100 | 0.73 | 0.74 | 0.79 | 0.79 | 0.81 | 0.81 |
| d2np5a2 | 99 | 0.43 | 0.43 | 0.93 | 0.93 | 0.92 | 0.93 |
| d3l20a1 | 100 | 0.42 | 0.46 | 0.92 | 0.90 | 0.94 | 0.94 |
| d3llpa2 | 8 | 0.97 | 0.95 | 0.98 | 0.98 | 0.96 | 0.98 |
| d3reta_ | 25 | 0.53 | 0.90 | 0.91 | 0.94 | 0.88 | 0.95 |
| d4ltta1 | 100 | 0.98 | 0.96 | 0.98 | 0.98 | 0.98 | 0.98 |
| d5m17a_ | 93 | 0.68 | 0.73 | 0.90 | 0.87 | 0.91 | 0.90 |
| d5muaa1 | 32 | 0.33 | 0.32 | 0.96 | 0.96 | 0.95 | 0.97 |

Time per protein for the Atlas part: 1 query embedding (ESM-2 3B): 0.15s, 2 Atlas binary search: 0.03s, 3 fetch sequences (API, 8 parallel): 1.63s, 4 embed query + candidates (ESM-2 150M): 3.65s, 5 calibrate + align + build rows: 0.14s
