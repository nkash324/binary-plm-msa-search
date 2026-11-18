# Atlas only (no MMseqs2, no Swiss-Prot): 19 fresh hard proteins

MSA = the protein + Atlas relatives (one 500k-protein shard; binary index -> top 100 -> embedding alignment -> chance cutoff). Proteins with 0 accepted rows fold from the single sequence. AlphaFold: ColabFold, 1 model, 3 recycles, no templates.

| MSA | mean TM | median TM | correct folds (TM>0.5) |
|---|---|---|---|
| JackHMMER | 0.590 | 0.477 | 9/19 |
| MMseqs2 | 0.669 | 0.725 | 12/19 |
| BinPLM (Swiss-Prot) | 0.756 | 0.897 | 15/19 |
| MMseqs2 + Atlas | 0.810 | 0.884 | 16/19 |
| Atlas only | 0.709 | 0.827 | 14/19 |

| comparison | mean TM change | better | worse | sign p | permutation p |
|---|---|---|---|---|---|
| Atlas only vs JackHMMER | +0.119 | 10 | 4 | 0.18 | 0.125 |
| Atlas only vs MMseqs2 | +0.039 | 8 | 5 | 0.58 | 0.593 |
| Atlas only vs BinPLM (Swiss-Prot) | -0.048 | 0 | 4 | 0.12 | 0.015 |
| Atlas only vs MMseqs2 + Atlas | -0.101 | 0 | 7 | 0.02 | 0.000 |

| protein | Atlas rows | JackHMMER | MMseqs2 | BinPLM (Swiss-Prot) | MMseqs2 + Atlas | Atlas only |
|---|---|---|---|---|---|---|
| d1cr5a1 | 0 | 0.48 | 0.86 | 0.30 | 0.86 | 0.31 |
| d1ei5a1 | 23 | 0.41 | 0.95 | 0.93 | 0.95 | 0.92 |
| d1guqa1 | 98 | 0.70 | 0.70 | 0.73 | 0.75 | 0.69 |
| d1odha_ | 8 | 0.31 | 0.31 | 0.38 | 0.47 | 0.29 |
| d1oi0a_ | 86 | 0.87 | 0.83 | 0.92 | 0.93 | 0.93 |
| d1pbya4 | 8 | 0.28 | 0.28 | 0.55 | 0.28 | 0.28 |
| d1twua_ | 99 | 0.47 | 0.66 | 0.87 | 0.88 | 0.88 |
| d1xg0a_ | 2 | 0.29 | 0.36 | 0.21 | 0.26 | 0.20 |
| d2b8ma1 | 100 | 0.46 | 0.46 | 0.87 | 0.84 | 0.83 |
| d2cj4a_ | 1 | 0.97 | 0.97 | 0.98 | 0.97 | 0.66 |
| d2dzqa1 | 0 | 0.89 | 0.84 | 0.24 | 0.84 | 0.27 |
| d2fyxa1 | 100 | 0.73 | 0.74 | 0.79 | 0.81 | 0.80 |
| d2np5a2 | 99 | 0.43 | 0.43 | 0.93 | 0.92 | 0.92 |
| d3l20a1 | 100 | 0.42 | 0.46 | 0.92 | 0.94 | 0.94 |
| d3llpa2 | 8 | 0.97 | 0.95 | 0.98 | 0.96 | 0.95 |
| d3reta_ | 25 | 0.53 | 0.90 | 0.91 | 0.88 | 0.80 |
| d4ltta1 | 100 | 0.98 | 0.96 | 0.98 | 0.98 | 0.98 |
| d5m17a_ | 93 | 0.68 | 0.73 | 0.90 | 0.91 | 0.88 |
| d5muaa1 | 32 | 0.33 | 0.32 | 0.96 | 0.95 | 0.95 |
