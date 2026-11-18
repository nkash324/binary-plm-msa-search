# New acceptance cutoff (random AND look-alike chance levels): 19 fresh hard proteins

Only proteins whose MSA changed were refolded; unchanged MSAs reuse the old fold. AlphaFold: ColabFold, 1 model, 3 recycles, no templates.

| MSA | mean TM | median TM | correct folds (TM>0.5) |
|---|---|---|---|
| JackHMMER | 0.590 | 0.477 | 9/19 |
| MMseqs2 | 0.669 | 0.725 | 12/19 |
| BinPLM (Swiss-Prot), old cutoff | 0.756 | 0.897 | 15/19 |
| Atlas 20 shards, old cutoff | 0.789 | 0.918 | 15/19 |
| BinPLM (Swiss-Prot), new cutoff | 0.687 | 0.837 | 12/19 |
| Atlas 20 shards, new cutoff | 0.733 | 0.860 | 14/19 |

| comparison | mean TM change | better | worse | sign p | permutation p |
|---|---|---|---|---|---|
| BinPLM (Swiss-Prot), new cutoff vs BinPLM (Swiss-Prot), old cutoff | -0.069 | 0 | 4 | 0.12 | 0.086 |
| Atlas 20 shards, new cutoff vs Atlas 20 shards, old cutoff | -0.056 | 0 | 5 | 0.06 | 0.010 |
| BinPLM (Swiss-Prot), new cutoff vs MMseqs2 | +0.018 | 7 | 3 | 0.34 | 0.753 |
| Atlas 20 shards, new cutoff vs MMseqs2 | +0.064 | 8 | 5 | 0.58 | 0.379 |

| protein | Swiss-Prot rows old -> new | Atlas rows old -> new | JackHMMER | MMseqs2 | BinPLM (Swiss-Prot), old cutoff | Atlas 20 shards, old cutoff | BinPLM (Swiss-Prot), new cutoff | Atlas 20 shards, new cutoff |
|---|---|---|---|---|---|---|---|---|
| d1cr5a1 | 118 -> 0 | 0 -> 0 | 0.48 | 0.86 | 0.30 | 0.31 | 0.31 | 0.31 |
| d1ei5a1 | 60 -> 17 | 49 -> 7 | 0.41 | 0.95 | 0.93 | 0.93 | 0.96 | 0.88 |
| d1guqa1 | 188 -> 158 | 95 -> 56 | 0.70 | 0.70 | 0.73 | 0.75 | 0.78 | 0.72 |
| d1odha_ | 99 -> 6 | 12 -> 0 | 0.31 | 0.31 | 0.38 | 0.39 | 0.33 | 0.29 |
| d1oi0a_ | 154 -> 11 | 94 -> 28 | 0.87 | 0.83 | 0.92 | 0.94 | 0.94 | 0.93 |
| d1pbya4 | 75 -> 0 | 38 -> 3 | 0.28 | 0.28 | 0.55 | 0.88 | 0.28 | 0.86 |
| d1twua_ | 172 -> 152 | 98 -> 19 | 0.47 | 0.66 | 0.87 | 0.91 | 0.90 | 0.90 |
| d1xg0a_ | 120 -> 5 | 93 -> 0 | 0.29 | 0.36 | 0.21 | 0.22 | 0.24 | 0.26 |
| d2b8ma1 | 200 -> 1 | 100 -> 3 | 0.46 | 0.46 | 0.87 | 0.87 | 0.46 | 0.72 |
| d2cj4a_ | 164 -> 79 | 23 -> 18 | 0.97 | 0.97 | 0.98 | 0.98 | 0.97 | 0.95 |
| d2dzqa1 | 70 -> 0 | 0 -> 0 | 0.89 | 0.84 | 0.24 | 0.27 | 0.27 | 0.27 |
| d2fyxa1 | 63 -> 26 | 100 -> 85 | 0.73 | 0.74 | 0.79 | 0.83 | 0.75 | 0.83 |
| d2np5a2 | 195 -> 118 | 99 -> 2 | 0.43 | 0.43 | 0.93 | 0.92 | 0.92 | 0.38 |
| d3l20a1 | 130 -> 76 | 100 -> 85 | 0.42 | 0.46 | 0.92 | 0.95 | 0.89 | 0.94 |
| d3llpa2 | 196 -> 79 | 26 -> 19 | 0.97 | 0.95 | 0.98 | 0.97 | 0.96 | 0.97 |
| d3reta_ | 37 -> 10 | 50 -> 24 | 0.53 | 0.90 | 0.91 | 0.95 | 0.89 | 0.95 |
| d4ltta1 | 54 -> 54 | 100 -> 8 | 0.98 | 0.96 | 0.98 | 0.98 | 0.98 | 0.98 |
| d5m17a_ | 148 -> 13 | 85 -> 4 | 0.68 | 0.73 | 0.90 | 0.97 | 0.84 | 0.82 |
| d5muaa1 | 178 -> 2 | 99 -> 99 | 0.33 | 0.32 | 0.96 | 0.96 | 0.38 | 0.96 |
