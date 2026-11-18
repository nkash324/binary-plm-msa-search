# DHR vs BinPLM (both on Swiss-Prot): 19 fresh hard proteins

| MSA (Swiss-Prot) | mean TM | median TM | correct folds (TM>0.5) | MSA rows (median) |
|---|---|---|---|---|
| JackHMMER | 0.590 | 0.477 | 9/19 | 3 |
| MMseqs2 | 0.669 | 0.725 | 12/19 | 8 |
| BinPLM | 0.756 | 0.897 | 15/19 | 130 |
| DHR | 0.569 | 0.465 | 9/19 | 4 |

| comparison | mean TM change | better | worse | sign p | permutation p |
|---|---|---|---|---|---|
| BinPLM vs DHR | +0.188 | 11 | 1 | 0.01 | 0.000 |
| DHR vs JackHMMER | -0.021 | 2 | 2 | 1.00 | 0.671 |
| DHR vs MMseqs2 | -0.101 | 1 | 5 | 0.22 | 0.061 |

| protein | JackHMMER | MMseqs2 | BinPLM | DHR | rows: BinPLM / DHR |
|---|---|---|---|---|---|
| d1cr5a1 | 0.48 | 0.86 | 0.30 | 0.26 | 118 / 1 |
| d1ei5a1 | 0.41 | 0.95 | 0.93 | 0.41 | 60 / 14 |
| d1guqa1 | 0.70 | 0.70 | 0.73 | 0.70 | 188 / 20 |
| d1odha_ | 0.31 | 0.31 | 0.38 | 0.31 | 99 / 7 |
| d1oi0a_ | 0.87 | 0.83 | 0.92 | 0.87 | 154 / 1 |
| d1pbya4 | 0.28 | 0.28 | 0.55 | 0.28 | 75 / 0 |
| d1twua_ | 0.47 | 0.66 | 0.87 | 0.47 | 172 / 1 |
| d1xg0a_ | 0.29 | 0.36 | 0.21 | 0.26 | 120 / 4 |
| d2b8ma1 | 0.46 | 0.46 | 0.87 | 0.55 | 200 / 3 |
| d2cj4a_ | 0.97 | 0.97 | 0.98 | 0.97 | 164 / 8 |
| d2dzqa1 | 0.89 | 0.84 | 0.24 | 0.27 | 70 / 0 |
| d2fyxa1 | 0.73 | 0.74 | 0.79 | 0.73 | 63 / 3 |
| d2np5a2 | 0.43 | 0.43 | 0.93 | 0.43 | 195 / 0 |
| d3l20a1 | 0.42 | 0.46 | 0.92 | 0.45 | 130 / 4 |
| d3llpa2 | 0.97 | 0.95 | 0.98 | 0.97 | 196 / 16 |
| d3reta_ | 0.53 | 0.90 | 0.91 | 0.89 | 37 / 7 |
| d4ltta1 | 0.98 | 0.96 | 0.98 | 0.98 | 54 / 3 |
| d5m17a_ | 0.68 | 0.73 | 0.90 | 0.68 | 148 / 8 |
| d5muaa1 | 0.33 | 0.32 | 0.96 | 0.31 | 178 / 8 |

DHR time per protein (median): embed query 0.02s, vector search 0.20s, jackhmmer on retrieved 0.03s
