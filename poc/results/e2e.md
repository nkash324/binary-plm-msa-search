# End-to-end: sequence -> MSA -> AlphaFold structure, 19 fresh hard proteins

Database: Swiss-Prot (576k proteins) for every pipeline. AlphaFold: ColabFold, 1 model, 3 recycles, no templates.
Each protein run start to finish, one at a time, after a warm-up. JackHMMER & MMseqs2: 6 CPU threads, each tool's own
alignment/MSA output. BinPLM: GPU + CPU. Hybrid times are sequential (MMseqs2, then BinPLM).

## Time per protein (median seconds)

| pipeline | query prep | search | align + calibrate | build MSA | **sequence -> MSA** | AlphaFold | **total** |
|---|---|---|---|---|---|---|---|
| JackHMMER (AlphaFold-style) | 0.00 | 0.61 | 0.00 | 0.00 | **0.61** | 27 | **27** |
| MMseqs2 (ColabFold-style) | 0.00 | 5.82 | 0.00 | 0.09 | **5.90** | 27 | **33** |
| BinPLM (combined) | 0.08 | 0.24 | 1.33 | 0.15 | **1.80** | 30 | **32** |
| Hybrid (MMseqs2 + BinPLM) | 0.03 | 5.99 | 0.26 | 0.26 | **6.47** | 30 | **36** |

JackHMMER and MMseqs2 align inside their search step. One-time setup (not in the table): JackHMMER none; MMseqs2 index 21 s; BinPLM: Swiss-Prot embedding ~1.7 h GPU + window index ~1.5 min + 256-bit codes ~4 min; ESM-2 model load 3 s per session.

## Accuracy

| pipeline | mean TM | median TM | correct folds (TM>0.5) | mean pLDDT | MSA rows (median) | Neff 80% (median) |
|---|---|---|---|---|---|---|
| JackHMMER (AlphaFold-style) | 0.590 | 0.477 | 9/19 | 59.5 | 3 | 3.0 |
| MMseqs2 (ColabFold-style) | 0.669 | 0.725 | 12/19 | 66.9 | 8 | 4.0 |
| BinPLM (combined) | 0.756 | 0.897 | 15/19 | 76.7 | 130 | 105.7 |
| Hybrid (MMseqs2 + BinPLM) | 0.767 | 0.873 | 15/19 | 76.7 | 134 | 114.2 |

## Head to head (TM-score; better/worse = difference > 0.05)

| A vs B | mean TM difference (A - B) | A better | B better | sign-test p | permutation p |
|---|---|---|---|---|---|
| JackHMMER (AlphaFold-style) vs MMseqs2 (ColabFold-style) | -0.079 | 0 | 5 | 0.06 | 0.057 |
| JackHMMER (AlphaFold-style) vs BinPLM (combined) | -0.166 | 3 | 12 | 0.04 | 0.034 |
| JackHMMER (AlphaFold-style) vs Hybrid (MMseqs2 + BinPLM) | -0.177 | 2 | 12 | 0.01 | 0.017 |
| MMseqs2 (ColabFold-style) vs BinPLM (combined) | -0.087 | 3 | 10 | 0.09 | 0.246 |
| MMseqs2 (ColabFold-style) vs Hybrid (MMseqs2 + BinPLM) | -0.098 | 3 | 10 | 0.09 | 0.161 |
| BinPLM (combined) vs Hybrid (MMseqs2 + BinPLM) | -0.011 | 0 | 2 | 0.50 | 0.403 |

## Per protein (TM-score)

| protein | length | jackhmmer | mmseqs2 | BinPLM | hybrid | rows: jackhmmer / mmseqs2 / BinPLM / hybrid |
|---|---|---|---|---|---|---|
| d2cj4a_ | 147 | 0.97 | 0.97 | 0.98 | 0.98 | 5 / 16 / 164 / 170 |
| d2np5a2 | 118 | 0.43 | 0.43 | 0.93 | 0.93 | 0 / 0 / 195 / 195 |
| d3reta_ | 97 | 0.53 | 0.90 | 0.91 | 0.94 | 2 / 17 / 37 / 49 |
| d1pbya4 | 138 | 0.28 | 0.28 | 0.55 | 0.55 | 0 / 0 / 75 / 75 |
| d5muaa1 | 147 | 0.33 | 0.32 | 0.96 | 0.96 | 1 / 15 / 178 / 190 |
| d3llpa2 | 119 | 0.97 | 0.95 | 0.98 | 0.98 | 16 / 11 / 196 / 205 |
| d1cr5a1 | 82 | 0.48 | 0.86 | 0.30 | 0.30 | 9 / 14 / 118 / 132 |
| d1ei5a1 | 82 | 0.41 | 0.95 | 0.93 | 0.92 | 14 / 14 / 60 / 73 |
| d2b8ma1 | 108 | 0.46 | 0.46 | 0.87 | 0.87 | 1 / 1 / 200 / 201 |
| d5m17a_ | 349 | 0.68 | 0.73 | 0.90 | 0.87 | 8 / 8 / 148 / 153 |
| d1oi0a_ | 121 | 0.87 | 0.83 | 0.92 | 0.93 | 1 / 1 / 154 / 153 |
| d1guqa1 | 176 | 0.70 | 0.70 | 0.73 | 0.74 | 20 / 18 / 188 / 199 |
| d1twua_ | 137 | 0.47 | 0.66 | 0.87 | 0.87 | 1 / 2 / 172 / 174 |
| d3l20a1 | 148 | 0.42 | 0.46 | 0.92 | 0.90 | 1 / 4 / 130 / 134 |
| d2fyxa1 | 129 | 0.73 | 0.74 | 0.79 | 0.79 | 3 / 3 / 63 / 63 |
| d1xg0a_ | 76 | 0.29 | 0.36 | 0.21 | 0.27 | 8 / 14 / 120 / 129 |
| d1odha_ | 158 | 0.31 | 0.31 | 0.38 | 0.37 | 7 / 7 / 99 / 102 |
| d2dzqa1 | 86 | 0.89 | 0.84 | 0.24 | 0.41 | 48 / 11 / 70 / 81 |
| d4ltta1 | 88 | 0.98 | 0.96 | 0.98 | 0.98 | 3 / 3 / 54 / 54 |
