# Atlas 20 shards (~8M proteins) vs 1 shard (0.5M): 19 fresh hard proteins

Same method as the 1-shard test (binary index -> top 100 -> 150M embedding alignment -> chance cutoff); only the database is bigger. AlphaFold: ColabFold, 1 model, 3 recycles, no templates.

| MSA | mean TM | median TM | correct folds (TM>0.5) |
|---|---|---|---|
| JackHMMER | 0.590 | 0.477 | 9/19 |
| MMseqs2 | 0.669 | 0.725 | 12/19 |
| BinPLM (Swiss-Prot) | 0.756 | 0.897 | 15/19 |
| Atlas 1 shard | 0.709 | 0.827 | 14/19 |
| MMseqs2 + Atlas 1 shard | 0.810 | 0.884 | 16/19 |
| Atlas 20 shards | 0.789 | 0.918 | 15/19 |
| MMseqs2 + Atlas 20 shards | 0.848 | 0.911 | 17/19 |

| comparison | mean TM change | better | worse | sign p | permutation p |
|---|---|---|---|---|---|
| Atlas 20 shards vs Atlas 1 shard | +0.080 | 6 | 0 | 0.03 | 0.000 |
| Atlas 20 shards vs MMseqs2 | +0.119 | 10 | 3 | 0.09 | 0.129 |
| Atlas 20 shards vs BinPLM (Swiss-Prot) | +0.032 | 2 | 0 | 0.50 | 0.003 |
| MMseqs2 + Atlas 20 shards vs MMseqs2 + Atlas 1 shard | +0.039 | 3 | 1 | 0.62 | 0.161 |
| MMseqs2 + Atlas 20 shards vs MMseqs2 | +0.179 | 11 | 1 | 0.01 | 0.001 |

| protein | rows 1 shard | rows 20 shards (from new shards) | JackHMMER | MMseqs2 | BinPLM (Swiss-Prot) | Atlas 1 shard | MMseqs2 + Atlas 1 shard | Atlas 20 shards | MMseqs2 + Atlas 20 shards |
|---|---|---|---|---|---|---|---|---|---|
| d1cr5a1 | 0 | 0 (0) | 0.48 | 0.86 | 0.30 | 0.31 | 0.86 | 0.31 | 0.86 |
| d1ei5a1 | 23 | 49 (45) | 0.41 | 0.95 | 0.93 | 0.92 | 0.95 | 0.93 | 0.91 |
| d1guqa1 | 98 | 100 (95) | 0.70 | 0.70 | 0.73 | 0.69 | 0.75 | 0.75 | 0.75 |
| d1odha_ | 8 | 12 (12) | 0.31 | 0.31 | 0.38 | 0.29 | 0.47 | 0.39 | 0.41 |
| d1oi0a_ | 86 | 94 (84) | 0.87 | 0.83 | 0.92 | 0.93 | 0.93 | 0.94 | 0.94 |
| d1pbya4 | 8 | 38 (38) | 0.28 | 0.28 | 0.55 | 0.28 | 0.28 | 0.88 | 0.88 |
| d1twua_ | 99 | 100 (93) | 0.47 | 0.66 | 0.87 | 0.88 | 0.88 | 0.91 | 0.91 |
| d1xg0a_ | 2 | 93 (93) | 0.29 | 0.36 | 0.21 | 0.20 | 0.26 | 0.22 | 0.26 |
| d2b8ma1 | 100 | 100 (86) | 0.46 | 0.46 | 0.87 | 0.83 | 0.84 | 0.87 | 0.88 |
| d2cj4a_ | 1 | 23 (23) | 0.97 | 0.97 | 0.98 | 0.66 | 0.97 | 0.98 | 0.98 |
| d2dzqa1 | 0 | 0 (0) | 0.89 | 0.84 | 0.24 | 0.27 | 0.84 | 0.27 | 0.84 |
| d2fyxa1 | 100 | 100 (90) | 0.73 | 0.74 | 0.79 | 0.80 | 0.81 | 0.83 | 0.83 |
| d2np5a2 | 99 | 99 (92) | 0.43 | 0.43 | 0.93 | 0.92 | 0.92 | 0.92 | 0.92 |
| d3l20a1 | 100 | 100 (85) | 0.42 | 0.46 | 0.92 | 0.94 | 0.94 | 0.95 | 0.93 |
| d3llpa2 | 8 | 26 (22) | 0.97 | 0.95 | 0.98 | 0.95 | 0.96 | 0.97 | 0.98 |
| d3reta_ | 25 | 51 (46) | 0.53 | 0.90 | 0.91 | 0.80 | 0.88 | 0.95 | 0.94 |
| d4ltta1 | 100 | 100 (63) | 0.98 | 0.96 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 |
| d5m17a_ | 93 | 90 (70) | 0.68 | 0.73 | 0.90 | 0.88 | 0.91 | 0.97 | 0.98 |
| d5muaa1 | 32 | 99 (84) | 0.33 | 0.32 | 0.96 | 0.95 | 0.95 | 0.96 | 0.96 |

Time per protein, 1 shard: 1 query embedding (ESM-2 3B): 0.15s, 2 Atlas binary search: 0.03s, 3 fetch sequences (API, 8 parallel): 1.63s, 4 embed query + candidates (ESM-2 150M): 3.65s, 5 calibrate + align + build rows: 0.14s

Time per protein, 20 shards: 1 query embedding (ESM-2 3B): 0.16s, 2 Atlas binary search (20 shards): 0.26s, 3 fetch sequences (API, 8 parallel): 1.50s, 4 embed query + candidates (ESM-2 150M): 4.19s, 5 calibrate + align + build rows: 0.19s
