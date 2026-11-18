# Quick full-length test: BinPLM Swiss-Prot method vs MMseqs2, 8 proteins

Search with the full-length parent protein (max 1022 aa around the domain), MSA cropped to the domain, domain folded (ColabFold, 1 model, 3 recycles, no templates). BinPLM unchanged (combined settings).

| MSA | mean TM | correct folds (TM>0.5) |
|---|---|---|
| MMseqs2, domain query | 0.541 | 3/8 |
| MMseqs2, full-length query | 0.567 | 4/8 |
| BinPLM (Swiss-Prot), domain query | 0.553 | 4/8 |
| BinPLM (Swiss-Prot), full-length query | 0.620 | 4/8 |

| protein | BinPLM rows domain / full | BinPLM TM domain / full | MMseqs2 TM domain / full |
|---|---|---|---|
| d1cr5a1 | 118 / 129 | 0.30 / 0.86 | 0.86 / 0.87 |
| d2dzqa1 | 70 / 180 | 0.24 / 0.39 | 0.84 / 0.57 |
| d1xg0a_ | 120 / 106 | 0.21 / 0.25 | 0.36 / 0.42 |
| d1odha_ | 99 / 141 | 0.38 / 0.26 | 0.31 / 0.32 |
| d1pbya4 | 75 / 101 | 0.55 / 0.48 | 0.28 / 0.28 |
| d2b8ma1 | 200 / 200 | 0.87 / 0.85 | 0.46 / 0.46 |
| d5muaa1 | 178 / 143 | 0.96 / 0.97 | 0.32 / 0.76 |
| d3reta_ | 37 / 74 | 0.91 / 0.91 | 0.90 / 0.85 |
