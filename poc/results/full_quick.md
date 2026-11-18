# Quick full-length test: MMseqs2 vs Atlas (20 shards), 8 proteins

Search with the full-length parent protein (UniProt; max 1022 residues around the domain), MSA cropped back to the domain, domain folded (ColabFold, 1 model, 3 recycles, no templates). 'Domain query' = earlier results.

| MSA | mean TM | correct folds (TM>0.5) |
|---|---|---|
| MMseqs2, domain query | 0.541 | 3/8 |
| MMseqs2, full-length query | 0.567 | 4/8 |
| Atlas, domain query | 0.608 | 4/8 |
| Atlas, full-length query | 0.536 | 3/8 |

| protein | parent (length) | MMseqs2 rows domain / full | MMseqs2 TM domain / full | Atlas rows domain / full | Atlas TM domain / full |
|---|---|---|---|---|---|
| d1cr5a1 | P18759 (758) | 14 / 13 | 0.86 / 0.87 | 0 / 0 | 0.31 / 0.31 |
| d2dzqa1 | Q9UHL9 (959) | 11 / 7 | 0.84 / 0.57 | 0 / 0 | 0.27 / 0.27 |
| d1xg0a_ | Q00433 (128) | 14 / 14 | 0.36 / 0.42 | 93 / 77 | 0.22 / 0.25 |
| d1odha_ | P70348 (436) | 7 / 7 | 0.31 / 0.32 | 12 / 9 | 0.39 / 0.36 |
| d1pbya4 | Q8VUT0 (512) | 0 / 0 | 0.28 / 0.28 | 38 / 43 | 0.88 / 0.31 |
| d2b8ma1 | Q58174 (116) | 1 / 1 | 0.46 / 0.46 | 100 / 100 | 0.87 / 0.87 |
| d5muaa1 | Q75WT9 (286) | 15 / 13 | 0.32 / 0.76 | 99 / 75 | 0.96 / 0.95 |
| d3reta_ | Q51507 (101) | 17 / 17 | 0.90 / 0.85 | 51 / 36 | 0.95 / 0.96 |
