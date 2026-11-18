# Diagnosis: d1cr5a1, d2dzqa1 (no Atlas rows, wrong with BinPLM Swiss-Prot rows) and d1xg0a_ (93 Atlas rows, still wrong)

Truth = structure: TM-score of each relative's predicted model (AlphaFold DB for Swiss-Prot, ESMFold for Atlas) vs
the protein's real structure, normalised by the protein. TM > 0.5 = same fold.

## Swiss-Prot: MMseqs2's relatives vs BinPLM pipeline

| protein | length | MMseqs2 relatives | same fold (of those with a model) | in BinPLM top 200 | best rank | above BinPLM cutoff | best z vs cutoff | BinPLM rows | overlap with MMseqs2 | BinPLM rows same fold (sample) |
|---|---|---|---|---|---|---|---|---|---|---|
| d1cr5a1 | 82 | 14 | 14/14 | 0 | 287853 | 1 | 19.2 vs 6.9 | 118 | 0 | 1/25 |
| d2dzqa1 | 86 | 11 | 11/11 | 0 | 7914 | 7 | 21.9 vs 7.8 | 70 | 0 | 0/35 |
| d1xg0a_ | 76 | 14 | 14/14 | 5 | 6 | 14 | 54.6 vs 5.4 | 120 | 5 | 0/30 |

## Atlas (20 shards, 7.9M proteins)

| protein | top-100 same fold | best TM | same-fold above cutoff |
|---|---|---|---|
| d1cr5a1 | 87 | 0.86 | 0 |
| d2dzqa1 | 0 | 0.43 | 0 |

d1xg0a_: 93 accepted Atlas rows; same fold 0/93, median TM 0.25; median coverage of the protein 92%. MMseqs2 relatives same fold: 14/14.
