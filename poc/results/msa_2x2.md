# MSAs for the 2x2 (retrieval x aligner), 20 e2e targets (19 folded in e2e + d4gq4a1, L=209, which crashed ColabFold), Swiss-Prot

Every cell uses the same acceptance rule: per target and aligner, 200 random Swiss-Prot proteins (not candidates of either search) set a length-corrected chance level (Gumbel tail); candidates scoring above the 1-in-2000 level are kept. Rows: matched query columns only. Neff = number of effective sequences at 80% identity (query included). Files: ~/folding-data/poc/msa_2x2/<cell>/<target>.a3m

Candidates per target (median): MMseqs2 8, BinPLM 200 (shared 4).

| cell | retrieval | aligner | rows median | rows mean | Neff80 median | Neff80 mean | targets with 0 rows |
|---|---|---|---|---|---|---|---|
| mm_blosum | MMseqs2 hits | BLOSUM SW | 8 | 7.2 | 4.0 | 5.0 | 2 |
| mm_emb | MMseqs2 hits | embedding SW | 6 | 7.3 | 3.5 | 5.5 | 2 |
| binplm_blosum | BinPLM top 200 | BLOSUM SW | 7 | 9.3 | 5.5 | 6.8 | 1 |
| binplm_emb | BinPLM top 200 | embedding SW | 125 | 125.9 | 98.3 | 100.4 | 0 |
| native_mmseqs2 | (ref) native MMseqs2 MSA | MMseqs2 own | 8 | 8.2 | 4.0 | 5.8 | 2 |
| native_yours | (ref) e2e BinPLM MSA | embedding SW | 125 | 125.9 | 98.3 | 100.4 | 0 |

## Per target (rows)

| target | L | cand mm / binplm | mm_blosum | mm_emb | binplm_blosum | binplm_emb | native mm / BinPLM |
|---|---|---|---|---|---|---|---|
| d2cj4a_ | 147 | 16 / 200 | 10 | 16 | 13 | 164 | 16 / 164 |
| d2np5a2 | 118 | 0 / 200 | 0 | 0 | 1 | 195 | 0 / 195 |
| d3reta_ | 97 | 17 / 200 | 12 | 17 | 22 | 37 | 17 / 37 |
| d1pbya4 | 138 | 0 / 200 | 0 | 0 | 0 | 75 | 0 / 75 |
| d5muaa1 | 147 | 15 / 200 | 9 | 15 | 22 | 178 | 15 / 178 |
| d3llpa2 | 119 | 11 / 200 | 11 | 11 | 11 | 196 | 11 / 196 |
| d1cr5a1 | 82 | 14 / 200 | 12 | 1 | 2 | 118 | 14 / 118 |
| d1ei5a1 | 82 | 14 / 200 | 14 | 14 | 6 | 60 | 14 / 60 |
| d2b8ma1 | 108 | 1 / 200 | 1 | 1 | 14 | 200 | 1 / 200 |
| d5m17a_ | 349 | 8 / 200 | 8 | 8 | 11 | 148 | 8 / 148 |
| d1oi0a_ | 121 | 1 / 200 | 1 | 1 | 8 | 154 | 1 / 154 |
| d1guqa1 | 176 | 18 / 200 | 18 | 18 | 18 | 188 | 18 / 188 |
| d1twua_ | 137 | 2 / 200 | 2 | 2 | 22 | 172 | 2 / 172 |
| d3l20a1 | 148 | 4 / 200 | 4 | 4 | 6 | 130 | 4 / 130 |
| d2fyxa1 | 129 | 3 / 200 | 3 | 3 | 3 | 63 | 3 / 63 |
| d1xg0a_ | 76 | 14 / 200 | 14 | 14 | 5 | 120 | 14 / 120 |
| d1odha_ | 158 | 7 / 200 | 7 | 7 | 6 | 99 | 7 / 99 |
| d2dzqa1 | 86 | 11 / 200 | 11 | 7 | 1 | 70 | 11 / 70 |
| d4ltta1 | 88 | 3 / 200 | 3 | 3 | 10 | 54 | 3 / 54 |
| d4gq4a1 | 209 | 5 / 200 | 5 | 5 | 5 | 97 | 5 / 97 |
