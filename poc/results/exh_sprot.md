# Swiss-Prot MSAs, 20 e2e targets: exhaustive vs two-stage vs window index

Same aligner for all (640-bit embedding SW, bias 0.1, gap 0.25) and the same cutoff: database-scale E < 0.1 (E = 576,073 x p; p from a Gumbel fit to the tail of 10,000 reversed-sequence decoys). Two-stage = cb1024 first pass, top 5% rescored. Files: ~/folding-data/poc/msa_exh/<method>/<target>.a3m

Calibration check: held-out decoys at E < 1, per target, mean 0.60 (expected 0.017).

| method | candidates / target | s / target | rows median | Neff80 median | targets with 0 rows | MMseqs2 hits accepted (sum) | database bytes scanned |
|---|---|---|---|---|---|---|---|
| exh640 | 576,073 | 3.52 | 208 | 144.0 | 0 | 135 / 164 | 16.7 GB (80 B/residue) |
| twostage | 28,803 | 150.93 | 78 | 53.0 | 1 | 124 / 164 | 0.42 GB first pass + 1% at 80 B/res |
| window | 200 | 1.12 | 60 | 40.9 | 2 | 106 / 164 | 143 MB window index + 200 proteins |
| depth_ctrl (2x2 binplm_emb rows, subsampled to exh640 depth) | - | - | 96 | 66.1 | - | - | - |

## Per target (rows)

| target | L | exh640 | twostage | window | exh rows not in window top-200 | MMseqs2 hits accepted exh/2s/win of n |
|---|---|---|---|---|---|---|
| d2cj4a_ | 147 | 172 | 76 | 86 | 86 | 16/11/16 of 16 |
| d2np5a2 | 118 | 402 | 188 | 166 | 236 | 0/0/0 of 0 |
| d3reta_ | 97 | 39 | 30 | 10 | 29 | 17/15/6 of 17 |
| d1pbya4 | 138 | 1 | 0 | 1 | 0 | 0/0/0 of 0 |
| d5muaa1 | 147 | 447 | 118 | 159 | 288 | 15/11/12 of 15 |
| d3llpa2 | 119 | 547 | 121 | 190 | 357 | 11/11/11 of 11 |
| d1cr5a1 | 82 | 1 | 1 | 0 | 1 | 1/1/0 of 14 |
| d1ei5a1 | 82 | 19 | 16 | 8 | 11 | 14/14/5 of 14 |
| d2b8ma1 | 108 | 1938 | 299 | 200 | 1738 | 1/1/1 of 1 |
| d5m17a_ | 349 | 758 | 203 | 142 | 616 | 8/8/8 of 8 |
| d1oi0a_ | 121 | 529 | 480 | 66 | 463 | 1/1/1 of 1 |
| d1guqa1 | 176 | 353 | 123 | 168 | 185 | 18/18/18 of 18 |
| d1twua_ | 137 | 250 | 145 | 152 | 98 | 2/2/2 of 2 |
| d3l20a1 | 148 | 245 | 80 | 79 | 166 | 4/4/4 of 4 |
| d2fyxa1 | 129 | 866 | 526 | 29 | 837 | 3/3/3 of 3 |
| d1xg0a_ | 76 | 7 | 7 | 5 | 2 | 7/7/5 of 14 |
| d1odha_ | 158 | 7 | 7 | 6 | 1 | 7/7/6 of 7 |
| d2dzqa1 | 86 | 2 | 2 | 0 | 2 | 2/2/0 of 11 |
| d4ltta1 | 88 | 73 | 26 | 54 | 19 | 3/3/3 of 3 |
| d4gq4a1 | 209 | 9 | 5 | 8 | 1 | 5/5/5 of 5 |
