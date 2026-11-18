# Test A: does adding BinPLM rows / Atlas rows help AlphaFold on top of big-database MSAs? (34 folded of 34)

Server MSAs so far: 37 of 200 (Neff 80%: <30: 7, 30-100: 3, >=100: 27).  AlphaFold: 1 model, 3 recycles, 3 seeds; TM-score = mean over seeds.  Better/worse = change > 0.05.

## shallow: Neff < 30 (main result): 7 proteins

| MSA | mean TM | median TM | correct folds (TM>0.5) |
|---|---|---|---|
| big-db MMseqs2 (ColabFold) alone | 0.663 | 0.711 | 4/7 |
| + BinPLM Swiss-Prot rows | 0.593 | 0.461 | 3/7 |
| + Atlas rows | 0.648 | 0.621 | 5/7 |

| comparison | mean TM change | better | worse | sign p | permutation p |
|---|---|---|---|---|---|
| + BinPLM Swiss-Prot rows vs alone | -0.070 | 0 | 3 | 0.25 | 0.154 |
| + Atlas rows vs alone | -0.015 | 1 | 2 | 1.00 | 0.782 |

## medium: Neff 30-100: 3 proteins

| MSA | mean TM | median TM | correct folds (TM>0.5) |
|---|---|---|---|
| big-db MMseqs2 (ColabFold) alone | 0.801 | 0.807 | 3/3 |
| + BinPLM Swiss-Prot rows | 0.601 | 0.625 | 2/3 |
| + Atlas rows | 0.788 | 0.770 | 3/3 |

| comparison | mean TM change | better | worse | sign p | permutation p |
|---|---|---|---|---|---|
| + BinPLM Swiss-Prot rows vs alone | -0.199 | 0 | 2 | 0.50 | 0.246 |
| + Atlas rows vs alone | -0.012 | 0 | 0 | 1.00 | 0.500 |

## deep: Neff >= 100: 24 proteins

| MSA | mean TM | median TM | correct folds (TM>0.5) |
|---|---|---|---|
| big-db MMseqs2 (ColabFold) alone | 0.881 | 0.910 | 24/24 |
| + BinPLM Swiss-Prot rows | 0.888 | 0.893 | 24/24 |
| + Atlas rows | 0.892 | 0.910 | 24/24 |

| comparison | mean TM change | better | worse | sign p | permutation p |
|---|---|---|---|---|---|
| + BinPLM Swiss-Prot rows vs alone | +0.007 | 1 | 0 | 1.00 | 0.763 |
| + Atlas rows vs alone | +0.011 | 1 | 0 | 1.00 | 0.528 |

## Per protein (mean TM over seeds)

| protein | big-db rows | Neff | + BinPLM rows | + Atlas rows | big-db MMseqs2 (ColabFold) alone | + BinPLM Swiss-Prot rows | + Atlas rows |
|---|---|---|---|---|---|---|---|
| d3w2wa2 | 3 | 1.0 | 93 | 24 | 0.46 | 0.39 | 0.44 |
| d2dlba1 | 18 | 2.2 | 78 | 55 | 0.31 | 0.34 | 0.51 |
| d1xaka_ | 29 | 3.4 | 6 | 4 | 0.76 | 0.46 | 0.62 |
| d1ezga_ | 72 | 6.3 | 15 | 38 | 0.96 | 0.96 | 0.96 |
| d1bmlc3 | 85 | 9.0 | 51 | 29 | 0.71 | 0.70 | 0.71 |
| d1l0sa_ | 48 | 14.9 | 112 | 81 | 0.47 | 0.34 | 0.34 |
| d2x9ba_ | 122 | 27.4 | 23 | 3 | 0.96 | 0.96 | 0.95 |
| d1wv3a1 | 67 | 42.5 | 133 | 6 | 0.96 | 0.43 | 0.93 |
| d2ftxa1 | 97 | 54.1 | 126 | 17 | 0.81 | 0.75 | 0.77 |
| d1iiea_ | 806 | 90.7 | 177 | 21 | 0.64 | 0.62 | 0.66 |
| d3c0na1 | 559 | 250.8 | 3 | 0 | 0.95 | 0.96 | 0.95 |
| d1bg8a_ | 812 | 280.7 | 6 | 2 | 0.92 | 0.92 | 0.92 |
| d2jnga1 | 1534 | 290.7 | 167 | 89 | 0.83 | 0.82 | 0.82 |
| d3ge3c_ | 744 | 291.0 | 177 | 89 | 0.98 | 0.98 | 0.98 |
| d2outa2 | 983 | 437.7 | 41 | 25 | 0.54 | 0.54 | 0.54 |
| d1ug8a1 | 1013 | 453.5 | 24 | 23 | 0.88 | 0.88 | 0.88 |
| d1pyya1 | 816 | 535.1 | 75 | 4 | 0.92 | 0.93 | 0.93 |
| d1l9la_ | 1383 | 560.9 | 98 | 14 | 0.97 | 0.97 | 0.97 |
| d2o4ta1 | 2771 | 645.9 | 68 | 6 | 0.78 | 0.78 | 0.78 |
| d2bw3a1 | 1008 | 692.2 | 68 | 0 | 0.85 | 0.84 | 0.85 |
| d3proc1 | 1261 | 744.1 | 3 | 3 | 0.90 | 0.88 | 0.88 |
| d2dofa1 | 2984 | 962.8 | 50 | 7 | 0.88 | 0.88 | 0.88 |
| d1tnsa_ | 3736 | 2264.9 | 90 | 61 | 0.80 | 0.80 | 0.80 |
| d1zhva1 | 5867 | inf | 197 | 97 | 0.87 | 0.87 | 0.87 |
| d2xmka_ | 7168 | inf | 146 | 90 | 0.96 | 0.96 | 0.96 |
| d2f06a1 | 6285 | inf | 105 | 88 | 0.93 | 0.94 | 0.93 |
| d1vmga_ | 9384 | inf | 199 | 98 | 0.98 | 0.98 | 0.98 |
| d2i5ua1 | 4540 | inf | 145 | 91 | 0.96 | 0.96 | 0.99 |
| d2p06a1 | 5613 | inf | 200 | 93 | 0.66 | 0.84 | 0.94 |
| d2phcb2 | 7431 | inf | 99 | 77 | 0.97 | 0.97 | 0.97 |
| d4qboa1 | 6654 | inf | 184 | 82 | 0.99 | 0.99 | 0.99 |
| d1b4ra_ | 8412 | inf | 24 | 93 | 0.84 | 0.83 | 0.84 |
| d1uz5a1 | 8823 | inf | 35 | 43 | 0.87 | 0.87 | 0.87 |
| d1f00i2 | 10463 | inf | 47 | 62 | 0.92 | 0.91 | 0.90 |
