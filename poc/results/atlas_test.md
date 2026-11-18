# ESM Atlas small test: 20 fresh hard proteins vs 500,000 metagenomic proteins (one shard)

Index: Meta's public ESM-2 3B per-protein embeddings. Truth = structure: each hit's ESMFold model vs the protein's
real SCOPe structure (US-align, TM normalised by the query). TM > 0.5 = same fold.

Reproducing the Atlas embedding with our own ESM-2 3B run: cosine 1.0000, 1.0000, 1.0000.
Binary index 16.0 MB (32 B/protein). Query embedding 0.25 s, search 29 ms per protein (median).

| hits | top-1 is same fold | proteins with >=1 same-fold hit in top 10 | same-fold hits per protein | precision | mean TM | same-fold hits a sequence search (phmmer --max) could detect |
|---|---|---|---|---|---|---|
| binary 256-bit, top 10 | 14/20 | 17/20 | 5.9 | 59% | 0.57 | 41 of 118 |
| full float vectors, top 10 | 15/20 | 16/20 | 5.3 | 54% | 0.56 | 42 of 107 |
| random shard proteins (5/protein) | - | 0/20 | 0.0 | 0% | 0.32 | 0 of 0 |

## Per protein (binary index, top 10)

| protein | same-fold hits | best TM | sequence-detectable same-fold |
|---|---|---|---|
| d2cj4a_ | 4 | 0.55 | 0 |
| d2np5a2 | 8 | 0.78 | 1 |
| d3reta_ | 5 | 0.76 | 0 |
| d1pbya4 | 0 | 0.43 | 0 |
| d5muaa1 | 10 | 0.91 | 1 |
| d3llpa2 | 1 | 0.85 | 0 |
| d1cr5a1 | 9 | 0.83 | 0 |
| d1ei5a1 | 6 | 0.68 | 0 |
| d2b8ma1 | 10 | 0.78 | 0 |
| d5m17a_ | 7 | 0.80 | 0 |
| d1oi0a_ | 9 | 0.90 | 3 |
| d1guqa1 | 6 | 0.91 | 6 |
| d1twua_ | 10 | 0.82 | 1 |
| d3l20a1 | 10 | 0.90 | 9 |
| d2fyxa1 | 10 | 0.79 | 10 |
| d1xg0a_ | 0 | 0.30 | 0 |
| d1odha_ | 1 | 0.51 | 0 |
| d2dzqa1 | 0 | 0.35 | 0 |
| d4ltta1 | 10 | 0.96 | 10 |
| d4gq4a1 | 2 | 0.53 | 0 |
