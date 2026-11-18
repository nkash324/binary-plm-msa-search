# Context test: small domains cut out of big proteins

Relatives = MMseqs2's Swiss-Prot hits (all same fold; AlphaFold folds these proteins correctly from them).
Window rank = where BinPLM fast search puts them among 576k proteins (top 200 become candidates).
z = embedding-alignment score; above the cutoff = accepted.

| protein | setup | relatives in top 200 | best rank | median rank | above cutoff | max z (cutoff) |
|---|---|---|---|---|---|---|
| d1cr5a1 | query = isolated domain (normal) | 0/14 | 16,657 | 16,657 | 1/14 | 19.2 (6.9) |
| d1cr5a1 | query = domain embedded inside its full parent protein | 14/14 | 1 | 10 | 14/14 | 131.1 (7.7) |
| d1cr5a1 | query isolated, relatives cut to their domain | - | - | - | 1/14 | 19.0 (6.9) |
| d2dzqa1 | query = isolated domain (normal) | 0/11 | 7,907 | 23,651 | 7/11 | 21.9 (7.8) |
| d2dzqa1 | query = domain embedded inside its full parent protein | 11/11 | 1 | 5 | 11/11 | 361.1 (5.8) |
| d2dzqa1 | query isolated, relatives cut to their domain | - | - | - | 4/11 | 20.0 (7.8) |
