# Test B: remote-relative search on SCOPe 2.08 (15,177 domains): BinPLM vs pLM-BLAST, PLMSearch, DHR

Sensitivity = share of a query's true relatives ranked above its first wrong-fold hit (standard SCOPe benchmark).
Superfamily = remote relatives (same superfamily, different family). Subset = the same 1000 random queries for every method (pLM-BLAST was only run on these).

| method | what | superfamily sens. (subset) | family sens. (subset) | fold sens. (subset) | remote recall@1000 (subset) | superfamily sens. (all queries) | bytes / protein | time (s) |
|---|---|---|---|---|---|---|---|---|
| exh640-z-old | exhaustive SW, z (old SCOPe-tuned params) | **0.600** | 0.792 | 0.067 | 0.876 | nan | 14,772 | 35 |
| exh640-raw-old | exhaustive SW, raw (old SCOPe-tuned params) | **0.597** | 0.786 | 0.065 | 0.879 | nan | 14,772 | 17 |
| exh640-raw | BinPLM: exhaustive embedding SW, raw score | **0.597** | 0.788 | 0.054 | 0.868 | nan | 14,772 | 18 |
| exh640-z | BinPLM: exhaustive embedding SW, decoy z-score | **0.597** | 0.788 | 0.053 | 0.865 | nan | 14,772 | 36 |
| plmsearch_nopfam | PLMSearch | **0.559** | 0.732 | 0.092 | 0.903 | 0.560 | 5,120 | 0 |
| residue-center-diag | yours: full per-residue codes (slow upper bound) | **0.527** | 0.733 | 0.036 | 0.825 | 0.528 | 14,772 | 2260 |
| dhr | DHR | **0.499** | 0.672 | 0.038 | 0.797 | 0.494 | 960 | 68 |
| window-64-256 | yours: window index (search only) | **0.462** | 0.670 | 0.036 | 0.806 | 0.447 | 108 | 79 |
| twostage-K1000 | yours: window + embedding alignment | **0.448** | 0.660 | 0.034 | 0.614 | 0.446 | 14,852 | 601 |
| protein-pca-256 | yours: 32 B/protein | **0.413** | 0.630 | 0.033 | 0.753 | 0.410 | 32 | 7 |
| 2s-cb1024-5 | 2s-cb1024-5 | **0.413** | 0.632 | 0.019 | 0.516 | nan | 369 | 21 |
| 2s-pq4-5 | 2s-pq4-5 | **0.368** | 0.580 | 0.012 | 0.461 | nan | 739 | 30 |
| 2s-cb256-5 | 2s-cb256-5 | **0.358** | 0.578 | 0.014 | 0.444 | nan | 185 | 21 |
| jackhmmer | letters | **0.354** | 0.589 | 0.002 | 0.368 | 0.400 | 185 | 663 |
| 2s-cb1024-1 | 2s-cb1024-1 | **0.340** | 0.562 | 0.010 | 0.516 | nan | 369 | 16 |
| 2s-pq4-1 | 2s-pq4-1 | **0.299** | 0.514 | 0.006 | 0.461 | nan | 739 | 23 |
| mmseqs2_iter3 | letters (ColabFold-style) | **0.298** | 0.514 | 0.002 | 0.310 | 0.311 | 185 | 122 |
| 2s-cb256-1 | 2s-cb256-1 | **0.291** | 0.509 | 0.007 | 0.444 | nan | 185 | 15 |
| cb-cb1024 | cb-cb1024 | **0.220** | 0.414 | 0.002 | 0.516 | nan | 369 | 14 |
| 2s-cb20-5 | 2s-cb20-5 | **0.206** | 0.371 | 0.009 | 0.250 | nan | 185 | 22 |
| cb-pq4 | cb-pq4 | **0.188** | 0.369 | 0.000 | 0.461 | nan | 739 | 20 |
| cb-cb256 | cb-cb256 | **0.186** | 0.362 | 0.001 | 0.444 | nan | 185 | 13 |
| 2s-cb20-1 | 2s-cb20-1 | **0.145** | 0.291 | 0.003 | 0.250 | nan | 185 | 25 |
| cb-cb20 | cb-cb20 | **0.071** | 0.147 | 0.000 | 0.250 | nan | 185 | 13 |

Time = total for the whole benchmark as each method was run (pLM-BLAST: 1000 queries only; not directly comparable). Bytes/protein = stored search data per database protein.
