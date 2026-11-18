# Test B: remote-relative search on SCOPe 2.08 (15,177 domains): BinPLM vs pLM-BLAST, PLMSearch, DHR

Sensitivity = share of a query's true relatives ranked above its first wrong-fold hit (standard SCOPe benchmark).
Superfamily = remote relatives (same superfamily, different family). Subset = the same 1000 random queries for every method (pLM-BLAST was only run on these).

| method | what | superfamily sens. (subset) | family sens. (subset) | fold sens. (subset) | remote recall@1000 (subset) | superfamily sens. (all queries) | bytes / protein | time (s) |
|---|---|---|---|---|---|---|---|---|
| plmsearch_nopfam | PLMSearch | **0.559** | 0.732 | 0.092 | 0.903 | 0.560 | 5,120 | 0 |
| residue-center-diag | BinPLM: full per-residue codes (slow upper bound) | **0.527** | 0.733 | 0.036 | 0.825 | 0.528 | 14,772 | 2260 |
| dhr | DHR | **0.499** | 0.672 | 0.038 | 0.797 | 0.494 | 960 | 68 |
| window-64-256 | BinPLM: window index (search only) | **0.462** | 0.670 | 0.036 | 0.806 | 0.447 | 108 | 79 |
| twostage-K1000 | BinPLM: window + embedding alignment | **0.448** | 0.660 | 0.034 | 0.614 | 0.446 | 14,852 | 601 |
| protein-pca-256 | BinPLM: 32 B/protein | **0.413** | 0.630 | 0.033 | 0.753 | 0.410 | 32 | 7 |
| jackhmmer | letters | **0.354** | 0.589 | 0.002 | 0.368 | 0.400 | 185 | 663 |
| mmseqs2_iter3 | letters (ColabFold-style) | **0.298** | 0.514 | 0.002 | 0.310 | 0.311 | 185 | 122 |

Time = total for the whole benchmark as each method was run (pLM-BLAST: 1000 queries only; not directly comparable). Bytes/protein = stored search data per database protein.
