# Alignment correctness, 2x2: retrieval source x aligner

Truth = structural superposition (US-align), residue pairs closer than 5 A. Cells: **F1** (sensitivity / precision) of aligned residue pairs, averaged over pairs. Embedding SW = 256-bit PCA residue codes, bias 0.05, gap 0.1 (tuned earlier on the other half of the SCOPe pairs). BLOSUM SW = BLOSUM62, gap open 11 / extend 1.

## A. SCOPe distant pairs (same superfamily, different family; 148 held-out pairs)

| aligner | F1 | sensitivity | precision |
|---|---|---|---|
| BLOSUM62 Smith-Waterman | 0.358 | 0.345 | 0.415 |
| MMseqs2 own alignment (pairwise, forced) | 0.181 | 0.183 | 0.183 |
| Embedding Smith-Waterman | 0.520 | 0.554 | 0.498 |

MMseqs2 returned no alignment for 114 of the pairs even with the e-value cut off and ungapped prefilter disabled (scored as 0 found). On the 34 pairs it did align, F1: BLOSUM SW 0.791, MMseqs2 0.789, embedding SW 0.838.

## B. 20 e2e targets vs the relatives each search returned

Relative structures: AlphaFold DB models of the Swiss-Prot entries. MMseqs2 hits = all rows of the e2e MMseqs2 MSA; BinPLM hits = up to 40 random rows per target of the e2e BinPLM MSA. 'MMseqs2 own' = the alignment MMseqs2 wrote into its MSA (3-iteration profile search).

### Relatives that really share the query's fold (TM >= 0.5)

| retrieval \ aligner | pairs | letters: BLOSUM SW | letters: MMseqs2 own | embedding SW |
|---|---|---|---|---|
| (a) MMseqs2 hits | 160 (18 targets) | 0.792 (0.78 / 0.82) | 0.807 (0.78 / 0.85) | 0.846 (0.86 / 0.84) |
| (b) BinPLM hits | 384 (19 targets) | 0.237 (0.21 / 0.36) | - | 0.617 (0.66 / 0.59) |

### All relatives with >= 20 structurally aligned residues

| retrieval \ aligner | pairs | letters: BLOSUM SW | letters: MMseqs2 own | embedding SW |
|---|---|---|---|---|
| (a) MMseqs2 hits | 161 (18 targets) | 0.787 (0.77 / 0.82) | 0.802 (0.78 / 0.84) | 0.841 (0.85 / 0.83) |
| (b) BinPLM hits | 695 (20 targets) | 0.133 (0.11 / 0.20) | - | 0.364 (0.39 / 0.34) |

Caveat: each hit set was accepted by its own aligner (MMseqs2 hits by MMseqs2's alignment score, BinPLM hits by the embedding SW score), so the diagonal cells have a selection advantage. The off-diagonal cells (BLOSUM on BinPLM hits, embedding SW on MMseqs2 hits) are the cleaner test of the aligner alone.
