# Exhaustive embedding SW on SCOPe40 (640 bits, 1000-query subset)

bias 0.1, gap 0.25; every query vs all 15,177 domains + 15,177 reversed-sequence decoys. 0.04 s/query including decoys (3060 Ti, fused Triton kernel).

Sensitivity numbers: see poc/results/B_benchmark.md (rows exh640-raw / exh640-z).

Hits at E < 1e-3 (per query, mean): all 59.8, remote (same superfamily, other family) 24.1, wrong fold 13.23; queries with >= 1 wrong-fold hit: 530/1000; decoys passing (expected ~0.001): 1.294; decoys at E < 1 (expected ~1): 5.19.

Hits scoring above the query's best decoy (empirical, no fit): 74.2 per query, of which wrong fold 27.27.

## Sensitivity (1000-query subset, from poc/results/B_benchmark.md)

| method | superfamily | family | fold | remote recall@1000 | s/query |
|---|---|---|---|---|---|
| exhaustive SW, leak-free params (bias 0.1, gap 0.25), z-ranked | **0.597** | 0.788 | 0.053 | 0.865 | 0.018 (+0.018 decoys) |
| exhaustive SW, leak-free params, raw score | 0.597 | 0.788 | 0.054 | 0.868 | 0.018 |
| exhaustive SW, old SCOPe-tuned params (0.05 / 0.1), z-ranked | 0.600 | 0.792 | 0.067 | 0.876 | 0.018 (+0.018) |
| PLMSearch | 0.559 | 0.732 | 0.092 | 0.903 | - |
| BinPLM residue-center-diag (same codes, diagonal votes) | 0.527 | 0.733 | 0.036 | 0.825 | 2.3 |
| MMseqs2 3-iter | 0.298 | 0.514 | 0.002 | 0.310 | - |

- Full Smith-Waterman on the same 640-bit codes beats diagonal voting by +0.07 and PLMSearch by +0.04
  (superfamily). The leak-free parameters lose nothing (0.597 vs 0.600), so the old leak did not inflate results.
- Raw and z-ranked are the same (0.597): the length correction doesn't matter for ranking.
- The draft criterion "strong = above 0.559" is met for exhaustive search. The two-stage half of that criterion is
  still open (step 3).
- Speed: fused Triton kernel on the RTX 3060 Ti, about 3 G cells/s.

## False homologs

- **The E-value fit is too optimistic.** The Gumbel tail fitted to reversed-sequence decoys is lighter than the real
  decoy tail: about 1.3 decoys per query reach E < 1e-3 (expected 0.001), and about 5 reach E < 1 (expected 1). Use
  "above the best decoy", or an empirical decoy count, instead of Gumbel E-values for acceptance.
- **Wrong-fold hits above the query's best decoy:** median 1 per query, mean 27 (heavy tail: 90th percentile 73,
  max 921). 537/1000 queries have at least 1, 355 have more than 5. The top-1 hit is a wrong fold for 90/1000
  queries.
- **They concentrate in α/β proteins (SCOPe class c):** 224/277 class-c queries have one, and 18,867 of 27,270
  wrong-fold hits are class-c targets. Membrane (f) and small proteins (g) contribute little. This looks like
  cross-fold similarity of shared βαβ units (SCOPe "different fold", often structurally similar motifs), not
  low-complexity junk.
- **What this means for MSAs:** a decoy-based cutoff alone will admit some look-alike rows from other α/β folds.
  This links to the earlier false-acceptance diagnosis (diagnose.md). One fix to test is a per-target
  coverage/consistency filter.
