# Exhaustive embedding SW on SCOPe40 (640 bits, 1000-query subset, old params)

bias 0.05, gap 0.1; every query vs all 15,177 domains + 15,177 reversed-sequence decoys. 0.03 s/query including decoys (3060 Ti, fused Triton kernel).

Sensitivity numbers: see poc/results/B_benchmark.md (rows exh640-raw / exh640-z).

Hits at E < 1e-3 (per query, mean): all 109.2, remote (same superfamily, other family) 22.5, wrong fold 62.16; queries with >= 1 wrong-fold hit: 444/1000; decoys passing (expected ~0.001): 1.041; decoys at E < 1 (expected ~1): 4.79.

Hits scoring above the query's best decoy (empirical, no fit): 132.0 per query, of which wrong fold 80.71.
