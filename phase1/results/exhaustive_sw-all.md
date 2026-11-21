# Exhaustive embedding SW on SCOPe40 (640 bits, all 15,177 queries)

bias 0.1, gap 0.25; every query vs all 15,177 domains + 15,177 reversed-sequence decoys. 0.03 s/query including decoys (3060 Ti, fused Triton kernel).

Sensitivity numbers: see poc/results/B_benchmark.md (rows exh640-raw / exh640-z).

Hits at E < 1e-3 (per query, mean): all 58.6, remote (same superfamily, other family) 23.2, wrong fold 13.85; queries with >= 1 wrong-fold hit: 7772/15177; decoys passing (expected ~0.001): 1.122; decoys at E < 1 (expected ~1): 4.89.

Hits scoring above the query's best decoy (empirical, no fit): 75.4 per query, of which wrong fold 30.29.
