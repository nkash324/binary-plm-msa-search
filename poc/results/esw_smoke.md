# Step 0: exhaustive embedding-SW engine smoke test

- correctness vs emb_align.sw_many (50 pairs): max abs diff 0.0e+00
- 10 SCOPe queries (mean length 215) x 15,177 domains: **1.66 s/query**, 0.36 G cells/s, GPU peak 1.46 GB
- query itself is the top hit: 100%
- projected: SCOPe 1000 queries + decoys 47 min; Swiss-Prot 20 queries 29 min
