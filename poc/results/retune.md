# Step 1b: leak-free embedding-SW parameters (Swiss-Prot / AFDB pairs, no SCOPe)

291 pairs (15-30% identity MMseqs2 hits, TM >= 0.5); tuned on 146, tested on 145.

| aligner | F1 | sensitivity | precision |
|---|---|---|---|
| embedding SW, re-tuned (bias 0.1, gap 0.25) | 0.815 | 0.854 | 0.789 |
| embedding SW, old SCOPe-tuned (bias 0.05, gap 0.1) | 0.812 | 0.866 | 0.772 |
| BLOSUM62 SW | 0.699 | 0.689 | 0.736 |
