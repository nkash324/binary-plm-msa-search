# Codebook first pass + 640-bit rescoring (two-stage), SCOPe40 1000-query subset

Codebooks = k-means on Swiss-Prot 640-bit residue codes (600k residues; no SCOPe). Cell score = table lookup (c_a . c_b / 640, the expected 640-bit score). The first-pass bias/gap are rescaled per codebook so the cell-score distribution on random Swiss-Prot residue pairs matches the 640-bit one. Without this rescaling, every variant is far worse (cb1024 top 5%: recall 0.44, sensitivity 0.313). Hits = targets above the query's best decoy in the exhaustive 640-bit search (74 per query, 24 remote).

| variant | bytes / residue (first pass) | rescored | recall of exhaustive hits | recall of remote hits | ms / query | speed-up vs exhaustive |
|---|---|---|---|---|---|---|
| cb20 | 1 | top 1% | 0.181 | 0.212 | 25.3 | 0.7x |
| cb20 | 1 | top 5% | 0.273 | 0.306 | 21.5 | 0.8x |
| cb256 | 1 | top 1% | 0.370 | 0.438 | 15.2 | 1.2x |
| cb256 | 1 | top 5% | 0.498 | 0.552 | 20.8 | 0.9x |
| cb1024 | 2 | top 1% | 0.449 | 0.508 | 15.9 | 1.1x |
| cb1024 | 2 | top 5% | 0.606 | 0.641 | 21.4 | 0.8x |
| pq4 | 4 | top 1% | 0.378 | 0.443 | 22.9 | 0.8x |
| pq4 | 4 | top 5% | 0.507 | 0.568 | 29.9 | 0.6x |

Exhaustive 640-bit SW: 18.1 ms/query (80 B/residue).

Superfamily sensitivity (poc/results/B_benchmark.md): exhaustive 0.597; best two-stage (cb1024, top 5%) 0.413; cb1024 first pass alone 0.220; MMseqs2 0.298.

## Verdict: the draft's two-stage criterion fails
- Recall: at most 61% of the exhaustive hits survive the first pass (target was 95%).
- Speed: none. The lookup kernel runs no faster than the 640-bit kernel, because the time goes into the DP recurrence, not the cell scores (XOR + popcount on 10 words is cheap). So a cheaper cell score cannot speed up an exhaustive DP.
- Why recall is low: a residue's 640-bit code is far from its group centroid (mean cosine 0.13 / 0.20 / 0.22 for K = 20 / 256 / 1024), so most of the per-residue signal is lost in the letters. Learned alphabets (TEA) are trained for this; plain k-means on sign bits is not.
- Storage is still a real win: 1-2 B/residue (209-418 MB for Swiss-Prot) vs 80 B (16.7 GB).
- To get real speed, cut the DP work instead: an ungapped diagonal prefilter, or the window index, before the gapped SW.
