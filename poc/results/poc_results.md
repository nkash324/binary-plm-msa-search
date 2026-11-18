# Proof of concept: 100 SCOPe queries vs Swiss-Prot (576k proteins)

| method | search_s_per_query | align_s_per_query | candidates | verified | distant | recall | median_neff | median_neff_hard_half | shallow_msas(neff<30) |
|---|---|---|---|---|---|---|---|---|---|
| phmmer_max | 269 | 0.25 | 110 | 110 | 31.6 | 1 | 17.3 | 5 | 63 |
| hybrid_mmseqs_emb | 5.19 | 1.51 | 2.19e+03 | 110 | 31.4 | 0.995 | 17.3 | 5 | 63 |
| mmseqs_cf | 0.469 | 0.451 | 305 | 110 | 31.4 | 0.994 | 17.3 | 5 | 63 |
| jackhmmer_af2 | 1.35 | 0.226 | 97.8 | 97.8 | 24.4 | 0.88 | 14.2 | 3 | 64 |
| emb_twostage_k20000 | 15.5 | 1.37 | 2e+03 | 75.4 | 21.2 | 0.795 | 13.4 | 3.5 | 70 |
| emb_twostage | 4.72 | 1.31 | 2e+03 | 59.2 | 14.5 | 0.651 | 10 | 3 | 73 |
| emb_stage1only | 0.0308 | 1.31 | 2e+03 | 50.5 | 11.3 | 0.586 | 7.5 | 2 | 75 |

- **phmmer_max**: 269.5 s/query on 1 core; 51.9 s/query wall with 5 in parallel
- **hybrid_mmseqs_emb**: MMseqs2 + embedding candidates merged (5.2 s/query)
- **mmseqs_cf**: 0.5 s/query wall (6 threads, batch of 100)
- **jackhmmer_af2**: 1.3 s/query on 1 core; 0.5 s/query wall with 3 in parallel
- **emb_twostage_k20000**: 15.47 s/query: stage1 80 ms (CPU Hamming, 576,073 codes), disk read 14664 ms, stage2 725 ms (GPU, 20000 candidates)
- **emb_twostage**: 4.72 s/query: stage1 31 ms (CPU Hamming, 576,073 codes), disk read 4354 ms, stage2 339 ms (GPU, 5000 candidates)
- **emb_stage1only**: per-protein Hamming top 2000 only
