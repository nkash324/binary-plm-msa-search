# Scaling with database size (nested random subsets of Swiss-Prot, same 50 test proteins)

True relatives = exhaustive phmmer --max hits inside each subset (sequence-detectable relatives).

| proteins | true rel./query | **BinPLM**: search s | recall@200 | @500 | @2000 | candidates for 90% (median) | index MB | **JackHMMER** s | recall | **MMseqs2** s/query (batch) | recall | index MB | index build s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50,000 | 10 | 0.04 | 0.940 | 0.947 | 0.954 | 7.0 | 10 | 0.10 | 0.937 | 0.06 | 0.997 | 1043 | 3 |
| 100,000 | 20 | 0.08 | 0.908 | 0.951 | 0.963 | 22.0 | 20 | 0.16 | 0.907 | 0.09 | 0.997 | 1186 | 4 |
| 200,000 | 40 | 0.16 | 0.861 | 0.890 | 0.936 | 31.0 | 39 | 0.25 | 0.895 | 0.14 | 0.992 | 1473 | 6 |
| 400,000 | 81 | 0.32 | 0.788 | 0.851 | 0.926 | 95.0 | 79 | 0.43 | 0.886 | 0.19 | 0.998 | 2047 | 12 |
| 576,073 | 117 | 0.47 | 0.770 | 0.847 | 0.927 | 130.5 | 114 | 0.59 | 0.877 | 0.26 | 0.991 | 2551 | 16 |

Growth exponent: 1.0 = doubles when the database doubles; 0.5 = grows with the square root. Projection to 100x Swiss-Prot (~58M proteins) is a straight-line extrapolation on a log-log plot over an 11x range: rough.

| quantity | exponent | projected at 100x |
|---|---|---|
| yours_search_s | 1.05 | 58.7 |
| jackhmmer_s | 0.71 | 14.7 |
| mmseqs_s_per_query_batch50 | 0.57 | 3.49 |
| yours_index_MB | 1.00 | 1.15e+04 |
| mmseqs_index_MB | 0.37 | 1.28e+04 |
| yours_K_for_90pct_median | 1.17 | 2.87e+04 |
