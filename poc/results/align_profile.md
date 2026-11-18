# Alignment profile: 4 proteins (lengths [219, 209, 363, 115], accepted [25, 84, 230, 84])

Total 5.47 s per protein

| part | s per protein | share |
|---|---|---|
| 0 window index search (CPU) | 0.657 | 12% |
| 1 chance level (2000 random): DP row loop (GPU) | 0.701 | 13% |
| 1 chance level (2000 random): pad+copy targets | 0.124 | 2% |
| 1 chance level (2000 random): read codes from disk | 0.003 | 0% |
| 1 chance level (2000 random): similarity grid (matmul) | 0.065 | 1% |
| 1 chance level (2000 random): unpack + send to GPU | 0.408 | 7% |
| 2 candidates (500): DP row loop (GPU) | 0.485 | 9% |
| 2 candidates (500): pad+copy targets | 0.050 | 1% |
| 2 candidates (500): read codes from disk | 0.001 | 0% |
| 2 candidates (500): similarity grid (matmul) | 0.088 | 2% |
| 2 candidates (500): unpack + send to GPU | 2.199 | 40% |
| 3 re-align accepted + traceback: DP row loop (GPU) | 0.176 | 3% |
| 3 re-align accepted + traceback: copy DP matrix to CPU | 0.084 | 2% |
| 3 re-align accepted + traceback: pad+copy targets | 0.007 | 0% |
| 3 re-align accepted + traceback: read codes from disk | 0.000 | 0% |
| 3 re-align accepted + traceback: similarity grid (matmul) | 0.003 | 0% |
| 3 re-align accepted + traceback: traceback (CPU, Python) | 0.043 | 1% |
| 3 re-align accepted + traceback: unpack + send to GPU | 0.018 | 0% |
| 4 decoys + traceback: DP row loop (GPU) | 0.179 | 3% |
| 4 decoys + traceback: copy DP matrix to CPU | 0.082 | 1% |
| 4 decoys + traceback: pad+copy targets | 0.010 | 0% |
| 4 decoys + traceback: read codes from disk | 0.000 | 0% |
| 4 decoys + traceback: similarity grid (matmul) | 0.008 | 0% |
| 4 decoys + traceback: traceback (CPU, Python) | 0.033 | 1% |
| 4 decoys + traceback: unpack + send to GPU | 0.030 | 1% |
| (not covered by stopwatches) | 0.015 | 0% |

GPU batches per protein: 1 chance level (2000 random) 11, 2 candidates (500) 6, 3 re-align accepted + traceback 2, 4 decoys + traceback 2
