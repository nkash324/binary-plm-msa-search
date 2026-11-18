# Speed-ups: what each one costs and buys

20 development proteins. Window search 0.38 s/protein in every config (identical). Data is read the same way in every config (memory-mapped file + OS cache); only the method's own settings change.

## Speed (seconds per protein, search + alignment)

| config | candidates | code size | chance level | cold | warm |
|---|---|---|---|---|---|
| base | 500 | 640 bits (80 B/aa) | per query, 2000 random | 3.47 | 2.70 |
| k200 | 200 | 640 bits (80 B/aa) | per query, 2000 random | 2.10 | 1.51 |
| bits256 | 500 | 256 bits (32 B/aa) | per query, 2000 random | 3.08 | 1.93 |
| bits128 | 500 | 128 bits (16 B/aa) | per query, 2000 random | 2.21 | 1.92 |
| precal | 500 | 640 bits (80 B/aa) | pre-computed (one formula) | 2.55 | 1.73 |
| cal200 | 500 | 640 bits (80 B/aa) | per query, 200 random + fitted tail | 2.91 | 2.24 |
| combined | 200 | 256 bits (32 B/aa) | per query, 200 random + fitted tail | 1.52 | 0.83 |

Warm-run breakdown (s/protein):

| config | 1 chance level: align | 1 chance level: read from disk | 1 chance level: send to GPU | 2 candidates: align | 2 candidates: read from disk | 2 candidates: send to GPU | 3 accepted: re-align + traceback |
|---|---|---|---|---|---|---|---|
| base | 0.38 | 0.01 | 0.39 | 0.23 | 0.96 | 0.12 | 0.25 |
| k200 | 0.44 | 0.01 | 0.37 | 0.11 | 0.00 | 0.04 | 0.19 |
| bits256 | 0.41 | 0.01 | 0.43 | 0.20 | 0.12 | 0.09 | 0.31 |
| bits128 | 0.46 | 0.01 | 0.41 | 0.26 | 0.00 | 0.11 | 0.31 |
| precal | 0.00 | 0.00 | 0.00 | 0.24 | 0.97 | 0.15 | 0.00 |
| cal200 | 0.08 | 0.00 | 0.04 | 0.22 | 0.98 | 0.11 | 0.43 |
| combined | 0.07 | 0.00 | 0.03 | 0.10 | 0.00 | 0.03 | 0.21 |

## Accuracy

Alignment accuracy = F1 on 148 held-out SCOPe distant pairs (structure = truth). TM = AlphaFold with MMseqs2 + embedding relatives, 19 proteins; MMseqs2 alone = 0.497.

| config | alignment F1 | relatives accepted | of base's kept | new vs base | mean TM | correct folds | vs base: better / worse | p |
|---|---|---|---|---|---|---|---|---|
| base | 0.530 | 3485 | 100% | 0 | 0.634 | 12/19 | 0 / 0 | 1.00 |
| k200 | 0.530 | 1782 | 51% | 0 | 0.661 | 12/19 | 3 / 2 | 1.00 |
| bits256 | 0.520 | 3805 | 91% | 618 | 0.671 | 13/19 | 4 / 1 | 0.38 |
| bits128 | 0.496 | 3470 | 77% | 771 | 0.672 | 13/19 | 6 / 2 | 0.29 |
| precal | 0.530 | 0 | 0% | 0 | 0.497 | 6/19 | 4 / 12 | 0.08 |
| cal200 | 0.530 | 5482 | 99% | 2032 | 0.620 | 12/19 | 2 / 4 | 0.69 |
| combined | 0.520 | 2965 | 54% | 1074 | 0.677 | 13/19 | 6 / 1 | 0.12 |
