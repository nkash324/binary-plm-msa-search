# Alignment check: distant relatives (same superfamily, different family), SCOPe

Truth = structural superposition (US-align, residues < 5 A). 148 held-out pairs.

| aligner | correct residue pairs found (sensitivity) | pairs that are correct (precision) | F1 |
|---|---|---|---|
| BLOSUM62 Smith-Waterman (letters) | 0.345 | 0.415 | 0.358 |
| Embedding Smith-Waterman, 256-bit codes (bias 0.05, gap 0.1) | 0.554 | 0.498 | 0.520 |

Verdict: GO - embedding aligner clearly better
