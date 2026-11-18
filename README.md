# binary-plm-msa-search

Proof of concept: can **binary-quantized protein language model embeddings** (ESM-2) replace or improve the
sequence homology search (JackHMMER / HHblits / MMseqs2) that builds the multiple sequence alignments (MSAs)
AlphaFold needs?

The method is called **BinPLM** in the code and reports.

## Method

1. Embed every database protein with ESM-2 150M (per-residue vectors, 640 dims).
2. Keep only the sign of each value: 640 bits (80 bytes) per residue, plus a 256-bit PCA-projected version
   used for alignment.
3. **Window index**: average the residue bits over 64-residue windows, project to 256 bits, and store 32-byte
   codes (~108 bytes per protein on average). Search is Hamming distance with `bitwise_count`.
4. Take the top 200 candidate proteins, align them to the query with a Smith-Waterman alignment on the
   binary residue embeddings, and keep hits whose score beats a calibrated chance cutoff (fitted extreme-value
   tail on 200 random proteins, about 1 false hit per 2,000).
5. Write the accepted alignments as an a3m MSA and fold with ColabFold (AlphaFold2).

All runs are on one desktop (Ryzen 5 5600X, 16 GB RAM, RTX 3060 Ti 8 GB) under WSL2 Ubuntu.

## Repository layout

| folder | contents |
|---|---|
| `setup/` | conda environment and ColabFold install scripts |
| `phase1/` | SCOPe 2.08 remote-homology benchmark: embeddings, MMseqs2/HMMER baselines, two-stage search |
| `poc/` | Swiss-Prot pipeline, index tests, alignment, end-to-end folding, ESM Atlas tests, DHR / PLMSearch / pLM-BLAST comparisons, and the shallow-MSA test |
| `poc/results/`, `phase1/results/` | Markdown reports and JSON results for every experiment |

Scripts are numbered in the order they were run. Most `run_*.sh` files are wrappers that activate the right
conda env (`fold` for ESM/PyTorch, `cf` for ColabFold) and log to `poc/results/`.

## Main results

### 1. End-to-end folding, Swiss-Prot only (`poc/results/e2e.md`)

19 fresh SCOPe domains that JackHMMER and MMseqs2 find few relatives for. All pipelines search Swiss-Prot
(576k proteins). ColabFold, 1 model, 3 recycles, no templates.

| pipeline | sequence -> MSA (median s) | mean TM | correct folds (TM > 0.5) |
|---|---|---|---|
| JackHMMER | 0.61 | 0.590 | 9/19 |
| MMseqs2 (ColabFold settings) | 5.90 | 0.669 | 12/19 |
| **BinPLM** | 1.80 | **0.756** | **15/19** |
| MMseqs2 + BinPLM | 6.47 | 0.767 | 15/19 |

### 2. Remote-homology ranking on SCOPe (`poc/results/B_benchmark.md`)

Superfamily-level sensitivity on a fixed 1,000-query subset.

| method | sensitivity | bytes / protein |
|---|---|---|
| PLMSearch (ESM-1b, no Pfam prefilter) | 0.559 | 5,120 |
| BinPLM, residue-level | 0.527 | 14,772 |
| DHR | 0.499 | 960 |
| **BinPLM, window index** | 0.462 | **108** |
| BinPLM, protein-level 256-bit | 0.413 | 32 |
| JackHMMER | 0.354 | - |
| MMseqs2 (3 iterations) | 0.298 | - |

### 3. Adding rows on top of big-database MSAs (`poc/results/A_dark_avail.md`)

MSAs from the ColabFold server (UniRef + environmental databases), with BinPLM Swiss-Prot rows or ESM Atlas
rows appended. 34 proteins, 3 seeds each, TM-score averaged over seeds.

| group | proteins | big-db alone | + BinPLM rows | + Atlas rows |
|---|---|---|---|---|
| shallow (Neff < 30) | 7 | 0.663 | 0.593 | 0.648 |
| medium (Neff 30-100) | 3 | 0.801 | 0.601 | 0.788 |
| deep (Neff >= 100) | 24 | 0.881 | 0.888 | 0.892 |

None of these differences are statistically significant at this sample size.

## Conclusions so far

- Against a small database (Swiss-Prot), BinPLM finds distant relatives that MMseqs2 and JackHMMER miss, and
  the resulting MSAs fold clearly better.
- The window index is very compact (~108 bytes per protein, about 47x smaller than PLMSearch) at a moderate
  cost in sensitivity.
- PLMSearch is more accurate as a ranker.
- Appending BinPLM rows to an already good big-database MSA does not help. On shallow MSAs it often hurts:
  the added distant hits (15-25% identity) seem to dilute a small but accurate alignment.
- ESM Atlas rows were roughly neutral, with one clear rescue (d2dlba1, TM 0.31 -> 0.51).

## Related approaches

| approach | examples | pros | cons |
|---|---|---|---|
| faster classic search | MMseqs2-GPU (Nature Methods 2025) | large speedups, same trusted method | still misses remote homologs |
| larger databases | ColabFold env DB, BFD, MGnify, ESM Atlas | most reliable way to deepen MSAs | terabytes of storage or a public server |
| pLM embedding retrieval | DHR, PLMSearch, knnProtT5, pLM-BLAST, Johnson et al. (eLife 2024) | finds twilight-zone homologs | extra distant rows can hurt folding; per-residue methods are slow |
| generated MSAs | MSAGPT, EvoGen, PLAME | aimed at orphan proteins | risk of hallucinated sequences, heavy models |
| MSA-free folding | ESMFold, OmegaFold | no search at all | less accurate than AlphaFold with a good MSA |

Johnson et al., "Sensitive remote homology search by local alignment of small positional embeddings from
protein language models" (eLife 2024) is the closest prior work to the compression idea here.

## Open next steps

- Finish the shallow-MSA test on all 200 screened proteins (about 35-40 shallow ones expected).
- A stricter chance cutoff when adding rows to an existing MSA (e.g. 1 in 20,000).
- Build MSAs from PLMSearch hits with the same alignment and cutoff, to see whether its better ranking
  translates into better folds (needs ESM-1b vectors for all of Swiss-Prot, ~5 h GPU).
- pLM-BLAST on a 200-query subset with a stricter prefilter.

## Data and tools

- SCOPe 2.08 (astral 40% domains), UniProtKB/Swiss-Prot
- ESM-2 and ESM-1b (Meta, MIT license); ESM Metagenomic Atlas (Meta, CC BY 4.0)
- ColabFold / AlphaFold2, MMseqs2, HMMER, US-align
- DHR, PLMSearch, pLM-BLAST (used as published, for comparison)

Large data files (embeddings, indexes, MSAs, structures) are not included; scripts regenerate them under
`~/folding-data/`.
