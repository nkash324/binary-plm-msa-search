# What is still needed to test the binary-embedding (BinPLM) method

Status 2026-11-20 (branch exhaustive-search):
- Alignment quality: done (2x2 and leak-free retune).
- Remote homology on SCOPe: done (1000-query subset; all 15,177 queries running now).
- Swiss-Prot MSAs for 20 targets: done.

Nothing below has been run yet.

Only folding (T1) answers the real question: do these MSAs give better structures? Everything else is a supporting check, or something that makes the result believable at scale.

## T1. Folding test (the real endpoint): the most important gap
- **What:** fold every MSA we have with ColabFold (AF2, 3 recycles, model 1–5, no templates) on ic-express.
  - The 4 cells of the 2x2: mm_blosum, mm_emb, binplm_blosum, binplm_emb.
  - The 4 Swiss-Prot MSA sets: exh640, twostage, window, depth_ctrl.
  - Two references: the standard ColabFold MSA (UniRef30 + envDB), and no MSA (single sequence).
- **Measure:**
  - TM-score and lDDT to the PDB/SCOPe structure.
  - pLDDT.
  - Paired difference per target vs the MMseqs2 MSA, with a Wilcoxon test.
- **Controls:**
  - depth_ctrl separates "more rows" from "better rows".
  - Add a "random rows" control at the same depth, and an MMseqs2 ∪ BinPLM union MSA.
- **Needs:** cluster access (folding-00 / you), and a copy of `~/folding-data/poc/msa_2x2` and `msa_exh` (a few MB).
- **Cost:** about 200 predictions (20 targets × 10 MSAs) × ~1–3 min each ≈ 4–10 GPU-hours on the cluster. It costs nothing on the PC.
- **Pass:** binplm_emb or exh640 has a median ΔTM > 0 vs MMseqs2 on targets where MMseqs2 is shallow (Neff < 30), and no loss on deep targets.

## T2. More, and cleaner, targets
- **Problem 1:** 20 targets is too few to make the statistical tests in T1 meaningful.
- **Problem 2:** SCOPe domains are in ESM-2's training data (UniRef50, 2021), which is leakage on the pLM side.
- **What:** add about 50–100 targets released after ESM-2's training cutoff, from CASP15/16 and CAMEO (2023+). Pick them by difficulty (MMseqs2 Neff < 30 is where the method could matter).
- **Cost:** small downloads (PDB files), ~1 h of PC time to embed and search. Folding goes on the cluster.

## T3. Fix the E-value calibration (known problem)
- **Problem:** E-values are about 35× optimistic. On held-out reversed decoys, E < 1 fires 0.60 times per target against 0.017 expected.
- **Try, all fitted on Swiss-Prot only (leak rule):**
  1. Fit the Gumbel to deeper tails (q99.9) on more decoys (100k instead of 10k).
  2. Use different decoys: windowed-shuffled sequences, or real non-homologs. For real non-homologs, use Swiss-Prot pairs in different Pfam clans, which needs `uniprot_sprot.dat` (~600 MB, needs your OK).
  3. Use an empirical p-value: the rank among decoys, with no fitted curve.
- **Check:** decoy hits at E < 1 ≈ expected, and the wrong-fold rate on SCOPe.
- **Cost:** 30–60 min of PC time.

## T4. False homologs in α/β proteins (class c)
- **Problem:** 69% of wrong-fold hits are in class c, probably βαβ look-alike motifs.
- **Test:**
  - A per-query "motif-only" filter: require alignment coverage of at least 50% of the shorter protein, or a minimum alignment length.
  - Optionally a low-complexity / repeat mask on embedding "letters".
- **Measure:** the SCOPe wrong-fold hit rate and the superfamily sensitivity trade-off.
- **Cost:** about 15 min. It only re-analyses the checkpoints we already have.

## T5. A prefilter that works (the codebook two-stage failed)
- **Problem:** exhaustive search is 3.5 s/target on 576k proteins and 80 B/residue. On UniRef50 (~60M sequences, ~17 B residues) that is about 6 min/target and 1.4 TB, which can't run on the PC.
- **Candidates:**
  - (a) An ungapped diagonal prefilter on the binary codes (BLAST-like double hits).
  - (b) A better window index, with a larger top-K (1k–5k) and multiple windows per query.
  - (c) The TEA letters with MMseqs2/STEAM as the prefilter, then 640-bit SW to rescore.
- **Measure:** recall of exhaustive hits (target ≥ 95%), and s/target.
- **Data:** the SCOPe and Swiss-Prot checkpoints already contain the "truth".
- **Cost:** 1–3 h of dev time plus short runs.

## T6. Scale test on a large database
- **What:** embed a UniRef50 subset (for example 5M sequences ≈ 1.5 B residues) with ESM-2 150M, and run the best prefilter from T5 plus rescoring.
- **Measure:**
  - Extra rows and Neff vs Swiss-Prot.
  - Time per target.
  - Disk use (binary codes ≈ 120 GB at 80 B/residue; codebook/TEA ≈ 3 GB).
- **Cost:**
  - A ~3–10 GB download.
  - Roughly 1–2 days of GPU embedding on the 3060 Ti, or a few hours on the cluster.
  - Both need your OK (large download, long job).

## T7. Baselines the paper will need
| baseline | status | needs |
|---|---|---|
| MMseqs2 (s 7.5, 3-iter) | done | - |
| jackhmmer | done | - |
| PLMSearch | done (0.559 superfamily) | - |
| pLM-BLAST | partial (1000 queries) | ~1 h |
| **TEA + STEAM** (closest competitor) | blocked: install not allowed | your permission for a `tea` conda env + ESM-2 650M (~2.5 GB) |
| Foldseek + ProstT5 (structure alphabet from sequence) | not run | ~3 GB weights, ~1 h |
| HHblits on UniRef30 | not run | ~50 GB database |

## Order I would run them in
1. T4 and T3, now on the PC: cheap, and they fix known weaknesses before anything is folded.
2. T1 (cluster), with the T2 targets added if you can get them in time. This is the deciding experiment.
3. T7: TEA/STEAM first (needs the install permission), then Foldseek/ProstT5.
4. T5, then T6, only if T1 shows the MSAs help.

## Approvals needed
- [ ] T1 cluster time (ic-express) and who runs it (folding-00?)
- [ ] TEA/STEAM install (conda env + pip + ~2.5 GB model)
- [ ] `uniprot_sprot.dat` (~600 MB) for the Pfam-clan decoys in T3
- [ ] ProstT5/Foldseek weights (~3 GB)
- [ ] T6 UniRef50 subset download and the multi-hour embedding
- [ ] GitHub push of branch `exhaustive-search`
