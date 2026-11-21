# Execution plan: exhaustive per-residue embedding search (this PC)

Written 2026-11-20 on the Windows PC (RTX 3060 Ti 8 GB, Ryzen 5 5600X, WSL2 Ubuntu). It is based on the draft in
`exhaustive_residue_search.md`, checked against what is actually on this machine. **Nothing has been run yet.** Every
step marked **ASK** waits for the user's OK.

Runtimes are estimates, not measurements. Step 0 measures the real speed of the aligner. Every later estimate scales
from that number and will be corrected after step 0.

## What is on the PC (checked)

| item | found | note |
|---|---|---|
| SCOPe40 float residues | `phase1/res_emb.npy` (2,802,407 × 640 fp16, 3.59 GB) | 15,177 domains, mean 185 aa |
| SCOPe40 bits | `phase1/res_bits.npy` 640-bit (224 MB), `res_bits256.npy` (90 MB), `res_bits128.npy` (45 MB) | 256/128-bit codes are PCA codes |
| Swiss-Prot bits | `poc/res_bits.npy` 640-bit (16.7 GB, 209,202,719 residues, 576,073 proteins); `res_bits256.npy` (6.7 GB); `res_bits128.npy` (3.3 GB) | no float residues (correct in the draft) |
| Swiss-Prot other | `prot_mean.npy` (737 MB), `pca_res.npz`, `mu.npz`, `win64_256.npz` (window index, 143 MB) | |
| Rankings for the evaluator | `phase1/rankings/*.npz` (26 methods, incl. plmsearch_nopfam, dhr, residue-center-diag) | format: top-1000 per query (`TOP_N`) |
| Scripts | `phase1/04_search.py`, `07_two_stage.py`, `05_evaluate.py`, `poc/emb_align.py`, `poc/45_B_eval.py` all present | 45_B_eval scores the fixed 1000-query subset (`rng(7)`) |
| GPU libs (`fold` env) | torch 2.15 + CUDA 13.2, **triton 3.9**, scikit-learn | no cupy, no faiss (not needed) |
| Tools | mmseqs, jackhmmer (`fold`); USalign (`cf`) | **Foldseek not installed** |
| pLM-BLAST | code `~/tools/plmblast`, env `plmblast`; ProtT5 embeddings done: SCOPe 6.1 GB (15,178 files), 1000 queries 406 MB | search never finished (see below) |
| TEA | **not on the PC**; code location not verified (the web search hit a bot wall) | |
| **RAM** | **WSL capped at 4 GB** (+4 GB swap), 6 CPUs, `.wslconfig` | the draft assumes data fits in RAM; it does not |
| Disk | WSL ext4 830 GB free (virtual); **physical C: 294 GB free** (holds the vhdx); F: 12 TB HDD | enough for everything here |

## Corrections and gaps in the draft

1. **RAM is 4 GB, not 16.**
   - Neither the Swiss-Prot bits (16.7 GB) nor the SCOPe floats (3.6 GB) fit in RAM. All search code must stream the
     database from disk in chunks: database chunk in the outer loop, a batch of queries in the inner loop. That way
     each pass reads the data once for many queries.
   - The current `emb_align.sw_many` is one query at a time, so it needs a multi-query wrapper.
   - If a step gets I/O-bound, raising WSL to ~8 GB is an option, but it needs the user's OK (memory notes: high caps
     starved Windows before).
2. **Parameter leakage already exists.**
   - The embedding-SW bias 0.05 / gap 0.1 were tuned in `phase1/09_align_check.py` on 150 SCOPe40 pairs. Those pairs
     are part of the 15,177 test set.
   - The leak is small (2 numbers), but the rule says no SCOPe. Fix: re-tune bias/gap on Swiss-Prot pairs with AFDB
     structures (the reference pipeline in `poc/48_align_2x2.py`).
   - Report both the old and the re-tuned numbers.
3. **Ranking by raw SW score is biased toward long targets.**
   - The first-FP sensitivity metric needs a length-corrected score.
   - Fix: a z-score per query from decoys (fit score ~ a + b·log L, as in `18_e2e.py`).
   - Report raw and z-ranked, because the choice can move the sensitivity a lot.
4. **Decoys must be re-embedded.**
   - Reversing a protein's stored codes is not the embedding of the reversed protein.
   - Real decoys: embed the reversed sequences of all SCOPe40 domains (2.8 M residues, about 10–15 min GPU, 224 MB of
     bits), and of a 20k-protein Swiss-Prot sample.
   - Shuffled decoys are optional (composition kept, more realistic for low-complexity regions).
   - E = p × N with p from an extreme-value fit to the decoy scores. Empirical p cannot reach 1e-6 with this many
     decoys.
5. **The codebook must be applicable to Swiss-Prot, which has only bits.**
   - Fitting k-means on floats is fine, but Swiss-Prot residues can only be assigned to groups from their 640-bit
     codes.
   - So assignment for both datasets must go through the bits: nearest centroid sign pattern by Hamming distance, or
     k-means fitted directly in bit space.
   - Otherwise SCOPe (floats) and Swiss-Prot (bits) get different alphabets.
6. **Float sample size.**
   - The draft's "~50k Swiss-Prot proteins" is about 18 M residues = 23 GB fp16. That doesn't fit the RAM and is more
     than needed.
   - Use about 5k random proteins = 1.8 M residues (2.3 GB on disk), about 2 min GPU.
7. **Shared memory.**
   - A 256×256 table fits in shared memory only as **int8** (64 KB). The 3060 Ti allows up to 99 KB per block, while
     fp16 would need 128 KB.
   - K = 1024 / 4096 tables (1 MB / 16 MB) and PQ with 4–8 tables (256–512 KB) will not fit. They would live in
     L2/global memory (4 MB L2), which is slower.
   - Plan 2b with int8 K=256. Treat larger K as a speed experiment, not a free win.
8. **A custom kernel is the main engineering risk.** The fast scan needs a fused SW kernel. Triton 3.9 is installed,
   so write it in Triton. The PyTorch version in step 0 stays the correctness reference.
9. **Foldseek is not needed.** MMseqs2 accepts a custom 20-letter substitution matrix (`--sub-mat`) when each group
   is mapped to an amino-acid letter, which is what 2a needs. Leave Foldseek out unless ProstT5+Foldseek is added as a
   baseline (conda install, ~100 MB, plus the ProstT5 weights, ~3 GB). **ASK**
10. **TEA.**
    - Not installed, and I could not confirm a public code link from here.
    - The Mac session should send the bioRxiv DOI / GitHub URL; then we install it (small) or reproduce it in 2a.
    - Our 2a is a reproduction of the idea, not TEA itself. Label it that way.
11. **Low-complexity / TM / coiled-coil enrichment.**
    - No annotation tools here, and SCOPe40 has classes a–g only (f = membrane, no coiled-coil class).
    - Use SCOPe class f for TM, a Python SEG for low complexity, and the Swiss-Prot feature annotations only if the
      user OKs downloading `uniprot_sprot.dat.gz` (~650 MB). **ASK**
12. **Why the pLM-BLAST row is missing.**
    - The ProtT5 embeddings finished, and the search started with `-cpc 70`. The first night attempt crashed because
      the SCOPe embeddings were not written yet (`missing file: scope/0.emb`).
    - The retry ran at about 5 s per item on 31,000 alignment jobs, roughly 43 h projected, so it was stopped.
      `44_B_plmblast_rank.py` was never reached.
    - Options: (a) `-cpc 90` on 200 of the 1000 subset queries, about 3 h CPU (earlier estimate), reported on that
      200-query subset; (b) `-cpc 90` on all 1000, probably 10–15 h. **ASK** (CPU-heavy, 5 workers).
13. Small corrections:
    - There are 20 e2e targets (19 folded + d4gq4a1, which crashes ColabFold).
    - The DHR/big-db fold numbers in the draft match `dhr_bigdb.md`: 13/16, 8/16, 10/16, 16/16.

## Steps

All jobs run with `nice -n 10` and at most 3 CPU threads. They watch `~/folding-data/poc/PAUSE`, leave
`gpu_rest`/`thermal_guard.ps1` untouched, and keep `torch.cuda.set_per_process_memory_fraction(0.6)`. New code goes in
`phase1/` (SCOPe work) and `poc/` (Swiss-Prot work), numbered from 50.

### Step 0: engine + timing smoke test (≈15 min, no OK needed)
- **Write:**
  - `poc/esw.py`: a multi-query exhaustive embedding-SW engine. The database is streamed in length-sorted chunks
    from an mmap; inputs are ±1 fp16 codes or fp16 floats; it keeps the best score per (query, target); traceback only
    on request.
  - It reuses the recurrence of `emb_align.sw` exactly, and a unit test checks it against `emb_align.sw_many` on 50
    random pairs: same scores within 1e-3.
- **Run:** 10 SCOPe queries × all 15,177 domains, 640-bit. Gives seconds/query and GPU memory, and fixes every
  estimate below.
- **Expected:** about 185 × 2.8 M = 5×10⁸ cells per query; in PyTorch, about 0.3–1 s/query.
- **Disk/RAM:** none / <2 GB.

### Step 1: decoys and leak-free parameters (≈30–45 min total)
- `phase1/50_decoys.py`: embed the reversed sequences of the 15,177 SCOPe domains (ESM-2 150M, the same recipe as
  `phase1/02_embed.py`). Writes `phase1/decoy_res_bits{,256,128}.npy` and `decoy_res_emb.npy`. About 15 min GPU; disk
  ~3.9 GB with floats (~0.4 GB without). **ASK** for the float decoys only if the float ceiling is run.
- `poc/51_retune.py`: bias/gap grid search (the same grid as 09) on about 300 Swiss-Prot pairs. Pairs: MMseqs2 hits
  at 20–35% identity with AFDB models, from `sprot.fasta` proteins not in any SCOPe40 superfamily (via MMseqs2 vs
  SCOPe40, e < 1e-3). Reference = US-align (the code from 48).
  - About 600 AFDB downloads, ~150 MB, ~10 min at the throttled rate. Writes `poc/emb_align_params_sprot.json`.

### Step 2: Phase 1, exhaustive SW on SCOPe40
- `phase1/52_exhaustive_sw.py [variant] [--subset 1000|all]`:
  - Variants: `bits640`, `bits256`, `bits128`, `bits2` (2-bit per dim, thresholds = quartiles of the Swiss-Prot
    float sample from step 3a, not SCOPe), `float16` (z-scored "std" floats, its own bias/gap from step 1).
  - Writes `phase1/rankings/exh-<variant>[-z].npz` (top 1000, raw and z-ranked), plus per-query decoy fits and
    false-positive counts at E < 1e-3.
  - E = p × 15,177; p from a Gumbel fit to that query's 15,177 decoy scores.
- Then run `poc/45_B_eval.py`, extended with the new method names, and `phase1/53_fp_report.py` (FP enrichment by
  class f / SEG). Output: `phase1/results/exhaustive_sw.md` + `.json`.
- Runtime per variant, 1000-query subset: queries + decoys means 2 × 1000 × ~0.5 s ≈ 15–30 min for bits. Float is
  similar for compute, but streams 3.6 GB per pass, so it runs 100 queries per pass (~1 min I/O per pass).
  - All 5 variants ≈ 1.5–2.5 h. **ASK** (proposal: bits640 first, ~25 min, then decide).
- All 15,177 queries: about 15× that per variant (≈4–8 h for bits640). **ASK** separately.
- Disk: rankings ~10 MB per variant. RAM: <3 GB (chunked).

### Step 3: Phase 2, codebook + two-stage
- **3a. `poc/54_float_sample.py`:** re-embed 5k random Swiss-Prot proteins (1.8 M residues). Writes
  `poc/sample_res_emb.npy` (2.3 GB) and their bits. About 3 min GPU. No OK needed (small).
- **3b. `poc/55_codebooks.py`:**
  - GPU k-means (torch) for K = 20, 256, 1024, 4096 on the float sample. Then a centroid → 640-bit pattern for
    assignment from bits, and the agreement of float vs bit assignment.
  - PQ: M = 4 and 8 sub-codebooks of 256 on 160/80-bit slices.
  - Score tables: (a) centroid dot product scaled to int8; (b) log-odds from Swiss-Prot homolog residue pairs (the
    MMseqs2 alignments of step 1's leak-free pairs, extended to ~20k pairs with no structures needed) vs random pairs.
  - About 10–20 min GPU; files <50 MB.
- **3c. `poc/56_assign.py`:**
  - Assign group IDs to all residues: SCOPe (2.8 M, seconds) and Swiss-Prot (209 M, one 16.7 GB streaming pass, about
    10–15 min).
  - Writes `poc/res_cb{K}.npy` (1 B/residue for K ≤ 256: 209 MB; 2 B for 1024/4096: 418 MB; PQ: 4–8 B, 0.8–1.7 GB).
- **3d. `phase1/57_mmseqs_alphabet.py` (2a, TEA-like):**
  - Writes SCOPe as K=20 letter strings plus a substitution matrix file, then runs MMseqs2 (`-s 7.5`, `--sub-mat`) →
    ranking `cb20-mmseqs`.
  - Minutes. If TEA's code is available, also run `tea-*` the same way.
- **3e. `poc/esw_lookup.py` (Triton kernel) + `phase1/58_codebook_sw.py` (2b/2c):**
  - SW where each cell's score is a table lookup (int8, K=256 in shared memory; PQ = the sum of M lookups).
  - Validated against an equivalent PyTorch gather version on 50 pairs.
  - Then two-stage: codebook SW over all, keep the top 1% / 5%, rescore with 640-bit SW (step 0 engine). Rankings
    `cb256-2s1`, `pq8-2s5`, etc.
  - Outputs: recall of the step 2 exh-bits640 hits (all, superfamily-level, fold-level), sensitivity via 45_B_eval,
    s/query, speedup.
  - Kernel work: about 1 day of coding/debugging. SCOPe runs: each ≤10 min on the 1000-query subset.
  - The optional ungapped-diagonal prefilter is one flag in the same kernel.
  - Output: `phase1/results/codebook_twostage.md`.

### Step 4: Phase 3, Swiss-Prot MSAs for the 20 e2e targets
- `poc/59_exh_sprot.py --method exh640|twostage|window` for the 20 targets (`e2e/queries.fasta`, minus the warm-up):
  - **exh640:** 20 queries × 209 M residues ≈ 3×10¹⁰ cells each. At step 0's speed this is roughly 15–30 s/query of
    compute. All 20 run in one streaming pass (16.7 GB read, ~1 min), ≈10–20 min total.
    **ASK** if step 0 puts it over 30 min.
  - **twostage:** codebook scan over Swiss-Prot `res_cb256` (209 MB, fits in RAM) + 640-bit rescoring of the top 1%
    (5.8k proteins). Expected seconds/query.
  - **window:** the existing index (the `18_e2e.py` code).
  - Acceptance for all three: the same calibrated chance cutoff as the 2x2 (200 random proteins, Gumbel, 1/2000),
    plus a DB-scale E-value column from 20k-protein Swiss-Prot decoys (`poc/60_sprot_decoys.py`, ~5 min GPU, 0.8 GB
    of bits).
  - Outputs: `~/folding-data/poc/msa_exh/<method>/<target>.a3m`, `poc/results/exh_sprot.md` (hits, Neff80, s/query,
    RAM/disk, and the window-index misses that exhaustive recovers: the MMseqs2 relatives of d1cr5a1/d2dzqa1/d1xg0a_
    are the known test cases).
  - Depth-matched control: subsample the `binplm_emb` 2x2 MSAs to each target's exhaustive depth (random rows, seed 0)
    → `msa_exh/depth_ctrl/`.
  - No folding here: ic-express later.
- 2-bit Swiss-Prot (full re-embed, ~1.7 h GPU, ~33 GB): only if step 2 shows bits2 clearly beats bits640. **ASK**

### Step 5: optional baselines (each **ASK**)
- pLM-BLAST row: `-cpc 90` on 200 subset queries (~3 h CPU) or on all 1000 (~10–15 h). Then `44_B_plmblast_rank.py`
  and `45_B_eval.py`.
- TEA itself (once the URL is known), and ProstT5+Foldseek (~3 GB weights).

## Approvals needed, in order

| # | what | why |
|---|---|---|
| 1 | step 2, bits640 on the 1000 subset (~25 min) then the other variants (~1.5–2.5 h) | GPU over 30 min |
| 2 | step 2, all 15,177 queries (~4–8 h per variant) | long |
| 3 | float decoys (+3.5 GB), only for the float ceiling | disk/GPU |
| 4 | step 4 exhaustive Swiss-Prot, if step 0 puts it over 30 min | long |
| 5 | 2-bit Swiss-Prot re-embed (1.7 h, 33 GB) | re-embed |
| 6 | pLM-BLAST (3 h or 10–15 h CPU), uniprot_sprot.dat (650 MB), Foldseek/ProstT5 (3 GB) | CPU / downloads |
| – | WSL RAM above 4 GB, only if streaming proves too slow | affects Windows |

Steps 0, 1 (bits decoys, retune), 3a–3d and the SCOPe runs of 3e are each under 30 min, so they run without asking
once the plan is approved.

## Decision criteria (from the draft, sharpened)
- **Strong:**
  - exh-bits640 (leak-free params, z-ranked) has a superfamily sensitivity above **0.559** on the 1000 subset,
    beating PLMSearch;
  - **and** two-stage keeps ≥95% of the exhaustive superfamily-level hits at ≥10× speed.
- **Useful:** two-stage ≈ exhaustive at much lower cost. The story becomes storage/speed plus alignment quality
  (align_2x2.md: embedding-SW F1 0.85 vs 0.81 on MMseqs2 hits, 0.62 vs 0.24 on BinPLM hits).
- **Null:** exhaustive ≈ 0.528 (residue-center-diag). Then full SW adds nothing at the ranking level, and the next
  lever is the embedding itself: ESM-2 650M, or learned binary codes.
- Also check: if raw-ranked and z-ranked differ by more than ~0.03, the length-correction choice is itself a finding
  and must be reported.
