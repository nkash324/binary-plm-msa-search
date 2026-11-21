# Plan: exhaustive per-residue embedding search (no window index)

*Draft written in the Mac session (folding-00), 2026-11-20. Saved here without changes. For the checked and corrected
version that matches this PC's data, see `exhaustive_residue_search_exec.md`.*

## Idea
Drop the window index. Align the query against EVERY database protein with embedding Smith-Waterman on per-residue
binary codes. Then make that fast with a cheap exhaustive first pass (codebook "letters"), followed by full 640-bit
rescoring of the top hits.

## Already known (from this repo)
SCOPe 2.08 40% (15,177 domains), superfamily sensitivity to first FP (phase1/results/results.md,
poc/results/B_benchmark.md):
- PLMSearch 0.559
- BinPLM residue-center-diag 0.528 (all residues, but diagonal-vote scoring, not SW)
- DHR 0.499
- window-64-256 0.462
- JackHMMER 0.354
- MMseqs2 0.298

Notes:
- Not yet tested: exhaustive embedding SW against every protein.
- The pLM-BLAST row is missing from B_benchmark.md. Find out why, and fill it in if possible.
- Folding (dhr_bigdb.md): BinPLM/Swiss-Prot gets 13/16 correct, DHR 8/16, MMseqs2 10/16. Big-db ColabFold gets 16/16,
  and adding BinPLM rows to it changes nothing.

## Rules
- Keep the GPU throttles (gpu_rest, PAUSE, thermal_guard.ps1) as they are. Run jobs with nice.
- Ask before anything over ~30 min or any large download.
- Branch exhaustive-search, no push.
- Reuse the existing evaluator so all numbers are comparable.
- Leakage: the codebook, score tables and calibration are fit on Swiss-Prot/random proteins, NEVER on the SCOPe test
  domains.
- Report back to folding-00 after each phase.

## Existing embeddings
- SCOPe40 (~/folding-data/phase1): res_emb.npy (float16 per residue, ~3.6 GB), res_bits.npy (640-bit), offsets.npy,
  prot_emb.npy.
  - Every Phase 1 variant can be derived from these, with no re-embed.
- Swiss-Prot (~/folding-data/poc): res_bits.npy (640-bit only, ~17 GB), prot_mean.npy, offsets.npy. No float
  per-residue vectors.
  - The codebook can be fit on the bits, or on floats from re-embedding a ~50k sample.
  - A 2-bit Swiss-Prot needs a full re-embed (~1.7 h, ~34 GB). Ask first. Never store float for all of Swiss-Prot
    (~270 GB).

## Phase 1: exhaustive embedding SW on SCOPe40 (the reference)
1. GPU embedding SW, one query against all domains, batched.
   - Same scoring and gaps as poc/emb_align.py.
   - Residue-pair scores via a matmul of ±1 codes (int8/fp16), then a batched SW pass.
2. Variants: 640-bit (main), 256-bit, 128-bit, 2-bit, fp16 float (the ceiling).
3. Run the 1,000-query subset first. Ask before running all 15,177.
4. Metrics: family/superfamily/fold sensitivity, remote recall@1000, s/query, bytes/residue.
5. False homologs:
   - E-values at database scale from reversed/shuffled decoys (E = p × N).
   - False positives per query at E < 1e-3.
   - Whether false hits are enriched in TM/coiled-coil/low-complexity regions.

Output: phase1/results/exhaustive_sw.md (+ json).

## Phase 2: make it fast (codebook + two-stage)
How the fast scan works:
- The vectors are used only to build the codebook and the score table.
- During the scan, each residue is a group ID, and a score is a table lookup (BLOSUM-like).
- Only the top hits get full 640-bit SW.

Prior art (don't present as new):
- TEA (bioRxiv 2025, "Rewriting protein alphabets with language models"): PLM embeddings → 20-letter alphabet → MMseqs2.
- ProstT5 + Foldseek.

Possibly new: more detail than 20 letters (256 groups, or product quantization), plus 640-bit SW rescoring and
db-scale E-values, aimed at MSAs for folding.

Steps:
- 2a. 20-group codebook through MMseqs2/Foldseek machinery (essentially reproducing TEA). Also run TEA itself if
  available.
- 2b. 256 groups: the 64 KB score table fits in GPU shared memory. Use your own GPU SW.
- 2c. Product quantization: split the 640 dims into 4–8 parts, with a 256-group codebook each. Score = sum of
  lookups, at 4–8 B/residue.
- Codebooks: k-means on 1–2 M Swiss-Prot residues, K = 20/256/1024/4096.
- Score table, two versions: (a) centroid dot product, or (b) log-odds from Swiss-Prot homolog vs random pairs.
- Two-stage: codebook SW over everything, keep the top 1%/5%, rescore with 640-bit SW.
- Report:
  - Recall of the Phase 1 640-bit hits, overall and for superfamily/fold-level hits.
  - Sensitivity.
  - s/query.
  - Speedup.
- Optional: an ungapped diagonal filter before the gapped SW.

Output: phase1/results/codebook_twostage.md.

## Phase 3: Swiss-Prot scale, MSAs for the 20 e2e targets
1. Search Swiss-Prot with three methods:
   - exhaustive 640-bit SW (batch all 20 queries per pass)
   - two-stage
   - current window index
2. Report:
   - hits
   - Neff
   - s/query
   - RAM/disk
   - window-index misses recovered by the exhaustive search
3. Write the MSAs to ~/folding-data/poc/msa_exh/<method>/<target>.a3m. Folding happens later on the Illinois cluster
   (ic-express), together with a depth-matched control from the 2x2 study.

## Decision criteria
- Strong: exhaustive SW > 0.559 superfamily (beats PLMSearch), AND two-stage keeps ≥95% of its hits at ≥10× speed.
- Useful: two-stage ≈ exhaustive at much lower cost. The claim then becomes storage/speed plus alignment quality
  (align_2x2.md).
- If exhaustive ≈ residue-center-diag (0.528): full SW adds nothing, and the gap is in the embeddings themselves.
  Next: try ESM-2 650M, or learned binary codes.

## Later (cluster, needs approval)
- UniRef50 embedding on ic-express (~61 GPU-h, ~8 resumable jobs). Ask the PI first.
- Exhaustive/two-stage UniRef50 search, with data on node-local SSD and queries batched.
