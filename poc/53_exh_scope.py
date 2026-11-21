"""Step 2-lite: exhaustive embedding Smith-Waterman on SCOPe40, 640-bit codes, the fixed 1000-query subset.

Every query is aligned against all 15,177 domains AND all 15,177 reversed-sequence decoys (step 1a), with the
leak-free bias/gap (step 1b; PARAMS=old uses the old SCOPe-tuned ones). Codes = res_bits_sp (Swiss-Prot mean).
Rankings (phase1/rankings, scored by poc/45_B_eval.py):
  exh640-raw[-old] : by raw SW score
  exh640-z[-old]   : by length-corrected z (per query: decoy scores ~ a + b*log(length))
E-values: per query, a Gumbel fit to the decoy z-scores; E = p * 15,177. Reports false-fold hits at E < 1e-3.
Resumable (per-batch checkpoints). Writes phase1/results/exhaustive_sw.{md,json}.  Run in the 'fold' env."""
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import esw_triton as et
from common import DATA, PHASE1

torch.cuda.set_per_process_memory_fraction(0.6)
spec = importlib.util.spec_from_file_location("p1common", "/mnt/c/Folding/phase1/common.py")
p1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p1)
meta = p1.load_meta()
fam, sf, fold = meta["fam"], meta["sf"], meta["fold"]
OLD = os.environ.get("PARAMS", "sprot") == "old"
P = json.loads((DATA / ("emb_align_params.json" if OLD else "emb_align_params_sprot.json")).read_text())
ALL = os.environ.get("ALL", "0") == "1"                                   # ALL=1: every domain is a query
BIAS, GAP, TAG = P["bias"], P["gap"], ("-old" if OLD else "") + ("-all" if ALL else "")
off = np.load(PHASE1 / "offsets.npy")
N, lens = len(off) - 1, np.diff(off)
real = np.load(PHASE1 / "res_bits_sp.npy", mmap_mode="r")
dec = np.load(PHASE1 / "decoy_res_bits_sp.npy", mmap_mode="r")
SUB = np.arange(N) if ALL else np.sort(np.random.default_rng(7).choice(N, 1000, replace=False))  # subset as 45_B_eval
CK = DATA / f"exh_scope{TAG}"
CK.mkdir(exist_ok=True)
QB = int(os.environ.get("QBATCH", 50))
log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)
log(f"bias {BIAS} gap {GAP}; {len(SUB)} queries x ({N} domains + {N} decoys)")

t_search = 0.0
for b in range(0, len(SUB), QB):
    f = CK / f"b{b:04d}.npz"
    if f.exists():
        t_search += float(np.load(f)["secs"])
        continue
    qs = SUB[b:b + QB]
    Qs = [np.asarray(real[off[q]:off[q + 1]]) for q in qs]
    t = time.time()
    S = et.search(Qs, real, off, BIAS, GAP, pause=DATA / "PAUSE")
    D = et.search(Qs, dec, off, BIAS, GAP, pause=DATA / "PAUSE")
    secs = time.time() - t
    np.savez(f, S=S.astype(np.float16), D=D.astype(np.float16), secs=secs)
    t_search += secs
    log(f"  queries {b + len(qs)}/{len(SUB)}: {secs:.0f}s for this batch ({secs / len(qs):.2f} s/query incl. decoys)")

A = np.c_[np.ones(N), np.log(lens)]


def rows():                                                               # one checkpoint at a time (RAM)
    for b in range(0, len(SUB), QB):
        z = np.load(CK / f"b{b:04d}.npz")
        S, D = z["S"].astype(np.float32), z["D"].astype(np.float32)
        for i in range(len(S)):
            yield b + i, S[i], D[i]



raw_rank, z_rank, stats = [np.zeros(0, np.int32)] * N, [np.zeros(0, np.int32)] * N, []
EG = 0.5772156649
for k, Sk, Dk in rows():
    q = SUB[k]
    coef, *_ = np.linalg.lstsq(A, Dk, rcond=None)
    sd = np.std(Dk - A @ coef)
    zr, zd = (Sk - A @ coef) / sd, (Dk - A @ coef) / sd
    q90, q99 = np.quantile(zd, [0.9, 0.99])                                # Gumbel fitted to the decoy TAIL
    beta = (q99 - q90) / (np.log(-np.log(0.9)) - np.log(-np.log(0.99)))   # (same recipe as 18_e2e.py)
    mu = q90 + beta * np.log(-np.log(0.9))
    E = N * -np.expm1(-np.exp(-(zr - mu) / beta))
    raw_rank[q] = np.argsort(-Sk, kind="stable")[:1000].astype(np.int32)
    z_rank[q] = np.argsort(-zr, kind="stable")[:1000].astype(np.int32)
    sig = (E < 1e-3) & (np.arange(N) != q)
    stats.append({"sig": int(sig.sum()), "sig_false_fold": int((sig & (fold != fold[q])).sum()),
                  "sig_remote": int((sig & (sf == sf[q]) & (fam != fam[q])).sum()),
                  "decoys_E<1e-3": int((N * -np.expm1(-np.exp(-(zd - mu) / beta)) < 1e-3).sum()),
                  "decoys_E<1": int((N * -np.expm1(-np.exp(-(zd - mu) / beta)) < 1).sum()),
                  "above_best_decoy": int(((zr > zd.max()) & (np.arange(N) != q)).sum()),
                  "above_best_decoy_false_fold": int(((zr > zd.max()) & (fold != fold[q])).sum())})
bpp = 80 * float(lens.mean())
note = f"exhaustive embedding SW, 640 bits, bias {BIAS} gap {GAP}, decoy-calibrated"
p1.save_ranking(f"exh640-raw{TAG}", raw_rank, t_search / 2, bpp, note + " (raw score)")
p1.save_ranking(f"exh640-z{TAG}", z_rank, t_search, bpp, note + " (z-score)")
st = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
st.update({"queries_with_false_fold_hit": int(sum(s["sig_false_fold"] > 0 for s in stats)),
           "s_per_query": t_search / len(SUB), "bias": BIAS, "gap": GAP})
out = Path("/mnt/c/Folding/phase1/results")
(out / f"exhaustive_sw{TAG}.json").write_text(json.dumps(st, indent=1))
L = [f"# Exhaustive embedding SW on SCOPe40 (640 bits, {'all 15,177 queries' if ALL else '1000-query subset'}{', old params' if OLD else ''})", "",
     f"bias {BIAS}, gap {GAP}; every query vs all 15,177 domains + 15,177 reversed-sequence decoys. "
     f"{st['s_per_query']:.2f} s/query including decoys (3060 Ti, fused Triton kernel).", "",
     "Sensitivity numbers: see poc/results/B_benchmark.md (rows exh640-raw / exh640-z).", "",
     "Hits at E < 1e-3 (per query, mean): "
     f"all {st['sig']:.1f}, remote (same superfamily, other family) {st['sig_remote']:.1f}, "
     f"wrong fold {st['sig_false_fold']:.2f}; queries with >= 1 wrong-fold hit: {st['queries_with_false_fold_hit']}/{len(SUB)}; "
     f"decoys passing (expected ~0.001): {st['decoys_E<1e-3']:.3f}; decoys at E < 1 (expected ~1): {st['decoys_E<1']:.2f}.",
     "", f"Hits scoring above the query's best decoy (empirical, no fit): {st['above_best_decoy']:.1f} per query, of which "
     f"wrong fold {st['above_best_decoy_false_fold']:.2f}."]
(out / f"exhaustive_sw{TAG}.md").write_text("\n".join(L) + "\n")
log("\n".join(L))
