"""Step 3: codebook first pass + 640-bit rescoring (two-stage), SCOPe40 1000-query subset.

First pass: Smith-Waterman on group IDs (codebook "letters"); the cell score is a table lookup (54_codebooks.py).
Variants: cb20, cb256, cb1024 (1-2 B/residue), pq4 (4 x 256 groups on 160-bit slices, 4 B/residue).
Per query, first-pass scores are length-corrected (fit over all database scores) and the top 1% / 5% go to the
exhaustive 640-bit SW (Triton). Final ranking = rescored candidates by 640-bit decoy z (as exh640-z), then the
rest by first-pass z. Measures: recall of the exhaustive search's hits, sensitivity (45_B_eval), s/query.
Writes phase1/rankings/2s-*.npz, cb-*.npz and phase1/results/codebook_twostage.{md,json}.  Run in the 'fold' env."""
import importlib.util
import json
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
off = np.load(PHASE1 / "offsets.npy")
N, lens = len(off) - 1, np.diff(off)
SUB = np.sort(np.random.default_rng(7).choice(N, 1000, replace=False))
P = json.loads((DATA / "emb_align_params_sprot.json").read_text())
BIAS, GAP = P["bias"], P["gap"]
CB = np.load(DATA / "codebooks.npz")
real = np.load(PHASE1 / "res_bits_sp.npy", mmap_mode="r")
log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)
A = np.c_[np.ones(N), np.log(lens)]

# exhaustive reference (step 2-lite checkpoints)
CK = DATA / "exh_scope"
S = np.concatenate([np.load(CK / f"b{b:04d}.npz")["S"] for b in range(0, 1000, 50)]).astype(np.float32)
D = np.concatenate([np.load(CK / f"b{b:04d}.npz")["D"] for b in range(0, 1000, 50)]).astype(np.float32)
t_exh = sum(float(np.load(CK / f"b{b:04d}.npz")["secs"]) for b in range(0, 1000, 50)) / 2 / 1000
Z, HIT = np.zeros_like(S), np.zeros(S.shape, bool)
for k, q in enumerate(SUB):
    coef, *_ = np.linalg.lstsq(A, D[k], rcond=None)
    sd = np.std(D[k] - A @ coef)
    Z[k], zd = (S[k] - A @ coef) / sd, (D[k] - A @ coef) / sd
    HIT[k] = Z[k] > zd.max()                                             # exhaustive "hit" = above the best decoy
    HIT[k, q] = False
remote = (sf[None, :] == sf[SUB][:, None]) & (fam[None, :] != fam[SUB][:, None])
log(f"exhaustive reference: {HIT.sum(1).mean():.1f} hits/query ({(HIT & remote).sum(1).mean():.1f} remote)")

variants = {"cb20": ("res_cb20_sp.npy", CB["T20"][None]), "cb256": ("res_cb256_sp.npy", CB["T256"][None]),
            "cb1024": ("res_cb1024_sp.npy", CB["T1024"][None]),
            "pq4": ("res_pq4_sp.npy", np.stack([CB[f"PQT{m}"] for m in range(4)]))}
bpr = {"cb20": 1, "cb256": 1, "cb1024": 2, "pq4": 4}
R = {"exhaustive_s_per_query": t_exh, "bias": BIAS, "gap": GAP}
# Per-variant score scale (leak-free, Swiss-Prot only): match the codebook cell-score distribution on random
# residue pairs to the 640-bit one, so the 640-bit bias/gap mean the same thing in both passes.
import os                                                                                    # noqa: E402
SCALE = os.environ.get("SCALE", "1") == "1"
spb = np.load(DATA / "res_bits.npy", mmap_mode="r")
rr = np.random.default_rng(9)
ia, ib = np.sort(rr.choice(len(spb), 20000, replace=False)), np.sort(rr.choice(len(spb), 20000, replace=False))
xa, xb = (np.unpackbits(np.asarray(spb[i]), axis=1).astype(np.float32) * 2 - 1 for i in (ia, ib))
s640 = (xa * xb).sum(1) / 640
m640, sd640 = float(s640.mean()), float(s640.std())


def cell_stats(v, table):
    ref = {"cb20": "res_cb20.npy", "cb256": "res_cb256.npy", "cb1024": "res_cb1024.npy", "pq4": "res_pq4.npy"}[v]
    g = np.load(DATA / ref, mmap_mode="r")
    a = np.asarray(g[ia]).reshape(len(ia), -1).astype(np.int64)
    b = np.asarray(g[ib]).reshape(len(ib), -1).astype(np.int64)
    sc = sum(table[m][a[:, m], b[:, m]] for m in range(table.shape[0]))
    return float(sc.mean()), float(sc.std())


for v, (fname, table) in variants.items():
    if SCALE:
        mc, sdc = cell_stats(v, table)
        k = sdc / sd640
        vb, vg = mc + (BIAS - m640) * k, GAP * k
    else:
        vb, vg = BIAS, GAP
    R[f"params_{v}"] = {"bias": vb, "gap": vg}
    log(f"{v}: first-pass bias {vb:.4f} gap {vg:.4f}")
    sym = np.load(PHASE1 / fname, mmap_mode="r")
    Qs = [np.asarray(sym[off[q]:off[q + 1]]) for q in SUB]
    et.search_lut(Qs[:2], sym, off, table, vb, vg, targets=np.arange(200))      # compile / warm-up
    torch.cuda.synchronize()
    t = time.time()
    F = np.concatenate([et.search_lut(Qs[b:b + 50], sym, off, table, vb, vg) for b in range(0, 1000, 50)])
    torch.cuda.synchronize()
    t_first = (time.time() - t) / 1000
    only = [np.zeros(0, np.int32)] * N
    for k, q in enumerate(SUB):
        zf = F[k] - A @ np.linalg.lstsq(A, F[k], rcond=None)[0]
        only[q] = np.argsort(-zf, kind="stable")[:1000].astype(np.int32)
    p1.save_ranking(f"cb-{v}", only, t_first * 1000, bpr[v] * lens.mean(), f"codebook SW only ({v})")
    for frac in (0.01, 0.05):
        nc = int(round(frac * N))
        ranks, rec, rec_r = [np.zeros(0, np.int32)] * N, [], []
        t_resc = 0.0
        for k, q in enumerate(SUB):
            coef, *_ = np.linalg.lstsq(A, F[k], rcond=None)
            zf = F[k] - A @ coef
            o = np.argsort(-zf, kind="stable")
            cand = o[:nc]
            if k < 50:                                                   # time the real rescoring on 50 queries
                tt = time.time()
                et.search([np.asarray(real[off[q]:off[q + 1]])], real, off, BIAS, GAP, targets=np.sort(cand))
                torch.cuda.synchronize()
                t_resc += time.time() - tt
            cs = cand[np.argsort(-Z[k][cand], kind="stable")]           # rescored = exhaustive values (identical)
            rest = o[nc:]
            ranks[q] = np.r_[cs, rest][:1000].astype(np.int32)
            h, hr = HIT[k], HIT[k] & remote[k]
            inc = np.zeros(N, bool)
            inc[cand] = True
            if h.any():
                rec.append((h & inc).sum() / h.sum())
            if hr.any():
                rec_r.append((hr & inc).sum() / hr.sum())
        tq = t_first + t_resc / 50
        name = f"2s-{v}-{int(frac * 100)}"
        p1.save_ranking(name, ranks, tq * 1000, bpr[v] * lens.mean(), f"two-stage {v}, top {frac:.0%} rescored")
        R[name] = {"recall_hits": float(np.mean(rec)), "recall_remote_hits": float(np.mean(rec_r)),
                   "first_pass_s": t_first, "rescore_s": t_resc / 50, "s_per_query": tq,
                   "speedup_vs_exhaustive": t_exh / tq, "bytes_per_residue_first_pass": bpr[v]}
        log(f"{name}: recall {R[name]['recall_hits']:.3f} (remote {R[name]['recall_remote_hits']:.3f}), "
            f"{tq * 1000:.1f} ms/query, speed-up {t_exh / tq:.1f}x")
out = Path("/mnt/c/Folding/phase1/results")
(out / "codebook_twostage.json").write_text(json.dumps(R, indent=1))
