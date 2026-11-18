"""Speed-ups to the embedding-alignment step, each tested alone and combined, on the 20 development proteins.
Only the method's own settings change; data is read exactly as before (memory-mapped file + OS cache).

  base      500 candidates, 640-bit residue codes, chance level measured per query (2000 random proteins)
  k200      200 candidates
  bits256   256-bit PCA residue codes  (32 B/aa instead of 80)
  bits128   128-bit PCA residue codes  (16 B/aa)
  precal    chance level pre-computed once from protein lengths (no per-query random alignments)
  combined  200 candidates + smallest code size that kept alignment accuracy + pre-computed chance level

Each config runs twice: COLD (file cache for the residue codes emptied first) and WARM.
Writes ~/folding-data/poc/abl/<config>/<query>.json and results/ablation.{json,md}.
"""
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

import emb_align as ea
from common import DATA, DB_FASTA, Q_FASTA, RESULTS, read_fasta

torch.cuda.set_per_process_memory_fraction(0.6)
sync = torch.cuda.synchronize
ABL = DATA / "abl"
LOG = open(RESULTS / "ablation.log", "a")


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


# ---------------------------------------------------------------- shared data
db_ids, db_seqs = read_fasta(DB_FASTA)
NP = len(db_ids)
off = np.load(DATA / "offsets.npy")
lens = np.diff(off)
q_ids, q_seqs = read_fasta(Q_FASTA)
q_off = np.load(DATA / "q_offsets.npy")
summ = json.loads((RESULTS / "poc_results.json").read_text())["per_query"]
ref_neff = {r["q"]: r["neff"] for r in summ["phmmer_max"]}
hard = sorted(q_ids, key=lambda q: ref_neff[q])[:int(os.environ.get("N_HARD", 20))]
mm_hits = json.loads((DATA / "hits" / "mmseqs_cf.json").read_text())["hits"]
id2idx = {d: i for i, d in enumerate(db_ids)}
null_pool = np.random.default_rng(42).choice(NP, 2000, replace=False)

CODES = {b: np.load(DATA / ("res_bits.npy" if b == 640 else f"res_bits{b}.npy"), mmap_mode="r") for b in (640, 256, 128)}
QCODES = {b: np.load(DATA / ("q_res_bits.npy" if b == 640 else f"q_res_bits{b}.npy")) for b in (640, 256, 128)}
PARAMS = {b: json.loads((DATA / ("emb_align_params.json" if b == 640 else f"emb_align_params_{b}.json")).read_text())
          for b in (640, 256, 128)}

# ---------------------------------------------------------------- window search (identical for every config; done once)
z = np.load(DATA / "win64_256.npz")
wcodes = np.ascontiguousarray(z["codes"]).view(np.uint64)
owner, MU, PM = z["owner"], z["mu"], z["P"]
W, B, TOPW = 64, 256, 2000


def window_search(qi):
    X = np.unpackbits(QCODES[640][q_off[qi]:q_off[qi + 1]], axis=1).astype(np.float32) * 2 - 1
    L = len(X)
    starts = sorted(set(list(range(0, max(1, L - W + 1), W // 4)) + [max(0, L - W)]))
    qc = np.ascontiguousarray(np.packbits((np.stack([X[a:a + W].mean(0) for a in starts]) - MU) @ PM > 0,
                                          axis=1)).view(np.uint64)
    score = np.zeros(NP, np.float32)
    for lo in range(0, len(wcodes), 500_000):
        c = wcodes[lo:lo + 500_000]
        h = np.zeros((len(qc), len(c)), np.uint16)
        for w in range(c.shape[1]):
            h += np.bitwise_count(c[None, :, w] ^ qc[:, None, w])
        i = np.argpartition(h, TOPW, axis=1)[:, :TOPW]
        zz = (B / 2 - np.take_along_axis(h, i, 1).astype(np.float32)) / (np.sqrt(B) / 2)
        for j in range(len(qc)):
            o = np.argsort(-zz[j])
            pp, first = np.unique(owner[lo + i[j][o]], return_index=True)
            score[pp] += np.maximum(zz[j][o][first], 0)
    return score


ranked, search_s = {}, []
for q in hard:
    qi = q_ids.index(q)
    t = time.perf_counter()
    sc = window_search(qi)
    ex = [id2idx[x] for x in mm_hits.get(q, []) if x in id2idx]
    sc[ex] = -1
    c = np.argsort(-sc)[:500]
    ranked[q] = c[sc[c] > 0]
    search_s.append(time.perf_counter() - t)
log(f"window search: {np.mean(search_s):.2f} s/protein (same for every config)")


# ---------------------------------------------------------------- pre-computed chance level (one-time, per code size)
def fit_precal(bits):
    fn = DATA / f"precal_{bits}.npz"
    if fn.exists():
        return dict(np.load(fn)), 0.0
    t0 = time.time()
    r = np.random.default_rng(7)
    pool = np.setdiff1d(np.arange(NP), null_pool)
    pq = r.choice(pool[(lens[pool] >= 60) & (lens[pool] <= 400)], 100, replace=False)
    tg = r.choice(pool, 1000, replace=False)
    bias, gap = PARAMS[bits]["bias"], PARAMS[bits]["gap"]
    C = CODES[bits]
    T = [ea.pm1(np.array(C[off[p]:off[p + 1]])) for p in tg]
    S = np.stack([ea.sw_many(ea.pm1(np.array(C[off[p]:off[p + 1]])), T, bias, gap)[0] for p in pq])
    lq, lt = np.log(lens[pq])[:, None] * np.ones((1, len(tg))), np.ones((len(pq), 1)) * np.log(lens[tg])[None]
    F = np.stack([np.ones(S.size), lq.ravel(), lt.ravel(), (lq * lt).ravel()], 1)
    coef, *_ = np.linalg.lstsq(F, S.ravel(), rcond=None)
    res = (S.ravel() - F @ coef).reshape(S.shape)
    sdq = res.std(1)
    sd_coef = np.polyfit(np.log(lens[pq]), np.log(sdq), 1)
    zz = res / np.exp(np.polyval(sd_coef, np.log(lens[pq])))[:, None]
    thr = float(np.quantile(zz.ravel(), 1 - 1 / 2000))          # same strictness as "beat the best of 2000"
    out = {"coef": coef, "sd_coef": sd_coef, "thr": np.array(thr)}
    np.savez(fn, **out)
    return out, time.time() - t0


# ---------------------------------------------------------------- one config
def evict(arr):
    fd = os.open(arr.filename, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def run(cfg, K, bits, cal, write):
    C, QC = CODES[bits], QCODES[bits]
    bias, gap = PARAMS[bits]["bias"], PARAMS[bits]["gap"]
    pc = fit_precal(bits)[0] if cal == "pre" else None
    tm = defaultdict(float)
    out = {}

    def tick(k, t):
        sync()
        tm[k] += time.perf_counter() - t
        return time.perf_counter()

    def load(ps, k):
        t = time.perf_counter()
        raw = [np.array(C[off[p]:off[p + 1]]) for p in ps]       # the actual disk read (same mmap as before)
        t = tick(f"{k}: read from disk", t)
        g = [ea.pm1(r) for r in raw]
        tick(f"{k}: send to GPU", t)
        return g

    for q in hard:
        qi = q_ids.index(q)
        Q = ea.pm1(QC[q_off[qi]:q_off[qi + 1]])
        L = len(Q)
        cand = ranked[q][:K]
        if cal in ("per_query", "small"):
            exclude = set(id2idx[x] for x in mm_hits.get(q, []) if x in id2idx)
            cs_ = set(cand.tolist())
            null = np.array([p for p in null_pool if p not in exclude and p not in cs_])
            if cal == "small":
                null = null[:200]
            Tn = load(null, "1 chance level")
            t = time.perf_counter()
            ns, _ = ea.sw_many(Q, Tn, bias, gap)
            tick("1 chance level: align", t)
            A = np.c_[np.ones(len(null)), np.log(lens[null])]
            coef, *_ = np.linalg.lstsq(A, ns, rcond=None)
            sd = np.std(ns - A @ coef)
            zf = lambda s, ps: (s - np.c_[np.ones(len(ps)), np.log(lens[ps])] @ coef) / sd
            nz = zf(ns, null)
            if cal == "small":            # Gumbel tail through the 90th/99th percentiles -> estimate the best of 2000
                q90, q99 = np.quantile(nz, [0.9, 0.99])
                beta = (q99 - q90) / (np.log(-np.log(0.9)) - np.log(-np.log(0.99)))
                thr = float(q90 + beta * (-np.log(-np.log(1 - 1 / 2000)) + np.log(-np.log(0.9))))
            else:
                thr = float(nz.max())
        else:
            lq = np.log(L)
            zf = lambda s, ps: (s - np.c_[np.ones(len(ps)), np.full(len(ps), lq), np.log(lens[ps]),
                                          lq * np.log(lens[ps])] @ pc["coef"]) / np.exp(np.polyval(pc["sd_coef"], lq))
            thr = float(pc["thr"])
        Tc = load(cand, "2 candidates")
        t = time.perf_counter()
        cs, _ = ea.sw_many(Q, Tc, bias, gap)
        tick("2 candidates: align", t)
        keep = zf(cs, cand) > thr
        acc = cand[keep]
        t = time.perf_counter()
        rows = []
        if len(acc):
            _, pr = ea.sw_many(Q, [Tc[k] for k in np.flatnonzero(keep)], bias, gap, keep=True)
            for p, pairs in zip(acc, pr):
                r = ["-"] * L
                for i, j in pairs:
                    r[i] = db_seqs[p][j]
                rows.append("".join(r))
        tick("3 accepted: re-align + traceback", t)
        out[q] = {"accepted": [db_ids[p] for p in acc], "rows_emb": rows}
    if write:
        (ABL / cfg).mkdir(parents=True, exist_ok=True)
        for q, rec in out.items():
            (ABL / cfg / f"{q}.json").write_text(json.dumps(rec))
    return {k: v / len(hard) for k, v in tm.items()}, out


# ---------------------------------------------------------------- choose code size for 'combined'
f1_640 = PARAMS[640]["embedding"]["f1"]
best_bits = min([b for b in (640, 256, 128) if PARAMS[b]["embedding"]["f1"] >= f1_640 - 0.02])
log("alignment accuracy (F1 on held-out SCOPe pairs): " +
    ", ".join(f"{b}-bit {PARAMS[b]['embedding']['f1']:.3f}" for b in (640, 256, 128)) + f" -> combined uses {best_bits}-bit")

CONFIGS = {"base": (500, 640, "per_query"), "k200": (200, 640, "per_query"), "bits256": (500, 256, "per_query"),
           "bits128": (500, 128, "per_query"), "precal": (500, 640, "pre"), "cal200": (500, 640, "small"),
           "combined": (200, best_bits, "small")}
ONLY = [c for c in os.environ.get("ONLY", "").split(",") if c]
RUN = {c: v for c, v in CONFIGS.items() if not ONLY or c in ONLY}
setup = {b: fit_precal(b)[1] for b in {v[1] for v in RUN.values() if v[2] == "pre"}}
log("pre-computed chance level, one-time setup: " + ", ".join(f"{b}-bit {s:.0f}s" for b, s in setup.items()))

# warm-up (GPU start-up) so the first config isn't penalised
run("warmup", 50, 640, "per_query", write=False)

old = json.loads((RESULTS / "ablation.json").read_text()) if (RESULTS / "ablation.json").exists() else {}
results, accepted = {k: v for k, v in old.items() if k in CONFIGS and k not in RUN}, {}
for cfg in CONFIGS:
    if cfg not in RUN and (ABL / cfg).exists():
        accepted[cfg] = {q: json.loads((ABL / cfg / f"{q}.json").read_text()) for q in hard}
for cfg, (K, bits, cal) in RUN.items():
    evict(CODES[bits])
    cold, _ = run(cfg, K, bits, cal, write=False)
    warm, out = run(cfg, K, bits, cal, write=True)
    accepted[cfg] = out
    for a in CODES.values():                                  # don't let WSL hoard file cache
        evict(a)
    results[cfg] = {"K": K, "bits": bits, "calibration": cal, "bytes_per_aa": bits // 8, "cold": cold, "warm": warm,
                    "cold_total": sum(cold.values()) + np.mean(search_s), "warm_total": sum(warm.values()) + np.mean(search_s)}
    log(f"{cfg}: cold {results[cfg]['cold_total']:.2f} s/protein, warm {results[cfg]['warm_total']:.2f} s/protein, "
        f"{sum(len(o['accepted']) for o in out.values())} relatives accepted")

# ---------------------------------------------------------------- accepted-relative comparison vs base
for cfg in RUN:
    kept = new = nb = 0
    for q in hard:
        b, c = set(accepted["base"][q]["accepted"]), set(accepted[cfg][q]["accepted"])
        kept += len(b & c)
        new += len(c - b)
        nb += len(b)
    results[cfg].update({"accepted": sum(len(accepted[cfg][q]["accepted"]) for q in hard),
                         "kept_from_base": kept / nb, "new_vs_base": new})
results["_search_s"] = float(np.mean(search_s))
results["_precal_setup_s"] = {**old.get("_precal_setup_s", {}), **{str(k): v for k, v in setup.items()}}
(RESULTS / "ablation.json").write_text(json.dumps(results, indent=1))
log("ablation search+align done")
