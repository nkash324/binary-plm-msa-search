"""Atlas relatives -> MSA rows, for the 20 fresh proteins of the end-to-end test.

 1. binary index of the Atlas shard (ESM-2 3B per-protein vectors -> 256 bits) -> top 100 per protein
 2. fetch those sequences from the Atlas API (8 at a time)
 3. embed candidates on the fly with ESM-2 150M, 256-bit residue codes, embedding Smith-Waterman, accept only
    above the chance cutoff (same calibration as the combined settings) -> MSA rows
 4. write two new MSAs per protein: MMseqs2 + Atlas rows, and hybrid (MMseqs2 + BinPLM) + Atlas rows
Run in the 'fold' env.  Output: ~/folding-data/poc/e2e/a3m/{mm_atlas,hyb_atlas}/
"""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

import emb_align as ea
from common import DATA, DB_FASTA, RESULTS, read_fasta

A, E = Path.home() / "folding-data" / "atlas", DATA / "e2e"
K = 100
LOG = open(RESULTS / "atlas_fold.log", "a")
T = {}


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


def tic(k, t0):
    torch.cuda.synchronize()
    T[k] = T.get(k, 0) + time.perf_counter() - t0


qids, qseqs = read_fasta(E / "queries.fasta")
qids, qseqs = qids[1:], qseqs[1:]

# ---------------------------------------------------------------- 1. Atlas binary search (ESM-2 3B query vectors)
t0 = time.perf_counter()
V = np.load(A / "shard_vecs.npy", mmap_mode="r")
ids = (A / "shard_ids.txt").read_text().split()
N = len(ids)
rng = np.random.default_rng(0)
samp = np.asarray(V[np.sort(rng.choice(N, 50_000, replace=False))], np.float32)
mu = samp.mean(0)
w, v = np.linalg.eigh(np.cov((samp - mu).T))
P = v[:, np.argsort(-w)[:256]].astype(np.float32)
codes = np.ascontiguousarray(np.concatenate([np.packbits((np.asarray(V[a:a + 50_000], np.float32) - mu) @ P > 0, axis=1)
                                             for a in range(0, N, 50_000)])).view(np.uint64)
log(f"Atlas binary index rebuilt ({codes.nbytes / 1e6:.0f} MB) in {time.perf_counter() - t0:.0f}s (one-time)")
tok3 = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
m3 = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D", add_pooling_layer=False, torch_dtype=torch.float16,
                              device_map="cuda").eval()
cand = {}
with torch.no_grad():
    for q, s in zip(qids, qseqs):
        t = time.perf_counter()
        e = m3(**tok3([s], return_tensors="pt").to("cuda")).last_hidden_state[0, 1:len(s) + 1].float().mean(0).cpu().numpy()
        tic("1 query embedding (ESM-2 3B)", t)
        t = time.perf_counter()
        qc = np.ascontiguousarray(np.packbits(((e - mu) @ P > 0)[None], axis=1)).view(np.uint64)
        cand[q] = np.argsort(np.bitwise_count(codes ^ qc).sum(1), kind="stable")[:K]
        tic("2 Atlas binary search", t)
del m3
torch.cuda.empty_cache()

# ---------------------------------------------------------------- 2. sequences from the Atlas API (cached)
cache_f = A / "seqs.json"
cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}


def fetch(i):
    if ids[i] in cache:
        return ids[i], cache[ids[i]]
    for k in range(4):
        try:
            with urllib.request.urlopen(f"https://api.esmatlas.com/fetchSequence/{ids[i]}", timeout=60) as r:
                return ids[i], json.loads(r.read())["sequence"]
        except Exception:                                   # noqa: BLE001
            time.sleep(2 * (k + 1))
    return ids[i], None


t = time.perf_counter()
need = sorted({int(i) for c in cand.values() for i in c})
with ThreadPoolExecutor(8) as ex:
    for k, s in ex.map(fetch, need):
        if s:
            cache[k] = s
cache_f.write_text(json.dumps(cache))
T["3 fetch sequences (API, 8 parallel)"] = time.perf_counter() - t
log(f"sequences: {sum(ids[i] in cache for i in need)}/{len(need)} fetched")

# ---------------------------------------------------------------- 3. embed candidates (150M) + embedding alignment
tok = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
m1 = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D", add_pooling_layer=False,
                              attn_implementation="sdpa").eval().cuda().half()
mu_res = torch.from_numpy(np.load(DATA / "mu.npz")["res"]).cuda()
pca = np.load(DATA / "pca_res.npz")
gmu, gP = torch.from_numpy(pca["mu"]).cuda(), torch.from_numpy(pca["P"][:, :256].copy()).cuda()
PRM = json.loads((DATA / "emb_align_params_256.json").read_text())
BIAS, GAP = PRM["bias"], PRM["gap"]


@torch.no_grad()
def codes256(seqs):
    out = []
    for s in seqs:
        s = s[:1022]
        r = m1(**tok([s], return_tensors="pt").to("cuda")).last_hidden_state[0, 1:len(s) + 1].float()
        out.append(((((r - mu_res >= 0).half() * 2 - 1).float() - gmu) @ gP > 0).half() * 2 - 1)
    return out


# chance level: same as the combined settings (200 random Swiss-Prot proteins, fitted tail)
db_ids, _ = read_fasta(DB_FASTA)
off = np.load(DATA / "offsets.npy")
lens = np.diff(off)
R256 = np.load(DATA / "res_bits256.npy", mmap_mode="r")
null = np.random.default_rng(42).choice(len(db_ids), 2000, replace=False)[:200]
Tnull = [ea.pm1(np.array(R256[off[p]:off[p + 1]])) for p in null]
for d in ("mm_atlas", "hyb_atlas"):
    (E / "a3m" / d).mkdir(parents=True, exist_ok=True)
stats = {}
for q, s in zip(qids, qseqs):
    t = time.perf_counter()
    Q = codes256([s])[0]
    cs_ids = [ids[i] for i in cand[q] if ids[i] in cache]
    Tc = codes256([cache[c] for c in cs_ids])
    tic("4 embed query + candidates (ESM-2 150M)", t)
    t = time.perf_counter()
    ns, _ = ea.sw_many(Q, Tnull, BIAS, GAP)
    Al = np.c_[np.ones(len(null)), np.log(lens[null])]
    coef, *_ = np.linalg.lstsq(Al, ns, rcond=None)
    sd = np.std(ns - Al @ coef)
    zf = lambda sc, L: (sc - np.c_[np.ones(len(L)), np.log(L)] @ coef) / sd
    q90, q99 = np.quantile(zf(ns, lens[null]), [0.9, 0.99])
    beta = (q99 - q90) / (np.log(-np.log(0.9)) - np.log(-np.log(0.99)))
    thr = q90 + beta * (-np.log(-np.log(1 - 1 / 2000)) + np.log(-np.log(0.9)))
    sc, _ = ea.sw_many(Q, Tc, BIAS, GAP)
    keep = np.flatnonzero(zf(sc, np.array([len(cache[c][:1022]) for c in cs_ids])) > thr)
    rows = []
    if len(keep):
        _, pr = ea.sw_many(Q, [Tc[k] for k in keep], BIAS, GAP, keep=True)
        for k, pairs in zip(keep, pr):
            r = ["-"] * len(s)
            for i, j in pairs:
                r[i] = cache[cs_ids[k]][j]
            rows.append((cs_ids[k], "".join(r)))
    tic("5 calibrate + align + build rows", t)
    add = "".join(f">{a}\n{r}\n" for a, r in rows)
    for src, d in (("mmseqs2", "mm_atlas"), ("hybrid", "hyb_atlas")):
        (E / "a3m" / d / f"{q}.a3m").write_text((E / "a3m" / src / f"{q}.a3m").read_text() + add)
    stats[q] = len(rows)
    log(f"  {q}: {len(cs_ids)} Atlas candidates -> {len(rows)} accepted")
per = {k: v / len(qids) for k, v in T.items()}
(E / "atlas_msa.json").write_text(json.dumps({"accepted": stats, "seconds_per_protein": per}, indent=1))
log("per protein: " + ", ".join(f"{k} {v:.2f}s" for k, v in per.items()))
