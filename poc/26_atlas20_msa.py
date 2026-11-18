"""Atlas relatives from the 20-shard index (~10M proteins) -> MSA rows, for the 20 fresh proteins.
Identical to 22_atlas_msa.py except the index: 20 shards instead of 1 (same bit recipe, same K=100 candidates,
same 150M codes, same embedding alignment and chance cutoff).
Writes e2e/a3m/{atlas20_only, mm_atlas20} and e2e/atlas20_msa.json.  Run in the 'fold' env."""
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
LOG = open(RESULTS / "atlas20.log", "a")
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

# ---------------------------------------------------------------- 1. load the 20-shard index into RAM
t0 = time.perf_counter()
names = [u.rsplit("/", 1)[1][:-4] for u in (RESULTS / "atlas20_shards.txt").read_text().split()]
codes = np.ascontiguousarray(np.concatenate([np.load(A / "idx20" / f"{n}.codes.npy") for n in names])).view(np.uint64)
ids, shard_of = [], []
for s, n in enumerate(names):
    x = (A / "idx20" / f"{n}.ids.txt").read_text().split()
    ids += x
    shard_of += [s] * len(x)
shard_of = np.array(shard_of)
N = len(ids)
pca = np.load(A / "pca.npz")
mu, P = pca["mu"], pca["P"]
log(f"20-shard index loaded: {N:,} proteins, {codes.nbytes / 1e6:.0f} MB in RAM, {time.perf_counter() - t0:.1f}s")

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
        ham = np.bitwise_count(codes ^ qc).sum(1)
        top = np.argpartition(ham, K)[:K]
        cand[q] = top[np.lexsort((top, ham[top]))]
        tic("2 Atlas binary search (20 shards)", t)
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
new = sum(ids[i] not in cache for i in need)
with ThreadPoolExecutor(8) as ex:
    for k, s in ex.map(fetch, need):
        if s:
            cache[k] = s
cache_f.write_text(json.dumps(cache))
T["3 fetch sequences (API, 8 parallel)"] = time.perf_counter() - t
log(f"sequences: {sum(ids[i] in cache for i in need)}/{len(need)} available ({new} newly fetched)")

# ---------------------------------------------------------------- 3. embed candidates (150M) + embedding alignment
tok = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
m1 = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D", add_pooling_layer=False,
                              attn_implementation="sdpa").eval().cuda().half()
mu_res = torch.from_numpy(np.load(DATA / "mu.npz")["res"]).cuda()
gp = np.load(DATA / "pca_res.npz")
gmu, gP = torch.from_numpy(gp["mu"]).cuda(), torch.from_numpy(gp["P"][:, :256].copy()).cuda()
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


db_ids, _ = read_fasta(DB_FASTA)
off = np.load(DATA / "offsets.npy")
lens = np.diff(off)
R256 = np.load(DATA / "res_bits256.npy", mmap_mode="r")
null = np.random.default_rng(42).choice(len(db_ids), 2000, replace=False)[:200]
Tnull = [ea.pm1(np.array(R256[off[p]:off[p + 1]])) for p in null]
for d in ("atlas20_only", "mm_atlas20"):
    (E / "a3m" / d).mkdir(parents=True, exist_ok=True)
stats, from_new = {}, {}
for q, s in zip(qids, qseqs):
    t = time.perf_counter()
    Q = codes256([s])[0]
    ci = [i for i in cand[q] if ids[i] in cache]
    cs_ids = [ids[i] for i in ci]
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
    mm = (E / "a3m" / "mmseqs2" / f"{q}.a3m").read_text()
    (E / "a3m" / "mm_atlas20" / f"{q}.a3m").write_text(mm + add)
    (E / "a3m" / "atlas20_only" / f"{q}.a3m").write_text("\n".join(mm.splitlines()[:2]) + "\n" + add)
    stats[q] = len(rows)
    from_new[q] = int(sum(shard_of[ci[k]] != 0 for k in keep))
    log(f"  {q}: {len(cs_ids)} candidates -> {len(rows)} accepted ({from_new[q]} from the 19 new shards)")
per = {k: v / len(qids) for k, v in T.items()}
(E / "atlas20_msa.json").write_text(json.dumps({"accepted": stats, "from_new_shards": from_new,
                                                "proteins": N, "seconds_per_protein": per}, indent=1))
log("per protein: " + ", ".join(f"{k} {v:.2f}s" for k, v in per.items()))
