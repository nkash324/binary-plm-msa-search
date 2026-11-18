"""Embedding search: per-protein / per-window / per-residue, float vs binary.

Binary codes are scored exactly via a +/-1 matrix product on the GPU:
    dot(a, b) = D - 2 * Hamming(a, b)
so the ranking is identical to a Hamming-distance search (this measures accuracy;
speed numbers here are GPU brute force, not an optimized popcount index).

Usage: python 04_search.py [method ...]   (default: all)
Methods are named <level>-<kind>, e.g. protein-float, residue-rot.
  level: protein | window | residue
  kind:  float (cosine) | std (z-scored dims, cosine) |
         sign (raw sign bits) | center (mean-subtracted sign bits) | rot (center + random rotation, sign bits)
"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import DATA, EMB_DIM, TOP_N, load_meta, save_ranking

DEV = "cuda"
D = EMB_DIM
WIN, STRIDE = 32, 16
K_HITS = 64                       # nearest vectors kept per query vector (multi-vector levels)
DIAG_BIN = {"residue": 8, "window": STRIDE}

meta = load_meta()
N = len(meta["sid"])
offsets = np.load(DATA / "offsets.npy")
lens = np.diff(offsets)


# ---------------------------------------------------------------- vectors per level
def window_vectors():
    p = DATA / "win_emb.npy"
    if p.exists():
        return (np.load(p), np.load(DATA / "win_owner.npy"),
                np.load(DATA / "win_pos.npy"), np.load(DATA / "win_off.npy"))
    res = np.load(DATA / "res_emb.npy", mmap_mode="r")
    vecs, owner, pos, off = [], [], [], [0]
    for i in range(N):
        r = np.asarray(res[offsets[i]:offsets[i + 1]], dtype=np.float32)
        L = len(r)
        starts = list(range(0, max(L - WIN, 0) + 1, STRIDE))
        if starts[-1] + WIN < L:
            starts.append(L - WIN)
        cs = np.vstack([np.zeros((1, D), np.float32), np.cumsum(r, 0)])
        for s in starts:
            e = min(s + WIN, L)
            vecs.append(((cs[e] - cs[s]) / (e - s)).astype(np.float16))
            owner.append(i)
            pos.append(s)
        off.append(len(vecs))
    out = (np.stack(vecs), np.array(owner, np.int32), np.array(pos, np.int32), np.array(off, np.int64))
    for name, a in zip(("win_emb", "win_owner", "win_pos", "win_off"), out):
        np.save(DATA / f"{name}.npy", a)
    return out


def level_vectors(level):
    if level == "protein":
        return np.load(DATA / "prot_emb.npy"), None, None, None
    if level == "window":
        return window_vectors()
    res = np.load(DATA / "res_emb.npy", mmap_mode="r")
    owner = np.repeat(np.arange(N, dtype=np.int32), lens)
    pos = (np.arange(offsets[-1]) - np.repeat(offsets[:-1], lens)).astype(np.int32)
    return res, owner, pos, offsets


# ---------------------------------------------------------------- encoding
def random_rotation(seed=0):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(D, D, generator=g))
    return (q * torch.sign(torch.diag(r))).to(DEV)


def build_db(vecs, kind, chunk=200_000):
    """Encode all vectors onto the GPU as fp16: unit vectors (float) or +/-1 codes (binary)."""
    n = vecs.shape[0]
    db = torch.empty((n, D), dtype=torch.float16, device=DEV)
    mu = sd = R = None
    if kind in ("rot", "std", "center"):
        sample = np.asarray(vecs[:: max(1, n // 200_000)], dtype=np.float32)
        mu = torch.from_numpy(sample.mean(0)).to(DEV)
        sd = torch.from_numpy(sample.std(0) + 1e-6).to(DEV)
        R = random_rotation()
    for s in range(0, n, chunk):
        x = torch.from_numpy(np.asarray(vecs[s:s + chunk], dtype=np.float32)).to(DEV)
        if kind == "float":
            y = F.normalize(x, dim=1)
        elif kind == "sign":
            y = (x >= 0).float() * 2 - 1
        elif kind == "std":                        # per-dimension z-score, then cosine
            y = F.normalize((x - mu) / sd, dim=1)
        elif kind == "center":                     # sign bits after subtracting the mean
            y = ((x - mu) >= 0).float() * 2 - 1
        elif kind == "rot":
            y = (((x - mu) @ R) >= 0).float() * 2 - 1
        else:
            raise ValueError(kind)
        db[s:s + chunk] = y.half()
    return db


def bytes_per_protein(level, kind, n_vecs):
    per_vec = D / 8 if kind in ("sign", "rot", "center") else D * 2   # bits vs fp16
    return per_vec * n_vecs / N


# ---------------------------------------------------------------- per-protein search
def search_protein(kind):
    vecs, *_ = level_vectors("protein")
    t0 = time.time()
    db = build_db(vecs, kind)
    ranked = []
    for s in range(0, N, 1024):
        S = (db[s:s + 1024] @ db.T).float()
        rows = torch.arange(S.shape[0], device=DEV)
        S[rows, rows + s] = -float("inf")
        S += torch.rand_like(S) * 1e-3            # random tie-break (binary scores are integers)
        ranked += list(S.topk(TOP_N, dim=1).indices.cpu().numpy().astype(np.int32))
    torch.cuda.synchronize()
    save_ranking(f"protein-{kind}", ranked, time.time() - t0, bytes_per_protein("protein", kind, N))
    del db
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- multi-vector search
def aggregate(hit_idx, hit_val, qpos, owner, pos, bin_w):
    """Turn per-vector nearest-neighbour hits into two per-target-protein scores.

    maxsim: for each query vector take its best hit inside each target, sum over query vectors.
    diag:   like maxsim but only counting hits that agree on one diagonal (target_pos - query_pos),
            i.e. hits that line up the way a real alignment would. Best (bin, bin+1) pair per target.
    """
    L, k = hit_idx.shape
    t = owner[hit_idx].ravel().astype(np.int64)
    s = hit_val.ravel().astype(np.float32)
    qrow = np.repeat(np.arange(L, dtype=np.int64), k)
    ok = np.isfinite(s)
    t, s, qrow, hi = t[ok], s[ok], qrow[ok], hit_idx.ravel()[ok]

    # maxsim
    key = qrow * N + t
    o = np.argsort(key, kind="stable")
    ks, ss = key[o], s[o]
    st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    best = np.maximum.reduceat(ss, st)
    maxsim = np.bincount(ks[st] % N, weights=best, minlength=N)

    # diagonal
    d = pos[hi].astype(np.int64) - np.repeat(qpos, k)[ok]
    nb = 2 * (2048 // bin_w) + 4
    b = d // bin_w + nb // 2
    key3 = (qrow * N + t) * nb + b
    o = np.argsort(key3, kind="stable")
    k3, s3 = key3[o], s[o]
    st = np.flatnonzero(np.r_[True, k3[1:] != k3[:-1]])
    best3 = np.maximum.reduceat(s3, st)
    tb = k3[st] % (N * nb)                       # (target, bin) key with query row removed
    utb, inv = np.unique(tb, return_inverse=True)
    tot = np.bincount(inv, weights=best3)
    nxt = np.zeros_like(tot)
    adj = np.flatnonzero(utb[1:] == utb[:-1] + 1)
    nxt[adj] = tot[adj + 1]
    diag = np.zeros(N)
    np.maximum.at(diag, utb // nb, tot + nxt)
    return maxsim, diag


def top_targets(score):
    nz = np.flatnonzero(score > 0)
    return nz[np.argsort(-score[nz], kind="stable")][:TOP_N].astype(np.int32)


def search_multi(level, kind, rows_per_chunk):
    vecs, owner, pos, voff = level_vectors(level)
    n = vecs.shape[0]
    t0 = time.time()
    db = build_db(vecs, kind)
    print(f"  {level}-{kind}: {n:,} vectors on GPU ({time.time() - t0:.0f}s)", flush=True)

    out_max, out_diag = [None] * N, [None] * N
    pend_i, pend_v, pend_start = [], [], 0
    p_next = 0
    for a in range(0, n, rows_per_chunk):
        b = min(a + rows_per_chunk, n)
        S = db[a:b] @ db.T
        for p in range(owner[a], owner[b - 1] + 1):      # mask the query protein's own vectors
            r0, r1 = max(voff[p], a) - a, min(voff[p + 1], b) - a
            S[r0:r1, voff[p]:voff[p + 1]] = -float("inf")
        v, i = S.topk(K_HITS, dim=1)
        pend_i.append(i.cpu().numpy())
        pend_v.append(v.float().cpu().numpy())

        if p_next < N and voff[p_next + 1] <= b:
            I, V = np.concatenate(pend_i), np.concatenate(pend_v)
            while p_next < N and voff[p_next + 1] <= b:
                r0, r1 = voff[p_next] - pend_start, voff[p_next + 1] - pend_start
                qpos = pos[voff[p_next]:voff[p_next + 1]].astype(np.int64)
                m, dg = aggregate(I[r0:r1], V[r0:r1], qpos, owner, pos, DIAG_BIN[level])
                m[p_next] = dg[p_next] = 0
                out_max[p_next], out_diag[p_next] = top_targets(m), top_targets(dg)
                p_next += 1
            keep = voff[p_next] - pend_start
            pend_i, pend_v, pend_start = [I[keep:]], [V[keep:]], voff[p_next]
        if (a // rows_per_chunk) % 500 == 0:
            print(f"    {b:,}/{n:,} vectors, {p_next} proteins done, {time.time() - t0:.0f}s", flush=True)

    torch.cuda.synchronize()
    secs = time.time() - t0
    bpp = bytes_per_protein(level, kind, n)
    save_ranking(f"{level}-{kind}-maxsim", out_max, secs, bpp)
    save_ranking(f"{level}-{kind}-diag", out_diag, secs, bpp)
    del db
    torch.cuda.empty_cache()


ALL = ["protein-float", "protein-std", "protein-sign", "protein-center", "protein-rot",
       "window-float", "window-std", "window-sign", "window-center", "window-rot",
       "residue-std", "residue-center"]

if __name__ == "__main__":
    for m in (sys.argv[1:] or ALL):
        level, kind = m.split("-")
        print(f"== {m}", flush=True)
        if level == "protein":
            search_protein(kind)
        else:
            search_multi(level, kind, rows_per_chunk=1024 if level == "window" else 192)
