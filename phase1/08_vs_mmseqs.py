"""Fair fight on the structure-labelled SCOPe benchmark:
  mmseqs2_iter3     MMseqs2 the way ColabFold runs it (3 profile iterations, -s 8), all-vs-all
  protein-pca-256   one 256-bit code per protein (PCA, then sign)          -- 32 B/protein
  window-64-256     one 256-bit code per 64-residue window (PCA of window mean of the residue sign bits)
Rankings saved like the other Phase 1 methods; then run 05_evaluate.py.
"""
import os
import shutil
import subprocess
import time
from collections import defaultdict

import numpy as np

from common import DATA, TOP_N, load_meta, save_ranking

meta = load_meta()
sid2i = {s: i for i, s in enumerate(meta["sid"])}
N = len(meta["sid"])
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- MMseqs2, iterative profile search
from common import RANKINGS
DONE = lambda n: (RANKINGS / f"{n}.npz").exists()
if not DONE("mmseqs2_iter3"):
    fa, tmp = DATA / "seqs.fasta", DATA / "mm_iter"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    db, res, m8 = tmp / "db", tmp / "res", DATA / "mmseqs_iter3.m8"
    t0 = time.time()
    run = lambda *a: subprocess.run(["mmseqs", *map(str, a)], check=True, stdout=subprocess.DEVNULL)
    run("createdb", fa, db)
    run("search", db, db, res, tmp / "t", "--num-iterations", 3, "-s", 8, "-e", 10, "--max-seqs", 4000,
        "--threads", os.cpu_count())
    run("convertalis", db, db, res, m8)
    secs = time.time() - t0
    hits = defaultdict(list)
    for line in open(m8):
        c = line.split("\t")
        q, t = sid2i[c[0]], sid2i[c[1]]
        if q != t:
            hits[q].append((float(c[10]), -float(c[11]), t))
    ranked = []
    for q in range(N):
        seen, r = set(), []
        for _, _, t in sorted(hits.get(q, [])):
            if t not in seen:
                seen.add(t)
                r.append(t)
        ranked.append(np.array(r[:TOP_N], dtype=np.int32))
    save_ranking("mmseqs2_iter3", ranked, secs, 185, note="ColabFold-style: 3 iterations, -s 8")
    shutil.rmtree(tmp, ignore_errors=True)


def pca_bits(X, B):
    mu = X.mean(0)
    w, v = np.linalg.eigh(np.cov((X - mu).T))
    P = v[:, np.argsort(-w)[:B]]
    return mu, P


def pack64(s):
    return np.ascontiguousarray(np.packbits(s, axis=1)).view(np.uint64)


def rank_by(scores_fn):
    out = []
    for q in range(N):
        s = scores_fn(q)
        s[q] = -np.inf
        top = np.argpartition(-s, TOP_N)[:TOP_N]
        out.append(top[np.argsort(-s[top])].astype(np.int32))
    return out


# ---------------------------------------------------------------- protein-pca-256
if not DONE("protein-pca-256"):
    prot = np.load(DATA / "prot_emb.npy").astype(np.float32)
    mu, P = pca_bits(prot, 256)
    codes = pack64((prot - mu) @ P > 0)
    t0 = time.time()
    ranked = rank_by(lambda q: -(np.bitwise_count(codes ^ codes[q]).sum(1).astype(np.float32)
                                 + rng.random(N, dtype=np.float32) * 0.5))
    save_ranking("protein-pca-256", ranked, time.time() - t0, 32, note="256-bit PCA code per protein")

# ---------------------------------------------------------------- window-64-256
W, B, TOPW = 64, 256, 1000
off = np.load(DATA / "offsets.npy")
lens = np.diff(off)
RB = np.load(DATA / "res_bits.npy")                      # packed: 224 MB
nw = -(-lens // W)
pm1 = lambda b: np.unpackbits(b, axis=1).astype(np.float32) * 2 - 1
Ms = []
for g in np.array_split(np.arange(N), 40):               # ~70k residues per chunk -> ~180 MB
    lg = lens[g]
    X = pm1(RB[off[g[0]]:off[g[-1] + 1]])
    local = np.arange(len(X)) - np.repeat(np.cumsum(lg) - lg, lg)
    wid = np.repeat(np.cumsum(nw[g]) - nw[g], lg) + local // W
    st = np.flatnonzero(np.r_[True, wid[1:] != wid[:-1]])
    Ms.append((np.add.reduceat(X, st) / np.diff(np.r_[st, len(wid)])[:, None]).astype(np.float32))
M = np.concatenate(Ms)
del Ms, X
own = np.repeat(np.arange(N), nw)
mu, P = pca_bits(M, B)
wcodes = pack64((M - mu) @ P > 0)
del M


def q_codes(q):
    Xq, L = pm1(RB[off[q]:off[q + 1]]), lens[q]
    s = W // 4
    starts = sorted(set(list(range(0, max(1, L - W + 1), s)) + [max(0, L - W)]))
    return pack64((np.stack([Xq[a:a + W].mean(0) for a in starts]) - mu) @ P > 0)


def window_scores(q):
    qc = q_codes(q)
    h = np.zeros((len(qc), len(wcodes)), np.uint16)
    for w in range(wcodes.shape[1]):
        h += np.bitwise_count(wcodes[None, :, w] ^ qc[:, None, w])
    i = np.argpartition(h, TOPW, axis=1)[:, :TOPW]
    z = (B / 2 - np.take_along_axis(h, i, 1).astype(np.float32)) / (np.sqrt(B) / 2)
    score = np.zeros(N, np.float32)
    for j in range(len(qc)):
        o = np.argsort(-z[j])
        p, first = np.unique(own[i[j][o]], return_index=True)
        score[p] += np.maximum(z[j][o][first], 0)
    return score + rng.random(N, dtype=np.float32) * 1e-3


t0 = time.time()
ranked = rank_by(window_scores)
save_ranking("window-64-256", ranked, time.time() - t0, float(wcodes.nbytes / N),
             note="256-bit PCA code per 64-residue window")
