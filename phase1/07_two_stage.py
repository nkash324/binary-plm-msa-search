"""Phase 1b: the two-stage search a real system would use, timed per query.

  Stage 1 (prefilter): per-protein 640-bit codes, true Hamming distance on the CPU
                        (XOR + popcount over 10 x uint64 words). Keep the top K proteins.
  Stage 2 (rerank):     per-residue 640-bit codes of ONLY those K proteins, diagonal score
                        (same scoring as residue-center-diag in 04_search.py), on the GPU.

Output ranking = reranked candidates, then the rest of the stage-1 list (so recall@1000 is fair).
Usage: python 07_two_stage.py [K ...]      (default: 200 1000 3000)
"""
import sys
import time

import numpy as np
import torch

from common import DATA, EMB_DIM, TOP_N, load_meta, save_ranking

DEV = "cuda"
D = EMB_DIM
K_HITS = 64
DIAG_BIN = 8
rng = np.random.default_rng(0)

meta = load_meta()
N = len(meta["sid"])
offsets = np.load(DATA / "offsets.npy")
lens = np.diff(offsets)


# ------------------------------------------------------------------ binary codes (built once)
def build_codes():
    pb, rb = DATA / "prot_bits.npy", DATA / "res_bits.npy"
    if not (pb.exists() and rb.exists()):
        prot = np.load(DATA / "prot_emb.npy")
        np.save(pb, np.packbits(prot - prot.mean(0) >= 0, axis=1))
        res = np.load(DATA / "res_emb.npy", mmap_mode="r")
        mu = np.asarray(res[:: max(1, len(res) // 200_000)], np.float32).mean(0)
        out = np.lib.format.open_memmap(rb, mode="w+", dtype=np.uint8, shape=(len(res), D // 8))
        for s in range(0, len(res), 200_000):
            out[s:s + 200_000] = np.packbits(np.asarray(res[s:s + 200_000], np.float32) - mu >= 0, axis=1)
        out.flush()
    return np.load(pb), np.load(rb, mmap_mode="r")


prot_bits, res_bits = build_codes()
prot64 = np.ascontiguousarray(prot_bits).view(np.uint64)                    # N x 10
SHIFTS = torch.arange(7, -1, -1, device=DEV, dtype=torch.uint8)
res_gpu = torch.from_numpy(np.asarray(res_bits)).to(DEV)                    # 2.8M x 80 bytes = 224 MB


def unpack_pm1(bits):
    """uint8 (n, 80) on GPU -> fp16 (n, 640) of +/-1."""
    b = (bits.unsqueeze(-1) >> SHIFTS) & 1
    return (b.reshape(bits.shape[0], D).half() * 2 - 1)


# ------------------------------------------------------------------ scoring
def diag_scores(hit_idx, hit_val, qpos, owner, pos, n_t):
    """Diagonal aggregation (same as 04_search.aggregate, diag part) over n_t local targets."""
    L, k = hit_idx.shape
    t = owner[hit_idx].ravel().astype(np.int64)
    s = hit_val.ravel().astype(np.float32)
    qrow = np.repeat(np.arange(L, dtype=np.int64), k)
    d = pos[hit_idx].ravel().astype(np.int64) - np.repeat(qpos, k)
    nb = 2 * (2048 // DIAG_BIN) + 4
    b = d // DIAG_BIN + nb // 2
    key = (qrow * n_t + t) * nb + b
    o = np.argsort(key, kind="stable")
    k3, s3 = key[o], s[o]
    st = np.flatnonzero(np.r_[True, k3[1:] != k3[:-1]])
    best = np.maximum.reduceat(s3, st)
    utb, inv = np.unique(k3[st] % (n_t * nb), return_inverse=True)
    tot = np.bincount(inv, weights=best)
    nxt = np.zeros_like(tot)
    adj = np.flatnonzero(utb[1:] == utb[:-1] + 1)
    nxt[adj] = tot[adj + 1]
    out = np.zeros(n_t)
    np.maximum.at(out, utb // nb, tot + nxt)
    return out


def search(K):
    ranked = []
    t_s1 = t_s2 = 0.0
    for q in range(N):
        # ---- stage 1: Hamming over all per-protein codes (CPU)
        t0 = time.perf_counter()
        ham = np.bitwise_count(prot64 ^ prot64[q]).sum(1).astype(np.float32)
        ham[q] = np.inf
        ham += rng.random(N, dtype=np.float32) * 0.5                       # random tie-break
        top = np.argpartition(ham, TOP_N)[:TOP_N]
        s1 = top[np.argsort(ham[top])]
        cand = s1[:K]
        t1 = time.perf_counter()

        # ---- stage 2: per-residue rerank of the K candidates (GPU)
        cl = lens[cand]
        rows = np.repeat(offsets[cand] - np.r_[0, np.cumsum(cl)[:-1]], cl) + np.arange(cl.sum())
        owner = np.repeat(np.arange(K, dtype=np.int64), cl)
        pos = rows - np.repeat(offsets[cand], cl)
        C = unpack_pm1(res_gpu[torch.from_numpy(rows).to(DEV)])
        Q = unpack_pm1(res_gpu[offsets[q]:offsets[q + 1]])
        v, i = (Q @ C.T).topk(min(K_HITS, C.shape[0]), dim=1)
        sc = diag_scores(i.cpu().numpy(), v.float().cpu().numpy(),
                         np.arange(lens[q], dtype=np.int64), owner, pos, K)
        good = np.flatnonzero(sc > 0)
        order = good[np.argsort(-sc[good], kind="stable")]
        rest = np.setdiff1d(np.arange(K), order, assume_unique=True)
        ranked.append(np.concatenate([cand[order], cand[rest], s1[K:]])[:TOP_N].astype(np.int32))
        t2 = time.perf_counter()
        t_s1 += t1 - t0
        t_s2 += t2 - t1
        if q % 3000 == 0:
            print(f"  K={K}: {q}/{N}", flush=True)
    note = (f"stage1 {t_s1 / N * 1e3:.2f} ms/query (CPU Hamming), "
            f"stage2 {t_s2 / N * 1e3:.2f} ms/query (GPU rerank of {K})")
    print("   ", note, flush=True)
    bpp = (D / 8) * (1 + lens.mean())          # per-protein code + per-residue codes
    save_ranking(f"twostage-K{K}", ranked, t_s1 + t_s2, bpp, note=note)


if __name__ == "__main__":
    for K in [int(a) for a in sys.argv[1:]] or [200, 1000, 3000]:
        search(K)
