"""Where does the ~3.3 s per protein of the embedding alignment step go?  Same steps and settings
as 11_find_align.py, with a stopwatch around each part. 1 warm-up protein + 4 measured."""
import json
import time
from collections import defaultdict

import numpy as np
import torch

import emb_align as ea
from common import DATA, DB_FASTA, Q_FASTA, RESULTS, read_fasta

T = defaultdict(float)
sync = torch.cuda.synchronize


class tick:
    def __init__(self, k):
        self.k = k

    def __enter__(self):
        sync()
        self.t = time.perf_counter()

    def __exit__(self, *a):
        sync()
        T[self.k] += time.perf_counter() - self.t


# ---- timed copy of emb_align.sw / sw_many
def sw(Q, Ts, bias, gap, keep, tag):
    Lq, D = Q.shape
    B, Lm = len(Ts), max(len(t) for t in Ts)
    with tick(f"{tag}: pad+copy targets"):
        Tt = torch.zeros(B, Lm, D, device="cuda", dtype=torch.half)
        pad = torch.ones(B, Lm, device="cuda", dtype=torch.bool)
        for k, t in enumerate(Ts):
            Tt[k, :len(t)] = t
            pad[k, :len(t)] = False
    with tick(f"{tag}: similarity grid (matmul)"):
        S = torch.einsum("id,bjd->bij", Q, Tt).float() / D - bias
        S.masked_fill_(pad[:, None, :], -1e4)
    with tick(f"{tag}: DP row loop (GPU)"):
        jg = gap * torch.arange(Lm + 1, device="cuda", dtype=torch.float32)
        prev = torch.zeros(B, Lm + 1, device="cuda")
        zero = torch.zeros(B, 1, device="cuda")
        best = torch.zeros(B, device="cuda")
        Hs = [prev] if keep else None
        for i in range(Lq):
            t = torch.clamp(torch.maximum(prev[:, :-1] + S[:, i], prev[:, 1:] - gap), min=0)
            cur = torch.cummax(torch.cat([zero, t], 1) + jg, 1).values - jg
            best = torch.maximum(best, cur.amax(1))
            prev = cur
            if keep:
                Hs.append(cur)
        out = best.cpu().numpy()
    if not keep:
        return out, None
    with tick(f"{tag}: copy DP matrix to CPU"):
        H = torch.stack(Hs, 1).cpu().numpy()
        S = S.cpu().numpy()
    return out, [(H[k, :, :len(t) + 1], S[k, :, :len(t)]) for k, t in enumerate(Ts)]


def sw_many(Q, Ts, bias, gap, tag, keep=False, cells=20_000_000):
    order = np.argsort([len(t) for t in Ts])
    scores, pairs, k, nb = np.zeros(len(Ts)), [None] * len(Ts), 0, 0
    while k < len(order):
        n = 1
        while k + n < len(order) and (n + 1) * len(Ts[order[k + n]]) * len(Q) <= cells:
            n += 1
        idx = order[k:k + n]
        s, mats = sw(Q, [Ts[i] for i in idx], bias, gap, keep, tag)
        scores[idx] = s
        nb += 1
        if keep:
            with tick(f"{tag}: traceback (CPU, Python)"):
                for i, (H, S) in zip(idx, mats):
                    pairs[i] = ea.traceback(H, S, gap)
        k += n
    T[f"{tag}: GPU batches"] += nb
    return scores, pairs


# ---- same setup as 11_find_align.py
P = json.loads((DATA / "emb_align_params.json").read_text())
BIAS, GAP = P["bias"], P["gap"]
db_ids, db_seqs = read_fasta(DB_FASTA)
NP = len(db_ids)
off = np.load(DATA / "offsets.npy")
lens = np.diff(off)
res_bits = np.load(DATA / "res_bits.npy", mmap_mode="r")
q_ids, _ = read_fasta(Q_FASTA)
q_off = np.load(DATA / "q_offsets.npy")
q_res = np.load(DATA / "q_res_bits.npy")
summ = json.loads((RESULTS / "poc_results.json").read_text())["per_query"]
ref_neff = {r["q"]: r["neff"] for r in summ["phmmer_max"]}
hard = sorted(q_ids, key=lambda q: ref_neff[q])[:20]
mm_hits = json.loads((DATA / "hits" / "mmseqs_cf.json").read_text())["hits"]
id2idx = {d: i for i, d in enumerate(db_ids)}
z = np.load(DATA / "win64_256.npz")
codes = np.ascontiguousarray(z["codes"]).view(np.uint64)
owner, MU, PM = z["owner"], z["mu"], z["P"]
W, B, TOPW = 64, 256, 2000
null_pool = np.random.default_rng(42).choice(NP, 2000, replace=False)


def window_search(qi):
    X = np.unpackbits(q_res[q_off[qi]:q_off[qi + 1]], axis=1).astype(np.float32) * 2 - 1
    L = len(X)
    starts = sorted(set(list(range(0, max(1, L - W + 1), W // 4)) + [max(0, L - W)]))
    qc = np.ascontiguousarray(np.packbits((np.stack([X[a:a + W].mean(0) for a in starts]) - MU) @ PM > 0,
                                          axis=1)).view(np.uint64)
    score = np.zeros(NP, np.float32)
    for lo in range(0, len(codes), 500_000):
        c = codes[lo:lo + 500_000]
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


def load(ps, tag):
    with tick(f"{tag}: read codes from disk"):
        raw = [np.asarray(res_bits[off[p]:off[p + 1]]) for p in ps]
    with tick(f"{tag}: unpack + send to GPU"):
        return [ea.pm1(r) for r in raw]


def one(q):
    qi = q_ids.index(q)
    Q = ea.pm1(q_res[q_off[qi]:q_off[qi + 1]])
    exclude = {id2idx[t] for t in mm_hits.get(q, []) if t in id2idx}
    with tick("0 window index search (CPU)"):
        sc = window_search(qi)
        sc[list(exclude)] = -1
        cand = np.argsort(-sc)[:500]
        cand = cand[sc[cand] > 0]
    cset = set(cand.tolist())
    null = np.array([p for p in null_pool if p not in exclude and p not in cset])
    ns, _ = sw_many(Q, load(null, "1 chance level (2000 random)"), BIAS, GAP, "1 chance level (2000 random)")
    A = np.c_[np.ones(len(null)), np.log(lens[null])]
    coef, *_ = np.linalg.lstsq(A, ns, rcond=None)
    sd = np.std(ns - A @ coef)
    zf = lambda s, ps: (s - np.c_[np.ones(len(ps)), np.log(lens[ps])] @ coef) / sd
    nz = zf(ns, null)
    cs, _ = sw_many(Q, load(cand, "2 candidates (500)"), BIAS, GAP, "2 candidates (500)")
    acc = cand[zf(cs, cand) > nz.max()]
    decoy = null[np.argsort(-nz)][:len(acc)]
    if len(acc):
        sw_many(Q, load(acc, "3 re-align accepted + traceback"), BIAS, GAP, "3 re-align accepted + traceback", keep=True)
        sw_many(Q, load(decoy, "4 decoys + traceback"), BIAS, GAP, "4 decoys + traceback", keep=True)
    return len(Q), len(acc)


one(hard[0])                       # warm-up (GPU start-up), not counted
T.clear()
picks = [hard[1], hard[4], hard[11], hard[12]]
t0 = time.perf_counter()
info = [one(q) for q in picks]
total = time.perf_counter() - t0
n = len(picks)
lines = [f"# Alignment profile: {n} proteins (lengths {[i[0] for i in info]}, accepted {[i[1] for i in info]})", "",
         f"Total {total / n:.2f} s per protein", "", "| part | s per protein | share |", "|---|---|---|"]
timed = {k: v for k, v in T.items() if not k.endswith("GPU batches")}
for k in sorted(timed):
    lines.append(f"| {k} | {timed[k] / n:.3f} | {100 * timed[k] / total:.0f}% |")
lines.append(f"| (not covered by stopwatches) | {(total - sum(timed.values())) / n:.3f} | "
             f"{100 * (total - sum(timed.values())) / total:.0f}% |")
lines += ["", "GPU batches per protein: " + ", ".join(f"{k.split(':')[0]} {v / n:.0f}"
                                                   for k, v in T.items() if k.endswith("GPU batches"))]
(RESULTS / "align_profile.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
