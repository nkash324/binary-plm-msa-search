"""Steps 1-3: for the 20 hardest test proteins, find distant relatives in Swiss-Prot with the binary
window index (64-residue windows, 256 bits), align them with the embedding aligner, keep only the ones
that score clearly above chance, and write MSA rows.

  index       : built once, saved to win64_256.npz
  candidates  : top 500 by window score, excluding proteins MMseqs2 already found
  chance level: the same aligner on 2000 random Swiss-Prot proteins (length-corrected z-score);
                a candidate is accepted only if it beats the BEST random protein
  decoys      : for the control condition, the highest-scoring random proteins (look-alikes that are
                NOT accepted), as many as there are accepted relatives, aligned the same way
Output: ~/folding-data/poc/embmsa/<query>.json   (run in the 'fold' env)
"""
import json
import os
import time

import numpy as np
import torch

import emb_align as ea
from common import DATA, DB_FASTA, Q_FASTA, RESULTS, read_fasta

N_Q = int(os.environ.get("N_Q", 20))
K_CAND, N_NULL, W, B, TOPW = 500, 2000, 64, 256, 2000
P = json.loads((DATA / "emb_align_params.json").read_text())
BIAS, GAP = P["bias"], P["gap"]
OUT = DATA / "embmsa"
OUT.mkdir(exist_ok=True)
torch.cuda.set_per_process_memory_fraction(0.6)
rng = np.random.default_rng(0)
LOG = open(RESULTS / "align_fold.log", "a")


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


db_ids, db_seqs = read_fasta(DB_FASTA)
NP = len(db_ids)
off = np.load(DATA / "offsets.npy")
lens = np.diff(off)
assert all(lens[i] == len(db_seqs[i]) for i in range(0, NP, 997))
res_bits = np.load(DATA / "res_bits.npy", mmap_mode="r")
q_ids, q_seqs = read_fasta(Q_FASTA)
q_off = np.load(DATA / "q_offsets.npy")
q_res = np.load(DATA / "q_res_bits.npy")
summ = json.loads((RESULTS / "poc_results.json").read_text())["per_query"]
ref_neff = {r["q"]: r["neff"] for r in summ["phmmer_max"]}
hard = sorted(q_ids, key=lambda q: ref_neff[q])[:N_Q]           # same order as 07_fold.py
mm_hits = json.loads((DATA / "hits" / "mmseqs_cf.json").read_text())["hits"]
truth = json.loads((DATA / "hits" / "phmmer_max.json").read_text())["hits"]
id2idx = {d: i for i, d in enumerate(db_ids)}


def pm1np(b):
    return np.unpackbits(b, axis=1).astype(np.float32) * 2 - 1


# ---------------------------------------------------------------- window index (build once)
IDX = DATA / "win64_256.npz"
if not IDX.exists():
    t0 = time.time()
    samp = np.sort(rng.choice(NP, 2000, replace=False))
    ms = []
    for g in np.array_split(samp, 4):
        rows = np.concatenate([np.arange(off[p], off[p + 1]) for p in g])
        X = pm1np(np.asarray(res_bits[rows]))
        lg = lens[g]
        nw = -(-lg // W)
        local = np.arange(lg.sum()) - np.repeat(np.cumsum(lg) - lg, lg)
        wid = np.repeat(np.cumsum(nw) - nw, lg) + local // W
        st = np.flatnonzero(np.r_[True, wid[1:] != wid[:-1]])
        ms.append(np.add.reduceat(X, st) / np.diff(np.r_[st, len(wid)])[:, None])
    M = np.concatenate(ms)
    mu = M.mean(0).astype(np.float32)
    w, v = np.linalg.eigh(np.cov((M - mu).T))
    Pm = v[:, np.argsort(-w)[:B]].astype(np.float32)
    gmu, gP = torch.from_numpy(mu).cuda(), torch.from_numpy(Pm).cuda()
    parts, owners = [], []
    p = 0
    while p < NP:
        p1 = int(min(NP, max(p + 1, np.searchsorted(off, off[p] + 400_000, side="right") - 1)))
        X = ea.pm1(np.asarray(res_bits[off[p]:off[p1]]))
        lg = lens[p:p1]
        nw = -(-lg // W)
        local = np.arange(lg.sum()) - np.repeat(np.cumsum(lg) - lg, lg)
        wid = torch.from_numpy(np.repeat(np.cumsum(nw) - nw, lg) + local // W).cuda()
        S = torch.zeros(int(nw.sum()), X.shape[1], device="cuda", dtype=torch.half).index_add_(0, wid, X)
        cnt = torch.bincount(wid, minlength=S.shape[0]).float()
        parts.append(np.packbits((((S.float() / cnt[:, None]) - gmu) @ gP > 0).cpu().numpy(), axis=1))
        owners.append(np.repeat(np.arange(p, p1), nw))
        p = p1
    np.savez(IDX, codes=np.concatenate(parts), owner=np.concatenate(owners), mu=mu, P=Pm)
    del parts, owners
    torch.cuda.empty_cache()
    log(f"window index built in {time.time() - t0:.0f}s")
z = np.load(IDX)
codes = np.ascontiguousarray(z["codes"]).view(np.uint64)
owner, MU, PM = z["owner"], z["mu"], z["P"]
log(f"index: {len(codes):,} windows, {codes.nbytes / 1e6:.0f} MB")


def window_search(qi):
    X = pm1np(q_res[q_off[qi]:q_off[qi + 1]])
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
            # best match per protein within this chunk, per query window (chunks split proteins rarely)
            score[pp] += np.maximum(zz[j][o][first], 0)
    return score


def emb(p):
    return ea.pm1(res_bits[off[p]:off[p + 1]])


def row(qlen, seq, pairs):
    r = ["-"] * qlen
    for i, j in pairs:
        r[i] = seq[j]
    return "".join(r)


null_pool = np.random.default_rng(42).choice(NP, N_NULL, replace=False)
for n, q in enumerate(hard):
    fn = OUT / f"{q}.json"
    if fn.exists():
        continue
    t0 = time.time()
    qi = q_ids.index(q)
    Q = ea.pm1(q_res[q_off[qi]:q_off[qi + 1]])
    L = len(Q)
    exclude = {id2idx[t] for t in mm_hits.get(q, []) if t in id2idx}
    sc = window_search(qi)
    sc[list(exclude)] = -1
    cand = np.argsort(-sc)[:K_CAND]
    cand = cand[sc[cand] > 0]
    t1 = time.time()
    cset = set(cand.tolist())
    null = np.array([p for p in null_pool if p not in exclude and p not in cset])
    ns, _ = ea.sw_many(Q, [emb(p) for p in null], BIAS, GAP)
    A = np.c_[np.ones(len(null)), np.log(lens[null])]
    coef, *_ = np.linalg.lstsq(A, ns, rcond=None)
    sd = np.std(ns - A @ coef)
    zfun = lambda s, ps: (s - np.c_[np.ones(len(ps)), np.log(lens[ps])] @ coef) / sd
    nz = zfun(ns, null)
    thr = float(nz.max())
    cs, _ = ea.sw_many(Q, [emb(p) for p in cand], BIAS, GAP)
    cz = zfun(cs, cand)
    acc = cand[cz > thr]
    acc_z = cz[cz > thr]
    decoy = null[np.argsort(-nz)][:len(acc)]
    rows_acc, rows_dec = [], []
    if len(acc):
        _, pr = ea.sw_many(Q, [emb(p) for p in acc], BIAS, GAP, keep=True)
        rows_acc = [row(L, db_seqs[p], pairs) for p, pairs in zip(acc, pr)]
        _, pr = ea.sw_many(Q, [emb(p) for p in decoy], BIAS, GAP, keep=True)
        rows_dec = [row(L, db_seqs[p], pairs) for p, pairs in zip(decoy, pr)]
    tset = {id2idx[t] for t in truth.get(q, []) if t in id2idx}
    cov = lambda rs: float(np.mean([sum(c != "-" for c in r) / L for r in rs])) if rs else 0.0
    rec = {"q": q, "L": L, "n_candidates": int(len(cand)), "n_accepted": int(len(acc)), "threshold_z": thr,
           "accepted": [db_ids[p] for p in acc], "accepted_z": acc_z.round(2).tolist(),
           "accepted_also_found_by_exhaustive_phmmer": int(len(set(acc.tolist()) & tset)),
           "coverage_accepted": cov(rows_acc), "coverage_decoys": cov(rows_dec),
           "rows_emb": rows_acc, "rows_decoy": rows_dec, "mmseqs_hits": len(exclude),
           "search_s": t1 - t0, "align_s": time.time() - t1}
    fn.write_text(json.dumps(rec))
    log(f"  [{n + 1}/{len(hard)}] {q} L={L}: MMseqs2 had {len(exclude)}, {len(cand)} candidates -> "
        f"{len(acc)} accepted (z>{thr:.1f}; {rec['accepted_also_found_by_exhaustive_phmmer']} also seen by phmmer), "
        f"coverage {rec['coverage_accepted']:.2f} vs decoys {rec['coverage_decoys']:.2f}, "
        f"{t1 - t0:.1f}s search + {time.time() - t1:.1f}s align")
log("find+align done")
