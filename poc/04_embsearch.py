"""The proposed search, as a real system would run it on this PC (tiered storage, simulated):

  RAM : per-protein 640-bit codes (576k x 80 B = 46 MB)  -> stage 1: Hamming (XOR+popcount) on CPU
  Disk: per-residue 640-bit codes (~17 GB, memory-mapped) -> stage 2: read only the K_PREFILTER
        candidates' codes, score them per residue with diagonal agreement on the GPU.

Also saves a 'stage 1 only' list (top K_ALIGN by Hamming) as a control for what stage 2 adds.
Timing uses a cold-ish disk cache: the residue file is ~4x larger than the 4 GB WSL RAM.
"""
import json
import time

import numpy as np
import torch

from common import DATA, EMB_DIM, K_ALIGN, K_PREFILTER, Q_FASTA, DB_FASTA, read_fasta

DEV, D = "cuda", EMB_DIM
K_HITS, DIAG_BIN, ROWS_PER_CHUNK = 64, 8, 400_000
HITS = DATA / "hits"
HITS.mkdir(exist_ok=True)

db_ids = read_fasta(DB_FASTA)[0]
q_ids = read_fasta(Q_FASTA)[0]
off = np.load(DATA / "offsets.npy")
lens = np.diff(off)
q_off = np.load(DATA / "q_offsets.npy")
prot64 = np.ascontiguousarray(np.load(DATA / "prot_bits.npy")).view(np.uint64)
q_prot64 = np.ascontiguousarray(np.load(DATA / "q_prot_bits.npy")).view(np.uint64)
res_bits = np.load(DATA / "res_bits.npy", mmap_mode="r")
q_res_bits = np.load(DATA / "q_res_bits.npy")
SHIFTS = torch.arange(7, -1, -1, device=DEV, dtype=torch.uint8)
rng = np.random.default_rng(0)


def unpack_pm1(bits):
    b = (torch.from_numpy(bits).to(DEV).unsqueeze(-1) >> SHIFTS) & 1
    return b.reshape(bits.shape[0], D).half() * 2 - 1


def diag_scores(hit_idx, hit_val, qpos, owner, pos, n_t):
    L, k = hit_idx.shape
    t = owner[hit_idx].ravel()
    s = hit_val.ravel().astype(np.float32)
    qrow = np.repeat(np.arange(L, dtype=np.int64), k)
    d = pos[hit_idx].ravel() - np.repeat(qpos, k)
    nb = 2 * (40_000 // DIAG_BIN) + 4
    key = (qrow * n_t + t) * nb + d // DIAG_BIN + nb // 2
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


stage2_hits, stage1_hits, times = {}, {}, {"stage1": [], "read": [], "stage2": []}
for qi, q in enumerate(q_ids):
    # ---- stage 1: per-protein Hamming over the whole database (RAM, CPU)
    t0 = time.perf_counter()
    ham = np.bitwise_count(prot64 ^ q_prot64[qi]).sum(1).astype(np.float32)
    ham += rng.random(len(ham), dtype=np.float32) * 0.5
    top = np.argpartition(ham, K_PREFILTER)[:K_PREFILTER]
    cand = top[np.argsort(ham[top])]
    t1 = time.perf_counter()

    # ---- read candidates' per-residue codes from disk, in file order
    disk_order = np.sort(cand)
    cl = lens[disk_order]
    rows = np.repeat(off[disk_order] - np.r_[0, np.cumsum(cl)[:-1]], cl) + np.arange(cl.sum())
    C_bits = res_bits[rows]
    owner = np.repeat(np.arange(len(disk_order), dtype=np.int64), cl)
    pos = rows - np.repeat(off[disk_order], cl)
    t2 = time.perf_counter()

    # ---- stage 2: per-residue similarity + diagonal agreement (GPU)
    Q = unpack_pm1(q_res_bits[q_off[qi]:q_off[qi + 1]])
    vs, is_ = [], []
    for s in range(0, len(rows), ROWS_PER_CHUNK):
        S = Q @ unpack_pm1(C_bits[s:s + ROWS_PER_CHUNK]).T
        v, i = S.topk(min(K_HITS, S.shape[1]), dim=1)
        vs.append(v)
        is_.append(i + s)
    v, j = torch.cat(vs, 1).topk(K_HITS, dim=1)
    i = torch.gather(torch.cat(is_, 1), 1, j)
    sc = diag_scores(i.cpu().numpy(), v.float().cpu().numpy(),
                     np.arange(len(Q), dtype=np.int64), owner, pos, len(disk_order))
    order = np.argsort(-sc, kind="stable")[:K_ALIGN]
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    stage2_hits[q] = [db_ids[disk_order[k]] for k in order]
    stage1_hits[q] = [db_ids[c] for c in cand[:K_ALIGN]]
    for k, dt in zip(("stage1", "read", "stage2"), (t1 - t0, t2 - t1, t3 - t2)):
        times[k].append(dt)
    print(f"  {qi + 1}/{len(q_ids)} {q}: stage1 {1e3 * (t1 - t0):.0f} ms, read {1e3 * (t2 - t1):.0f} ms "
          f"({C_bits.nbytes / 1e6:.0f} MB), stage2 {1e3 * (t3 - t2):.0f} ms", flush=True)

m = {k: float(np.mean(v)) for k, v in times.items()}
tot = sum(m.values())
note = (f"{tot:.2f} s/query: stage1 {m['stage1'] * 1e3:.0f} ms (CPU Hamming, {len(prot64):,} codes), "
        f"disk read {m['read'] * 1e3:.0f} ms, stage2 {m['stage2'] * 1e3:.0f} ms (GPU, {K_PREFILTER} candidates)")
print(note)
SUFFIX = "" if K_PREFILTER == 5000 else f"_k{K_PREFILTER}"
(HITS / f"emb_twostage{SUFFIX}.json").write_text(json.dumps({"hits": stage2_hits, "seconds_per_query": tot, "note": note}))
if not SUFFIX:
  (HITS / "emb_stage1only.json").write_text(json.dumps(
    {"hits": stage1_hits, "seconds_per_query": m["stage1"], "note": "per-protein Hamming top 2000 only"}))
