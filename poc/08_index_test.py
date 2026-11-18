"""Binary index test: can a binary index that is SMALLER than the sequences find the right proteins,
so the real alignment only has to look at a few thousand candidates?

Index variants (built from the ESM-2 embeddings we already computed, no new embedding run):
  protein-640      one 640-bit code per protein (what last night used)
  protein-pca-B    one B-bit code per protein (PCA of the protein's mean embedding)
  window-W-B       one B-bit code per non-overlapping window of W residues (PCA of the window's mean)
Score: candidate recall = share of the exhaustive reference's true relatives (phmmer --max over all of
Swiss-Prot) that land in the top K. The alignment step uses the same scoring, so a true relative in the
list is found. Then the real alignment step is timed for the best indexes (phmmer filters on vs --max).
50 queries (every other one of the 100).  Run in the 'fold' env.   SMOKE=1 -> tiny test.
"""
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from common import DATA, DB_FASTA, Q_FASTA, RESULTS, read_fasta

SMOKE = int(os.environ.get("SMOKE", 0))
THREADS = 4                       # of 6 WSL CPUs
WINDOWS, BITS = (32, 64), (128, 256)
KS = (500, 2000, 5000, 20000)
TOPW = 2000                       # nearest DB windows kept per query window
Z_DB = 576073
DEV = "cuda"
torch.cuda.set_per_process_memory_fraction(0.35)
tag = "_smoke" if SMOKE else ""
LOG = open(RESULTS / f"index_test{tag}.log", "a")


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


# ---------------------------------------------------------------- data
db_ids, db_seqs = read_fasta(DB_FASTA)
off = np.load(DATA / "offsets.npy")
NP = 20000 if SMOKE else len(db_ids)
lens = np.diff(off)[:NP]
N_RES = int(lens.sum())
res_bits = np.load(DATA / "res_bits.npy", mmap_mode="r")
q_ids_all, q_seqs_all = read_fasta(Q_FASTA)
q_off = np.load(DATA / "q_offsets.npy")
q_res = np.load(DATA / "q_res_bits.npy")
QSEL = list(range(0, 100, 2))[: 3 if SMOKE else 50]
q_ids = [q_ids_all[i] for i in QSEL]
id2idx = {d: i for i, d in enumerate(db_ids[:NP])}
ref = json.loads((DATA / "hits" / "phmmer_max.json").read_text())["hits"]
truth = {q: {id2idx[t] for t in ref[q] if t in id2idx} for q in q_ids}
log(f"start: {NP:,} proteins, {N_RES:,} residues, {len(q_ids)} queries, "
    f"{np.mean([len(t) for t in truth.values()]):.0f} true relatives/query")
rng = np.random.default_rng(0)


def pm1(bits):
    return np.unpackbits(bits, axis=1).astype(np.float32) * 2 - 1


def window_ids(lens_g, W):
    nw = -(-lens_g // W)
    local = np.arange(lens_g.sum()) - np.repeat(np.cumsum(lens_g) - lens_g, lens_g)
    return np.repeat(np.cumsum(nw) - nw, lens_g) + local // W, nw


def pca(X, n):
    mu = X.mean(0)
    w, v = np.linalg.eigh(np.cov((X - mu).T))
    return mu.astype(np.float32), v[:, np.argsort(-w)[:n]].astype(np.float32)


def pack64(signs):
    return np.ascontiguousarray(np.packbits(signs, axis=1)).view(np.uint64)


# ---------------------------------------------------------------- PCA fits (sample of proteins)
t0 = time.time()
samp = np.sort(rng.choice(NP, 2000, replace=False))
wm = {W: [] for W in WINDOWS}
for g in np.array_split(samp, 4):
    rows = np.concatenate([np.arange(off[p], off[p + 1]) for p in g])
    X = pm1(np.asarray(res_bits[rows]))
    for W in WINDOWS:
        wid, nw = window_ids(lens[g], W)
        st = np.flatnonzero(np.r_[True, wid[1:] != wid[:-1]])
        wm[W].append(np.add.reduceat(X, st) / np.diff(np.r_[st, len(wid)])[:, None])
PC = {W: pca(np.concatenate(wm[W]), max(BITS)) for W in WINDOWS}
pm = np.load(DATA / "prot_mean.npy", mmap_mode="r")
PCP = pca(np.asarray(pm[np.sort(rng.choice(NP, min(NP, 50000), replace=False))], dtype=np.float32), max(BITS))
log(f"PCA fitted in {time.time() - t0:.0f}s")

# ---------------------------------------------------------------- build indexes
index = {}   # name -> (codes uint64 [n, words], owner protein [n], bits)
index["protein-640"] = (np.ascontiguousarray(np.load(DATA / "prot_bits.npy")[:NP]).view(np.uint64),
                        np.arange(NP), 640)
Pm = (np.asarray(pm[:NP], dtype=np.float32) - PCP[0]) @ PCP[1]
for B in BITS:
    index[f"protein-pca-{B}"] = (pack64(Pm[:, :B] > 0), np.arange(NP), B)
del Pm

t0 = time.time()
SH = torch.arange(7, -1, -1, device=DEV, dtype=torch.uint8)
gpu_pc = {W: (torch.from_numpy(PC[W][0]).to(DEV), torch.from_numpy(PC[W][1]).to(DEV)) for W in WINDOWS}
parts = {(W, B): [] for W in WINDOWS for B in BITS}
owners = {W: [] for W in WINDOWS}
p, last = 0, 0
while p < NP:
    p1 = int(min(NP, max(p + 1, np.searchsorted(off, off[p] + 400_000, side="right") - 1)))
    bits = torch.from_numpy(np.asarray(res_bits[off[p]:off[p1]])).to(DEV)
    X = ((bits.unsqueeze(-1) >> SH) & 1).reshape(bits.shape[0], -1).half() * 2 - 1
    for W in WINDOWS:
        wid, nw = window_ids(lens[p:p1], W)
        S = torch.zeros(int(nw.sum()), X.shape[1], device=DEV, dtype=torch.half)
        S.index_add_(0, torch.from_numpy(wid).to(DEV), X)
        cnt = torch.bincount(torch.from_numpy(wid).to(DEV), minlength=S.shape[0]).float()
        Zs = ((S.float() / cnt[:, None] - gpu_pc[W][0]) @ gpu_pc[W][1] > 0).cpu().numpy()
        for B in BITS:
            parts[(W, B)].append(np.packbits(Zs[:, :B], axis=1))
        owners[W].append(np.repeat(np.arange(p, p1), nw))
    del bits, X, S
    p = p1
    if p - last > 50_000 or p == NP:
        last = p
        log(f"  windows: {p:,}/{NP:,} proteins ({time.time() - t0:.0f}s)")
for W in WINDOWS:
    own = np.concatenate(owners[W])
    for B in BITS:
        index[f"window-{W}-{B}"] = (np.ascontiguousarray(np.concatenate(parts[(W, B)])).view(np.uint64), own, B)
del parts, owners
torch.cuda.empty_cache()
log(f"window indexes built in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- query codes
def query_codes(name, qi):
    L = q_off[qi + 1] - q_off[qi]
    if name == "protein-640":
        return np.ascontiguousarray(np.load(DATA / "q_prot_bits.npy")[qi:qi + 1]).view(np.uint64)
    if name.startswith("protein-pca"):
        B = int(name.split("-")[-1])
        v = (np.load(DATA / "q_prot_mean.npy")[qi:qi + 1].astype(np.float32) - PCP[0]) @ PCP[1]
        return pack64(v[:, :B] > 0)
    _, W, B = name.split("-")
    W, B = int(W), int(B)
    X = pm1(q_res[q_off[qi]:q_off[qi + 1]])
    s = max(1, W // 4)
    starts = sorted(set(list(range(0, max(1, L - W + 1), s)) + [max(0, L - W)]))
    M = np.stack([X[a:a + W].mean(0) for a in starts])
    return pack64(((M - PC[W][0]) @ PC[W][1])[:, :B] > 0)


def ham_topk(codes, qc, k):
    """Hamming distance of every query code to every index code; keep the k nearest per query code."""
    def chunk(lo):
        c = codes[lo:lo + 250_000]
        h = np.zeros((len(qc), len(c)), np.uint16)
        for w in range(codes.shape[1]):
            h += np.bitwise_count(c[None, :, w] ^ qc[:, None, w])
        kk = min(k, h.shape[1] - 1)
        i = np.argpartition(h, kk, axis=1)[:, :kk]
        return i + lo, np.take_along_axis(h, i, 1)
    res = list(POOL.map(chunk, range(0, len(codes), 250_000)))
    I, H = np.concatenate([r[0] for r in res], 1), np.concatenate([r[1] for r in res], 1)
    kk = min(k, H.shape[1] - 1)
    j = np.argpartition(H, kk, axis=1)[:, :kk]
    return np.take_along_axis(I, j, 1), np.take_along_axis(H, j, 1)


def search(name, qi, kmax):
    codes, own, B = index[name]
    qc = query_codes(name, qi)
    if name.startswith("protein"):
        I, H = ham_topk(codes, qc, kmax)
        o = np.argsort(H[0] + rng.random(H.shape[1]) * 0.5)
        return I[0][o]
    I, H = ham_topk(codes, qc, TOPW)
    z = (B / 2 - H.astype(np.float32)) / (np.sqrt(B) / 2)      # how far above chance each window match is
    score = np.zeros(NP, np.float32)
    for j in range(len(qc)):                                    # per query window: best match per protein
        o = np.argsort(-z[j])
        p, first = np.unique(own[I[j][o]], return_index=True)
        score[p] += np.maximum(z[j][o][first], 0)
    nz = np.flatnonzero(score)
    top = nz[np.argsort(-score[nz])][:kmax]
    return top


POOL = ThreadPoolExecutor(THREADS)
rows, cand_lists = [], {}
for name, (codes, own, B) in index.items():
    size_b = codes.nbytes
    recs, times, ncand = {k: [] for k in KS}, [], []
    cand_lists[name] = {}
    for n, qi in enumerate(QSEL):
        q = q_ids_all[qi]
        t = time.perf_counter()
        top = search(name, qi, max(KS))
        times.append(time.perf_counter() - t)
        cand_lists[name][q] = top
        ncand.append(len(top))
        if truth[q]:
            for k in KS:
                recs[k].append(len(truth[q] & set(top[:k].tolist())) / len(truth[q]))
    r = {"index": name, "bits_per_code": B, "codes": len(codes), "size_MB": size_b / 1e6,
         "bytes_per_aa": size_b / N_RES, "s_per_query": float(np.mean(times)),
         **{f"recall@{k}": float(np.mean(recs[k])) for k in KS}}
    rows.append(r)
    log(f"{name}: {r['size_MB']:.0f} MB ({r['bytes_per_aa']:.2f} B/aa), {r['s_per_query']:.2f} s/query, "
        + ", ".join(f"R@{k}={r[f'recall@{k}']:.3f}" for k in KS))

# ---------------------------------------------------------------- real alignment on the candidates
small = [r for r in rows if r["bytes_per_aa"] <= 1.0]
best_win = max([r for r in small if r["index"].startswith("window")], key=lambda r: r["recall@2000"])["index"]
seqs = {i: db_seqs[i] for i in range(NP)}
align_rows = []
WORK = Path(tempfile.mkdtemp(prefix="idx_", dir=DATA))
for name in ("protein-640", best_win):
    for K in (500, 2000):
        for mode in ("filtered", "max"):
            def one(qi):
                q = q_ids_all[qi]
                cf, qf, tb = WORK / f"{q}.{K}.fa", WORK / f"{q}.{K}.q.fa", WORK / f"{q}.{K}.{mode}.tbl"
                cf.write_text("".join(f">{i}\n{seqs[i]}\n" for i in cand_lists[name][q][:K].tolist()))
                qf.write_text(f">{q}\n{q_seqs_all[qi]}\n")
                cmd = ["phmmer", "--cpu", "1", "-E", "1e-3", "-Z", str(Z_DB), "-o", "/dev/null",
                       "--tblout", str(tb), str(qf), str(cf)]
                if mode == "max":
                    cmd.insert(1, "--max")
                t = time.perf_counter()
                subprocess.run(cmd, check=True)
                dt = time.perf_counter() - t
                found = {int(l.split()[0]) for l in open(tb) if not l.startswith("#")}
                return dt, (len(found & truth[q]) / len(truth[q]) if truth[q] else None)
            res = list(POOL.map(one, QSEL))
            rec = [r[1] for r in res if r[1] is not None]
            a = {"index": name, "K": K, "mode": mode, "align_s_per_query": float(np.mean([r[0] for r in res])),
                 "recall": float(np.mean(rec))}
            align_rows.append(a)
            log(f"align {name} K={K} {mode}: {a['align_s_per_query']:.2f} s/query (1 core), recall {a['recall']:.3f}")

# ---------------------------------------------------------------- report
seq_bytes = N_RES
lines = [f"# Binary index test: {len(q_ids)} queries vs Swiss-Prot ({NP:,} proteins)", "",
         f"Sequences themselves: {seq_bytes / 1e6:.0f} MB (1 byte per amino acid). Index must be smaller to count as a win.",
         "Recall = share of the true relatives (exhaustive phmmer --max) that make the candidate list.", "",
         "| index | size MB | bytes/aa | s/query (CPU, 4 threads) | " + " | ".join(f"recall@{k}" for k in KS) + " |",
         "|---|---|---|---|" + "---|" * len(KS)]
for r in rows:
    lines.append(f"| {r['index']} | {r['size_MB']:.0f} | {r['bytes_per_aa']:.2f} | {r['s_per_query']:.2f} | "
                 + " | ".join(f"{r[f'recall@{k}']:.3f}" for k in KS) + " |")
lines += ["", "## Real alignment of the candidates (phmmer, 1 CPU core per query)", "",
          "| index | K | mode | align s/query | final recall |", "|---|---|---|---|---|"]
for a in align_rows:
    lines.append(f"| {a['index']} | {a['K']} | {a['mode']} | {a['align_s_per_query']:.2f} | {a['recall']:.3f} |")
lines += ["", "For comparison (same 100-query run last night): JackHMMER 1.35 s + 0.23 s align, recall 0.88; "
          "MMseqs2 0.47 s (6 threads), recall 0.994."]
(RESULTS / f"index_test{tag}.md").write_text("\n".join(lines) + "\n")
(RESULTS / f"index_test{tag}.json").write_text(json.dumps({"index": rows, "align": align_rows}, indent=1))
for f in WORK.iterdir():
    f.unlink()
WORK.rmdir()
log("DONE")
