"""Smallest index (one 128/256-bit PCA code per protein) with many more candidates:
does recall keep rising, and what does aligning that many candidates cost?  50 queries, 6 CPUs."""
import json
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from common import DATA, DB_FASTA, Q_FASTA, RESULTS, read_fasta

KS = (2000, 5000, 20000, 50000, 100000)
ALIGN_KS = (2000, 5000, 20000, 50000)
Z_DB = 576073
LOG = open(RESULTS / "compressed_k.log", "a")


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


db_ids, db_seqs = read_fasta(DB_FASTA)
NP = len(db_ids)
q_all, qs_all = read_fasta(Q_FASTA)
QSEL = list(range(0, 100, 2))
id2idx = {d: i for i, d in enumerate(db_ids)}
ref = json.loads((DATA / "hits" / "phmmer_max.json").read_text())["hits"]
truth = {q_all[i]: {id2idx[t] for t in ref[q_all[i]] if t in id2idx} for i in QSEL}

rng = np.random.default_rng(0)
pm = np.load(DATA / "prot_mean.npy", mmap_mode="r")
X = np.asarray(pm[np.sort(rng.choice(NP, 50000, replace=False))], dtype=np.float32)
mu = X.mean(0)
w, v = np.linalg.eigh(np.cov((X - mu).T))
P = v[:, np.argsort(-w)[:256]].astype(np.float32)
proj = (np.asarray(pm, dtype=np.float32) - mu) @ P
qproj = (np.load(DATA / "q_prot_mean.npy").astype(np.float32) - mu) @ P

rows, cands = [], {}
for B in (128, 256):
    codes = np.ascontiguousarray(np.packbits(proj[:, :B] > 0, axis=1)).view(np.uint64)
    qc = np.ascontiguousarray(np.packbits(qproj[:, :B] > 0, axis=1)).view(np.uint64)
    rec, times = {k: [] for k in KS}, []
    for qi in QSEL:
        q = q_all[qi]
        t = time.perf_counter()
        h = np.bitwise_count(codes ^ qc[qi]).sum(1).astype(np.float32) + rng.random(NP, dtype=np.float32) * 0.5
        top = np.argpartition(h, max(KS))[:max(KS)]
        top = top[np.argsort(h[top])]
        times.append(time.perf_counter() - t)
        cands[(B, q)] = top
        if truth[q]:
            for k in KS:
                rec[k].append(len(truth[q] & set(top[:k].tolist())) / len(truth[q]))
    r = {"index": f"protein-pca-{B}", "size_MB": codes.nbytes / 1e6, "s_per_query": float(np.mean(times)),
         **{f"recall@{k}": float(np.mean(rec[k])) for k in KS}}
    rows.append(r)
    log(f"{r['index']}: " + ", ".join(f"R@{k}={r[f'recall@{k}']:.3f}" for k in KS))

WORK = Path(tempfile.mkdtemp(prefix="ck_", dir=DATA))
align = []
with ThreadPoolExecutor(6) as pool:
    for B in (128, 256):
        for K in ALIGN_KS:
            def one(qi):
                q = q_all[qi]
                cf, qf, tb = WORK / f"{q}.fa", WORK / f"{q}.q.fa", WORK / f"{q}.tbl"
                cf.write_text("".join(f">{i}\n{db_seqs[i]}\n" for i in cands[(B, q)][:K].tolist()))
                qf.write_text(f">{q}\n{qs_all[qi]}\n")
                t = time.perf_counter()
                subprocess.run(["phmmer", "--cpu", "1", "-E", "1e-3", "-Z", str(Z_DB), "-o", "/dev/null",
                                "--tblout", str(tb), str(qf), str(cf)], check=True)
                dt = time.perf_counter() - t
                found = {int(l.split()[0]) for l in open(tb) if not l.startswith("#")}
                return dt, (len(found & truth[q]) / len(truth[q]) if truth[q] else None)
            res = list(pool.map(one, QSEL))
            a = {"index": f"protein-pca-{B}", "K": K, "align_s_per_query": float(np.mean([r[0] for r in res])),
                 "recall": float(np.mean([r[1] for r in res if r[1] is not None]))}
            align.append(a)
            log(f"align pca-{B} K={K}: {a['align_s_per_query']:.2f} s/query (1 core), recall {a['recall']:.3f}")

for f in WORK.iterdir():
    f.unlink()
WORK.rmdir()
(RESULTS / "compressed_k.json").write_text(json.dumps({"index": rows, "align": align}, indent=1))
log("DONE")
