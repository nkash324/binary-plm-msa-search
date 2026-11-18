"""Uniform last step for every method: align its candidates to the query and build the MSA.

Same aligner for everyone (phmmer --max, E < E_VERIFY at full-database size), so differences
come only from the SEARCH step. Per query and method we report:
  verified    homologs confirmed by alignment
  distant     of those, how many share < 30% identity with the query (the informative relatives)
  Neff        effective number of sequences in the MSA (80% identity clustering), the depth
              measure AlphaFold accuracy tracks
  recall      verified / verified by the exhaustive reference (phmmer_max)
"""
import json
import os
import re
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from common import CPU_WORKERS, DATA, DB_FASTA, E_VERIFY, Q_FASTA, RESULTS, read_fasta

HITS = DATA / "hits"
WORK = DATA / "verify_work"
WORK.mkdir(exist_ok=True)
db_ids, db_seqs = read_fasta(DB_FASTA)
Z = len(db_ids)
db = dict(zip(db_ids, db_seqs))
del db_ids, db_seqs
q_ids, q_seqs = read_fasta(Q_FASTA)
AA = {a: i for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}

GPU_LOCK = threading.Lock()
methods = {p.stem: json.loads(p.read_text()) for p in sorted(HITS.glob("*.json"))}


def read_sto(path):
    """Match-state columns only (#=GC RF == 'x'): each row has query length."""
    seqs, rf = defaultdict(str), ""
    for line in open(path):
        if line.startswith("#=GC RF"):
            rf += line.split()[-1]
        elif line.strip() and not line.startswith(("#", "//")):
            name, s = line.split()
            seqs[name] += s
    keep = np.frombuffer(rf.encode(), dtype=np.uint8) == ord("x")
    return {n: "".join(np.array(list(s))[keep]) for n, s in seqs.items()}


def neff_and_identity(qseq, rows):
    """Neff at 80% identity (incl. query) + identity of each row to the query, on the GPU."""
    L = len(qseq)
    allr = [qseq] + rows
    X = torch.zeros(len(allr), L, 21, device="cuda", dtype=torch.float16)
    for r, s in enumerate(allr):
        idx = [(p, AA.get(c.upper(), 20)) for p, c in enumerate(s) if c not in "-."]
        if idx:
            p, a = zip(*idx)
            X[r, list(p), list(a)] = 1
    X = X.view(len(allr), -1)
    nongap = X.sum(1).float()
    G = (X @ X.T).float()                                     # identical residues per pair
    both = (X.view(len(allr), L, 21).sum(2) @ X.view(len(allr), L, 21).sum(2).T).float()
    ident = G / both.clamp(min=1)
    w = 1.0 / (ident >= 0.8).float().sum(1)
    q_ident = (G[0, 1:] / nongap[1:].clamp(min=1)).cpu().numpy()
    return float(w.sum()), q_ident


def verify(method, qi):
    q = q_ids[qi]
    cands = [t for t in methods[method]["hits"][q] if t in db]
    if not cands:
        return 0, 0, 1.0, set(), 0.0
    cf, qf, sto = WORK / f"{method}.{q}.fa", WORK / f"{q}.q.fa", WORK / f"{method}.{q}.sto"
    cf.write_text("".join(f">{t}\n{db[t]}\n" for t in cands))
    qf.write_text(f">{q}\n{q_seqs[qi]}\n")
    t0 = time.time()
    tbl = WORK / f"{method}.{q}.tbl"
    subprocess.run(["phmmer", "--max", "--cpu", "1", "-E", str(E_VERIFY), "--incE", str(E_VERIFY), "--incdomE", str(E_VERIFY),
                    "-Z", str(Z), "-o", "/dev/null", "-A", str(sto), "--tblout", str(tbl), str(qf), str(cf)],
                   check=True)
    secs = time.time() - t0
    found = {l.split()[0] for l in open(tbl) if not l.startswith("#")}
    cf.unlink()
    if not found:
        return 0, 0, 1.0, found, secs
    rows = list(read_sto(sto).values())
    with GPU_LOCK:                                   # one GPU job at a time (8 GB card)
        neff, qid = neff_and_identity(q_seqs[qi], rows)
        torch.cuda.empty_cache()
    return len(found), int((qid < 0.3).sum()), neff, found, secs


results = {}
for m in methods:
    t0 = time.time()
    with ThreadPoolExecutor(CPU_WORKERS) as ex:
        results[m] = list(ex.map(lambda qi: verify(m, qi), range(len(q_ids))))
    print(f"verified {m} in {time.time() - t0:.0f}s", flush=True)

ref = [r[3] for r in results["phmmer_max"]]
rows, per_query = [], {}
for m, res in results.items():
    rec = [len(r[3] & ref[k]) / len(ref[k]) for k, r in enumerate(res) if ref[k]]
    # "hard" queries: those with a shallow reference MSA (few homologs exist at all)
    rows.append({
        "method": m, "search_s_per_query": methods[m]["seconds_per_query"],
        "align_s_per_query": float(np.mean([r[4] for r in res])),
        "candidates": float(np.mean([len(methods[m]["hits"][q]) for q in q_ids])),
        "verified": float(np.mean([r[0] for r in res])), "distant": float(np.mean([r[1] for r in res])),
        "median_neff": float(np.median([r[2] for r in res])), "recall": float(np.mean(rec)),
        "note": methods[m]["note"]})
    per_query[m] = [{"q": q_ids[k], "verified": r[0], "distant": r[1], "neff": r[2]} for k, r in enumerate(res)]

ref_neff = np.array([r[2] for r in results["phmmer_max"]])
hard = ref_neff < np.median(ref_neff)
for row in rows:
    nf = np.array([p["neff"] for p in per_query[row["method"]]])
    row["median_neff_hard_half"] = float(np.median(nf[hard]))
    row["shallow_msas(neff<30)"] = int((nf < 30).sum())

rows.sort(key=lambda r: -r["recall"])
(RESULTS / "poc_results.json").write_text(json.dumps({"summary": rows, "per_query": per_query}, indent=1))
hdr = ["method", "search_s_per_query", "align_s_per_query", "candidates", "verified", "distant",
       "recall", "median_neff", "median_neff_hard_half", "shallow_msas(neff<30)"]
lines = ["# Proof of concept: 100 SCOPe queries vs Swiss-Prot (576k proteins)", "",
         "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
for r in rows:
    lines.append("| " + " | ".join(f"{r[h]:.3g}" if isinstance(r[h], float) else str(r[h]) for h in hdr) + " |")
lines += ["", *[f"- **{r['method']}**: {r['note']}" for r in rows]]
(RESULTS / "poc_results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
