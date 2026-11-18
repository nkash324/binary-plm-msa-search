"""Today's search tools on the same workload (CPU, 6 processes / threads).

  jackhmmer_af2 : AlphaFold2's UniRef90/MGnify settings (-N 1, --F1/F2/F3 filters, E 1e-4)
  mmseqs_cf     : ColabFold-style MMseqs2 (3 iterations, -s 8, E 0.1, max 10000 seqs)
  phmmer_max    : HMMER with all speed filters OFF = exhaustive reference (slow; the "answer key")

Writes hits/<method>.json = {"hits": {query: [targets]}, "seconds_per_query": ..., "note": ...}
Usage: python 03_baselines.py [method ...]
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from common import CPU_WORKERS, DATA, DB_FASTA, Q_FASTA, read_fasta

HITS = DATA / "hits"
HITS.mkdir(exist_ok=True)
q_ids, q_seqs = read_fasta(Q_FASTA)
NPROC = CPU_WORKERS

HMMER = {
    "jackhmmer_af2": ["jackhmmer", "-N", "1", "--F1", "0.0005", "--F2", "0.00005", "--F3", "0.0000005",
                      "--incE", "0.0001", "-E", "0.0001"],
    "phmmer_max": ["phmmer", "--max", "-E", "1e-3"],
}


def run_hmmer(name):
    work = DATA / f"{name}_work"
    work.mkdir(exist_ok=True)

    def one(k):
        qf, tbl = work / f"{q_ids[k]}.fa", work / f"{q_ids[k]}.tbl"
        sf = work / f"{q_ids[k]}.secs"
        if sf.exists():                                  # finished in an earlier (interrupted) run
            secs = float(sf.read_text())
        else:
            qf.write_text(f">{q_ids[k]}\n{q_seqs[k]}\n")
            t = time.time()
            subprocess.run(HMMER[name] + ["--cpu", "1", "--noali", "-o", "/dev/null", "--tblout", str(tbl),
                                          str(qf), str(DB_FASTA)], check=True)
            secs = time.time() - t
            sf.write_text(str(secs))
        hits = []
        for line in open(tbl):
            if not line.startswith("#"):
                t_id = line.split()[0]
                if t_id not in hits:
                    hits.append(t_id)
        return hits, secs

    t0 = time.time()
    with ThreadPoolExecutor(NPROC) as ex:
        out = list(ex.map(one, range(len(q_ids))))
    wall = time.time() - t0
    cpu = sum(s for _, s in out) / len(out)
    return {q: h for q, (h, _) in zip(q_ids, out)}, cpu, \
        f"{cpu:.1f} s/query on 1 core; {wall / len(q_ids):.1f} s/query wall with {NPROC} in parallel"


def run_mmseqs():
    work = DATA / "mmseqs_work"
    work.mkdir(exist_ok=True)
    sh = lambda *a: subprocess.run(["mmseqs", *map(str, a)], check=True, stdout=subprocess.DEVNULL)
    if not (work / "db").exists():
        sh("createdb", DB_FASTA, work / "db")
    sh("createdb", Q_FASTA, work / "q")
    for f in work.glob("res*"):
        f.unlink()
    t0 = time.time()
    sh("search", work / "q", work / "db", work / "res", work / "tmp", "--num-iterations", 3, "-s", 8,
       "-e", 0.1, "--max-seqs", 10000, "--threads", NPROC)
    secs = time.time() - t0
    sh("convertalis", work / "q", work / "db", work / "res", work / "res.m8")
    hits = {q: [] for q in q_ids}
    for line in open(work / "res.m8"):
        q, t = line.split("\t")[:2]
        if t not in hits[q]:
            hits[q].append(t)
    return hits, secs / len(q_ids), \
        f"{secs / len(q_ids):.1f} s/query wall ({NPROC} threads, batch of {len(q_ids)})"


for m in sys.argv[1:] or ["mmseqs_cf", "jackhmmer_af2", "phmmer_max"]:
    print(f"== {m}", flush=True)
    hits, spq, note = run_mmseqs() if m == "mmseqs_cf" else run_hmmer(m)
    (HITS / f"{m}.json").write_text(json.dumps({"hits": hits, "seconds_per_query": spq, "note": note}))
    print(f"   {note}; mean hits/query {sum(map(len, hits.values())) / len(hits):.0f}", flush=True)
