"""Profile-HMM baselines (the tools AlphaFold's own pipeline uses):
  phmmer     single-pass profile search (sequence query)
  jackhmmer  iterative profile search, -N 3 rounds (as in AlphaFold2's UniRef90/MGnify searches)

All-vs-all over SCOPe40; queries are split into chunks run as parallel single-threaded processes.
Usage: python 06_hmmer.py phmmer|jackhmmer [n_queries_for_timing_test]
"""
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import DATA, TOP_N, load_meta, read_fasta, save_ranking

tool = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
assert tool in ("phmmer", "jackhmmer")

meta = load_meta()
sid2i = {s: i for i, s in enumerate(meta["sid"])}
N = len(meta["sid"])
db = DATA / "seqs.fasta"
ids, seqs = read_fasta(db)
if limit:
    ids, seqs = ids[:limit], seqs[:limit]

work = DATA / f"{tool}_work"
work.mkdir(exist_ok=True)
n_jobs = os.cpu_count()
chunks = [list(range(j, len(ids), n_jobs)) for j in range(n_jobs)]


def run_chunk(j):
    qf, tbl = work / f"q{j}.fa", work / f"hits{j}.tbl"
    with open(qf, "w") as f:
        for i in chunks[j]:
            f.write(f">{ids[i]}\n{seqs[i]}\n")
    cmd = [tool, "--cpu", "1", "--noali", "-o", "/dev/null", "--tblout", str(tbl), "-E", "10"]
    if tool == "jackhmmer":
        cmd += ["-N", "3"]
    subprocess.run(cmd + [str(qf), str(db)], check=True)
    return tbl


t0 = time.time()
with ThreadPoolExecutor(n_jobs) as ex:
    tbls = list(ex.map(run_chunk, range(n_jobs)))
secs = time.time() - t0

# tblout columns: target, t_acc, query, q_acc, E-value, score, ...
# jackhmmer may report a target once per round; the last line seen is the final round.
best = defaultdict(dict)
for tbl in tbls:
    for line in open(tbl):
        if line.startswith("#"):
            continue
        c = line.split()
        q, t = sid2i[c[2]], sid2i[c[0]]
        if q != t:
            best[q][t] = (float(c[4]), -float(c[5]))

ranked = []
for q in range(N):
    h = sorted((v, t) for t, v in best.get(q, {}).items())
    ranked.append(np.array([t for _, t in h[:TOP_N]], dtype=np.int32))

if limit:
    print(f"{tool}: {limit} queries in {secs:.1f}s with {n_jobs} processes "
          f"-> full {N} queries ~ {secs / limit * N / 60:.0f} min")
else:
    lens = np.array([len(s) for s in seqs])
    note = "-N 3 iterations" if tool == "jackhmmer" else "single pass"
    save_ranking(f"{tool}", ranked, secs, lens.mean(), note=note)
