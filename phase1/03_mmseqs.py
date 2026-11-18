"""Baseline: MMseqs2 all-vs-all at maximum sensitivity (-s 7.5)."""
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
fa, out, tmp = DATA / "seqs.fasta", DATA / "mmseqs.m8", DATA / "mmseqs_tmp"

t0 = time.time()
subprocess.run(["mmseqs", "easy-search", str(fa), str(fa), str(out), str(tmp),
                "-s", "7.5", "-e", "10", "--max-seqs", "2000",
                "--threads", str(os.cpu_count())], check=True, stdout=subprocess.DEVNULL)
secs = time.time() - t0
shutil.rmtree(tmp, ignore_errors=True)

hits = defaultdict(list)
with open(out) as f:
    for line in f:
        c = line.split("\t")
        q, t = sid2i[c[0]], sid2i[c[1]]
        if q != t:
            hits[q].append((float(c[10]), -float(c[11]), t))   # e-value asc, then bits desc

ranked = []
for q in range(N):
    h = sorted(hits.get(q, []))
    seen, r = set(), []
    for _, _, t in h:
        if t not in seen:
            seen.add(t)
            r.append(t)
    ranked.append(np.array(r[:TOP_N], dtype=np.int32))

lens = np.array([int(l.split("\t")[3]) for l in open(DATA / "meta.tsv").readlines()[1:]])
save_ranking("mmseqs2_s7.5", ranked, secs, lens.mean(), note="sequence (1 B/residue)")
