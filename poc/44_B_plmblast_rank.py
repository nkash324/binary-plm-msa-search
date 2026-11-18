"""Test B: turn pLM-BLAST's hit table (1000 fixed queries vs all 15,177 SCOPe domains) into a ranking in the benchmark
format (hits sorted by pLM-BLAST score, best first; queries are the same 1000 used for every method's subset score).
Usage: python 44_B_plmblast_rank.py <plmblast_output.csv> <seconds>"""
import importlib.util
import sys

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("p1common", "/mnt/c/Folding/phase1/common.py")
p1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p1)
meta = p1.load_meta()
sid2i = {s: i for i, s in enumerate(meta["sid"])}
df = pd.read_csv(sys.argv[1], sep=";")
print("pLM-BLAST columns:", list(df.columns), "rows:", len(df))
qc = next(c for c in df.columns if c.lower() in ("qid", "query_id", "query"))
tc = next(c for c in df.columns if c.lower() in ("sid", "target_id", "tid", "target"))
sc = next(c for c in df.columns if c.lower() == "score")
ranked = [np.zeros(0, np.int32) for _ in meta["sid"]]
for q, g in df.groupby(qc):
    g = g.sort_values(sc, ascending=False)
    seen, r = set(), []
    for t in g[tc]:
        t = sid2i.get(str(t).split()[0])
        if t is not None and t not in seen:
            seen.add(t)
            r.append(t)
    qi = sid2i.get(str(q).split()[0])
    if qi is not None:
        ranked[qi] = np.array(r[:p1.TOP_N], np.int32)
print("queries with hits:", sum(len(r) > 0 for r in ranked))
p1.save_ranking("plmblast", ranked, float(sys.argv[2]), 0, note="pLM-BLAST defaults (cpc 70, cutoff 0.3), ProtT5; 1000 queries")
