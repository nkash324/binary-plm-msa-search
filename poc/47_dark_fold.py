"""Test A, stage 4: fold each protein with big-db MSA alone / + BinPLM / + Atlas / + both, 3 random seeds each (TM-score
averaged over seeds, because single seeds swing a lot on shallow MSAs), score, report by MSA depth.
Main result = the shallow group (Neff < 30); deeper groups are secondary.  Run in the 'cf' env.
Env: A_SHALLOW (protein list), A_ROWS (rows json), A_REPORT (report name)."""
import glob
import json
import os
import shutil
import subprocess
import time
from math import comb

import numpy as np

from common import DATA, RESULTS

E = DATA / "e2e"
D = E / "dark"
SETS = os.environ.get("A_SETS", "bigdb,bigdb_yours,bigdb_atlas,bigdb_both").split(",")
NSEED, OUT = 3, "out3"
CF = ["--num-models", "1", "--num-recycle", "3", "--model-type", "alphafold2_ptm", "--random-seed", "0",
      "--num-seeds", str(NSEED)]
LOG = open(RESULTS / "overnight.log", "a")


def log(m):
    m = f"{time.strftime('%H:%M')} A: {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


qs = [l.strip() for l in (D / os.environ.get("A_SHALLOW", "shallow.txt")).read_text().split() if l.strip()]
rows = json.loads((D / os.environ.get("A_ROWS", "rows.json")).read_text())
depth = {l.split("\t")[0]: (int(l.split("\t")[1]), float(l.split("\t")[2]))
         for l in (D / "depth.tsv").read_text().splitlines()[1:] if l.strip()}
qs = sorted(qs, key=lambda q: depth[q][1])                 # shallowest first


def pdb(p, q):
    f = sorted(glob.glob(str(D / OUT / p / f"{q}_unrelaxed_rank_*.pdb")))
    return f if len(f) >= NSEED else None


for p in SETS:
    t0 = time.time()
    work = D / "work" / p                                  # only the listed proteins, shallowest first
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for q in qs:
        if not pdb(p, q):
            shutil.copy(D / "a3m" / p / f"{q}.a3m", work)
    if any(work.iterdir()):
        subprocess.run(["colabfold_batch", str(work), str(D / OUT / p), *CF], capture_output=True, text=True)
    for q in qs:
        if not pdb(p, q):                                  # one retry for crashes
            d = D / "retry" / p / q
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
            shutil.copy(D / "a3m" / p / f"{q}.a3m", d)
            subprocess.run(["colabfold_batch", str(d), str(D / OUT / p), *CF], capture_output=True, text=True)
    log(f"folded {p}: {sum(bool(pdb(p, q)) for q in qs)}/{len(qs)} in {time.time() - t0:.0f}s")


def tm1(f, q):
    o = subprocess.run(["USalign", str(f), str(E / "ref" / f"{q}.ent"), "-outfmt", "2"], capture_output=True,
                       text=True).stdout
    return float([l for l in o.splitlines() if l and not l.startswith("#")][0].split("\t")[3])


def tm(fs, q):
    return float(np.mean([tm1(f, q) for f in fs[:NSEED]])) if fs else np.nan


def sign_p(w, l):
    n = w + l
    return min(1.0, sum(comb(n, k) for k in range(max(w, l), n + 1)) / 2 ** n * 2) if n else 1.0


def perm_p(d, rng=np.random.default_rng(0)):
    return float(np.mean([abs((d * rng.choice([-1, 1], len(d))).mean()) >= abs(d.mean()) - 1e-12 for _ in range(20000)]))


T = {p: {q: tm(pdb(p, q), q) for q in qs} for p in SETS}
ok = [q for q in qs if all(np.isfinite(T[p][q]) for p in SETS)]
NM = {"bigdb": "big-db MMseqs2 (ColabFold) alone", "bigdb_yours": "+ BinPLM Swiss-Prot rows",
      "bigdb_atlas": "+ Atlas rows", "bigdb_both": "+ both"}
GROUPS = [("shallow: Neff < 30 (main result)", 0, 30), ("medium: Neff 30-100", 30, 100), ("deep: Neff >= 100", 100, float("inf"))]
ne = np.array([depth[q][1] for q in depth])
L = [f"# Test A: does adding BinPLM rows / Atlas rows help AlphaFold on top of big-database MSAs? ({len(ok)} folded of {len(qs)})",
     "", f"Server MSAs so far: {len(depth)} of 200 (Neff 80%: <30: {(ne < 30).sum()}, 30-100: "
     f"{((ne >= 30) & (ne < 100)).sum()}, >=100: {(ne >= 100).sum()}).  AlphaFold: 1 model, 3 recycles, "
     f"{NSEED} seeds; TM-score = mean over seeds.  Better/worse = change > 0.05.", ""]
for gname, lo, hi in GROUPS:
    g = [q for q in ok if lo <= depth[q][1] < hi or (hi == float("inf") and depth[q][1] >= lo)]
    L += [f"## {gname}: {len(g)} proteins", ""]
    if not g:
        L += ["(none yet)", ""]
        continue
    L += ["| MSA | mean TM | median TM | correct folds (TM>0.5) |", "|---|---|---|---|"]
    for p in SETS:
        t = np.array([T[p][q] for q in g])
        L.append(f"| {NM[p]} | {t.mean():.3f} | {np.median(t):.3f} | {(t > 0.5).sum()}/{len(g)} |")
    L += ["", "| comparison | mean TM change | better | worse | sign p | permutation p |", "|---|---|---|---|---|---|"]
    for a in SETS[1:]:
        d = np.array([T[a][q] - T["bigdb"][q] for q in g])
        w, l = int((d > 0.05).sum()), int((d < -0.05).sum())
        L.append(f"| {NM[a]} vs alone | {d.mean():+.3f} | {w} | {l} | {sign_p(w, l):.2f} | {perm_p(d):.3f} |")
    L.append("")
L += ["## Per protein (mean TM over seeds)", "",
      "| protein | big-db rows | Neff | + BinPLM rows | + Atlas rows | " + " | ".join(NM[p] for p in SETS) + " |",
      "|---|---|---|---|---|" + "---|" * len(SETS)]
for q in ok:
    L.append(f"| {q} | {depth[q][0]} | {depth[q][1]:.1f} | {rows[q]['yours_rows']} | {rows[q]['atlas_rows']} | "
             + " | ".join(f"{T[p][q]:.2f}" for p in SETS) + " |")
rep = os.environ.get("A_REPORT", "A_dark.md")
(RESULTS / rep).write_text("\n".join(L) + "\n")
log(f"REPORT written: results/{rep}")
print("\n".join(L))
