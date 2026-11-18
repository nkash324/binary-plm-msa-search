"""Fold the quick full-length MSAs (MMseqs2, Atlas 20 shards; cropped to the domain) and compare with the same
methods searched with the isolated domain.  Run in the 'cf' env.  Writes results/full_quick.md."""
import glob
import json
import shutil
import subprocess
import time

import numpy as np

from common import DATA, RESULTS

E = DATA / "e2e"
INFO = json.loads((E / "full_quick.json").read_text())
QS = list(INFO["rows"])
CF = ["--num-models", "1", "--num-recycle", "3", "--model-type", "alphafold2_ptm"]
LOG = open(RESULTS / "full_quick.log", "a")


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


def pdb(p, q):
    f = sorted(glob.glob(str(E / "out" / p / f"{q}_unrelaxed_rank_001*.pdb")))
    return f[0] if f else None


for p in ("mm_full", "atlas_full"):
    t0 = time.time()
    shutil.rmtree(E / "out" / p, ignore_errors=True)
    subprocess.run(["colabfold_batch", str(E / "a3m" / p), str(E / "out" / p), *CF], capture_output=True, text=True)
    for q in QS:
        if not pdb(p, q):
            d = E / "retry_full" / p / q
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(E / "a3m" / p / f"{q}.a3m", d)
            subprocess.run(["colabfold_batch", str(d), str(E / "out" / p), *CF], capture_output=True, text=True)
    log(f"folded {p} in {time.time() - t0:.0f}s")


def tm(f, q):
    if not f:
        return np.nan
    o = subprocess.run(["USalign", str(f), str(E / "ref" / f"{q}.ent"), "-outfmt", "2"], capture_output=True, text=True).stdout
    return float([l for l in o.splitlines() if l and not l.startswith("#")][0].split("\t")[3])


T = {p: {q: tm(pdb(p, q), q) for q in QS} for p in ("mmseqs2", "mm_full", "atlas20_only", "atlas_full")}
old = {q: sum(1 for l in (E / "a3m" / d / f"{q}.a3m").read_text().splitlines() if l.startswith(">")) - 1
       for d in ("mmseqs2",) for q in QS}
olda = {q: sum(1 for l in (E / "a3m" / "atlas20_only" / f"{q}.a3m").read_text().splitlines() if l.startswith(">")) - 1
        for q in QS}
L = [f"# Quick full-length test: MMseqs2 vs Atlas (20 shards), {len(QS)} proteins", "",
     "Search with the full-length parent protein (UniProt; max 1022 residues around the domain), MSA cropped back to "
     "the domain, domain folded (ColabFold, 1 model, 3 recycles, no templates). 'Domain query' = earlier results.", "",
     "| MSA | mean TM | correct folds (TM>0.5) |", "|---|---|---|"]
NM = {"mmseqs2": "MMseqs2, domain query", "mm_full": "MMseqs2, full-length query",
      "atlas20_only": "Atlas, domain query", "atlas_full": "Atlas, full-length query"}
for p in T:
    t = np.array([T[p][q] for q in QS])
    L.append(f"| {NM[p]} | {np.nanmean(t):.3f} | {(t > 0.5).sum()}/{len(QS)} |")
L += ["", "| protein | parent (length) | MMseqs2 rows domain / full | MMseqs2 TM domain / full | Atlas rows domain / full | "
      "Atlas TM domain / full |", "|---|---|---|---|---|---|"]
for q in QS:
    pa, r = INFO["parents"][q], INFO["rows"][q]
    L.append(f"| {q} | {pa['acc']} ({pa['parent_len']}) | {old[q]} / {r['mm_rows_domain']} | "
             f"{T['mmseqs2'][q]:.2f} / {T['mm_full'][q]:.2f} | {olda[q]} / {r['atlas_rows_domain']} | "
             f"{T['atlas20_only'][q]:.2f} / {T['atlas_full'][q]:.2f} |")
(RESULTS / "full_quick.md").write_text("\n".join(L) + "\n")
log("\n".join(L))
