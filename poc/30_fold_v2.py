"""Fold only the proteins whose MSA changed under the v2 cutoff (unchanged MSA = reuse the old fold), score, report.
Run in the 'cf' env.  Writes results/cutoff_v2.md."""
import glob
import json
import shutil
import subprocess
import time
from math import comb

import numpy as np

from common import DATA, RESULTS

E = DATA / "e2e"
INFO = json.loads((E / "cutoff_v2.json").read_text())
OLD = {"yours_v2": "yours", "atlas20_v2": "atlas20_only"}
CF = ["--num-models", "1", "--num-recycle", "3", "--model-type", "alphafold2_ptm"]
LOG = open(RESULTS / "cutoff_v2.log", "a")


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


def pdb(p, q):
    f = sorted(glob.glob(str(E / "out" / p / f"{q}_unrelaxed_rank_001*.pdb")))
    return f[0] if f else None


for p, old in OLD.items():
    t0 = time.time()
    todo = [q for q, v in INFO[p].items() if v["changed"]]
    if todo:
        d = E / "batch_v2" / p
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        for q in todo:
            shutil.copy(E / "a3m" / p / f"{q}.a3m", d)
        subprocess.run(["colabfold_batch", str(d), str(E / "out" / p), *CF], capture_output=True, text=True)
        for q in todo:                                         # crash part-way? retry singly
            if not pdb(p, q):
                s = E / "retry_v2" / p / q
                s.mkdir(parents=True, exist_ok=True)
                shutil.copy(E / "a3m" / p / f"{q}.a3m", s)
                subprocess.run(["colabfold_batch", str(s), str(E / "out" / p), *CF], capture_output=True, text=True)
    log(f"folded {p}: {len(todo)} changed proteins in {time.time() - t0:.0f}s")


def tm_score(f, q):
    o = subprocess.run(["USalign", str(f), str(E / "ref" / f"{q}.ent"), "-outfmt", "2"], capture_output=True, text=True).stdout
    return float([l for l in o.splitlines() if l and not l.startswith("#")][0].split("\t")[3])


def sign_p(w, l):
    n = w + l
    return min(1.0, sum(comb(n, k) for k in range(max(w, l), n + 1)) / 2 ** n * 2) if n else 1.0


def perm_p(d, rng=np.random.default_rng(0)):
    return float(np.mean([abs((d * rng.choice([-1, 1], len(d))).mean()) >= abs(d.mean()) - 1e-12 for _ in range(20000)]))


qs = sorted(INFO["yours_v2"])
tm = {}
for p in ("jackhmmer", "mmseqs2", "yours", "atlas20_only"):
    tm[p] = {q: tm_score(pdb(p, q), q) for q in qs if pdb(p, q)}
for p, old in OLD.items():
    tm[p] = {}
    for q in qs:
        f = pdb(p, q) if INFO[p][q]["changed"] else pdb(old, q)
        if f:
            tm[p][q] = tm_score(f, q)
qs = [q for q in qs if all(q in tm[p] for p in tm)]
NAMES = {"jackhmmer": "JackHMMER", "mmseqs2": "MMseqs2", "yours": "BinPLM (Swiss-Prot), old cutoff",
         "yours_v2": "BinPLM (Swiss-Prot), new cutoff", "atlas20_only": "Atlas 20 shards, old cutoff",
         "atlas20_v2": "Atlas 20 shards, new cutoff"}
L = [f"# New acceptance cutoff (random AND look-alike chance levels): {len(qs)} fresh hard proteins", "",
     "Only proteins whose MSA changed were refolded; unchanged MSAs reuse the old fold. AlphaFold: ColabFold, 1 model, "
     "3 recycles, no templates.", "", "| MSA | mean TM | median TM | correct folds (TM>0.5) |", "|---|---|---|---|"]
for p in tm:
    t = np.array([tm[p][q] for q in qs])
    L.append(f"| {NAMES[p]} | {t.mean():.3f} | {np.median(t):.3f} | {(t > 0.5).sum()}/{len(qs)} |")
L += ["", "| comparison | mean TM change | better | worse | sign p | permutation p |", "|---|---|---|---|---|---|"]
for a, b in (("yours_v2", "yours"), ("atlas20_v2", "atlas20_only"), ("yours_v2", "mmseqs2"), ("atlas20_v2", "mmseqs2")):
    d = np.array([tm[a][q] - tm[b][q] for q in qs])
    w, l = int((d > 0.05).sum()), int((d < -0.05).sum())
    L.append(f"| {NAMES[a]} vs {NAMES[b]} | {d.mean():+.3f} | {w} | {l} | {sign_p(w, l):.2f} | {perm_p(d):.3f} |")
L += ["", "| protein | Swiss-Prot rows old -> new | Atlas rows old -> new | " + " | ".join(NAMES[p] for p in tm) + " |",
      "|---|---|---|" + "---|" * len(tm)]
for q in qs:
    y, a = INFO["yours_v2"][q], INFO["atlas20_v2"][q]
    L.append(f"| {q} | {y['old']} -> {y['new']} | {a['old']} -> {a['new']} | " + " | ".join(f"{tm[p][q]:.2f}" for p in tm) + " |")
(RESULTS / "cutoff_v2.md").write_text("\n".join(L) + "\n")
log("\n".join(L))
