"""Fold the two Atlas MSAs and compare with the end-to-end folds (same proteins, same AlphaFold settings).
Run in the 'cf' env.  Writes results/atlas_fold.md."""
import glob
import json
import subprocess
import time
from math import comb
from pathlib import Path

import numpy as np

from common import DATA, RESULTS

E = DATA / "e2e"
NEW = ["mm_atlas", "hyb_atlas"]
LOG = open(RESULTS / "atlas_fold.log", "a")


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


def tm_score(pred, ref):
    o = subprocess.run(["USalign", str(pred), str(ref), "-outfmt", "2"], capture_output=True, text=True).stdout
    return float([l for l in o.splitlines() if l and not l.startswith("#")][0].split("\t")[3])


for p in NEW:
    t0 = time.time()
    subprocess.run(["colabfold_batch", str(E / "a3m" / p), str(E / "out" / p), "--num-models", "1", "--num-recycle", "3",
                    "--model-type", "alphafold2_ptm"], capture_output=True, text=True)
    for a in sorted((E / "a3m" / p).glob("*.a3m")):          # batch crashed part-way? fold the missing ones singly
        q = a.stem
        if not glob.glob(str(E / "out" / p / f"{q}_unrelaxed_rank_001*.pdb")):
            d = E / "retry_atlas" / p / q
            d.mkdir(parents=True, exist_ok=True)
            (d / a.name).write_text(a.read_text())
            subprocess.run(["colabfold_batch", str(d), str(E / "out" / p), "--num-models", "1", "--num-recycle", "3",
                            "--model-type", "alphafold2_ptm"], capture_output=True, text=True)
    log(f"folded {p} in {time.time() - t0:.0f}s")

prev = json.loads((RESULTS / "e2e.json").read_text())["accuracy"]
acc = json.loads((E / "atlas_msa.json").read_text())
qs = sorted({k.split("|")[1] for k in prev})
tm = {p: {q: prev[f"{p}|{q}"]["tm"] for q in qs} for p in ("jackhmmer", "mmseqs2", "yours", "hybrid")}
for p in NEW:
    tm[p] = {}
    for q in qs:
        pdb = sorted(glob.glob(str(E / "out" / p / f"{q}_unrelaxed_rank_001*.pdb")))
        if pdb:
            tm[p][q] = tm_score(pdb[0], E / "ref" / f"{q}.ent")
qs = [q for q in qs if all(q in tm[p] for p in tm)]
NAMES = {"jackhmmer": "JackHMMER", "mmseqs2": "MMseqs2", "yours": "BinPLM (Swiss-Prot)", "hybrid": "Hybrid (MMseqs2 + BinPLM)",
         "mm_atlas": "MMseqs2 + Atlas", "hyb_atlas": "Hybrid + Atlas"}


def sign_p(w, l):
    n = w + l
    return min(1.0, sum(comb(n, k) for k in range(max(w, l), n + 1)) / 2 ** n * 2) if n else 1.0


def perm_p(d, rng=np.random.default_rng(0)):
    return float(np.mean([abs((d * rng.choice([-1, 1], len(d))).mean()) >= abs(d.mean()) - 1e-12 for _ in range(20000)]))


L = [f"# Adding ESM Atlas relatives (one 500k-protein shard) to the MSA: {len(qs)} fresh hard proteins", "",
     "Atlas relatives: binary index of Meta's ESM-2 3B per-protein vectors -> top 100 -> embedding alignment -> "
     "only those above the chance cutoff. AlphaFold: ColabFold, 1 model, 3 recycles, no templates.", "",
     "| MSA | mean TM | median TM | correct folds (TM>0.5) |", "|---|---|---|---|"]
for p in tm:
    t = np.array([tm[p][q] for q in qs])
    L.append(f"| {NAMES[p]} | {t.mean():.3f} | {np.median(t):.3f} | {(t > 0.5).sum()}/{len(qs)} |")
L += ["", "| comparison | mean TM change | better | worse | sign p | permutation p |", "|---|---|---|---|---|---|"]
for a, b in (("mm_atlas", "mmseqs2"), ("hyb_atlas", "hybrid"), ("mm_atlas", "hybrid"), ("hyb_atlas", "mmseqs2")):
    d = np.array([tm[a][q] - tm[b][q] for q in qs])
    w, l = int((d > 0.05).sum()), int((d < -0.05).sum())
    L.append(f"| {NAMES[a]} vs {NAMES[b]} | {d.mean():+.3f} | {w} | {l} | {sign_p(w, l):.2f} | {perm_p(d):.3f} |")
L += ["", "| protein | Atlas rows added | " + " | ".join(NAMES[p] for p in tm) + " |", "|---|---|" + "---|" * len(tm)]
for q in qs:
    L.append(f"| {q} | {acc['accepted'].get(q, 0)} | " + " | ".join(f"{tm[p][q]:.2f}" for p in tm) + " |")
L += ["", "Time per protein for the Atlas part: " + ", ".join(f"{k}: {v:.2f}s" for k, v in acc["seconds_per_protein"].items())]
(RESULTS / "atlas_fold.md").write_text("\n".join(L) + "\n")
log("\n".join(L))
