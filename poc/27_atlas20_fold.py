"""Fold the 20-shard Atlas MSAs and compare with the 1-shard results (same proteins, same AlphaFold settings).
Run in the 'cf' env.  Writes results/atlas20.md."""
import glob
import json
import subprocess
import time
from math import comb

import numpy as np

from common import DATA, RESULTS

E = DATA / "e2e"
NEW = ["atlas20_only", "mm_atlas20"]
LOG = open(RESULTS / "atlas20.log", "a")
CF = ["--num-models", "1", "--num-recycle", "3", "--model-type", "alphafold2_ptm"]


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


for p in NEW:
    t0 = time.time()
    subprocess.run(["colabfold_batch", str(E / "a3m" / p), str(E / "out" / p), *CF], capture_output=True, text=True)
    for a in sorted((E / "a3m" / p).glob("*.a3m")):          # batch crashed part-way? fold the missing ones singly
        if not glob.glob(str(E / "out" / p / f"{a.stem}_unrelaxed_rank_001*.pdb")):
            d = E / "retry_atlas" / p / a.stem
            d.mkdir(parents=True, exist_ok=True)
            (d / a.name).write_text(a.read_text())
            subprocess.run(["colabfold_batch", str(d), str(E / "out" / p), *CF], capture_output=True, text=True)
    log(f"folded {p} in {time.time() - t0:.0f}s")


def tm_score(pred, ref):
    o = subprocess.run(["USalign", str(pred), str(ref), "-outfmt", "2"], capture_output=True, text=True).stdout
    return float([l for l in o.splitlines() if l and not l.startswith("#")][0].split("\t")[3])


def sign_p(w, l):
    n = w + l
    return min(1.0, sum(comb(n, k) for k in range(max(w, l), n + 1)) / 2 ** n * 2) if n else 1.0


def perm_p(d, rng=np.random.default_rng(0)):
    return float(np.mean([abs((d * rng.choice([-1, 1], len(d))).mean()) >= abs(d.mean()) - 1e-12 for _ in range(20000)]))


prev = json.loads((RESULTS / "e2e.json").read_text())["accuracy"]
qs = sorted({k.split("|")[1] for k in prev})
tm = {p: {q: prev[f"{p}|{q}"]["tm"] for q in qs} for p in ("jackhmmer", "mmseqs2", "yours")}
for p in ("atlas_only", "mm_atlas", *NEW):
    tm[p] = {}
    for q in qs:
        pdb = sorted(glob.glob(str(E / "out" / p / f"{q}_unrelaxed_rank_001*.pdb")))
        if pdb:
            tm[p][q] = tm_score(pdb[0], E / "ref" / f"{q}.ent")
qs = [q for q in qs if all(q in tm[p] for p in tm)]
NAMES = {"jackhmmer": "JackHMMER", "mmseqs2": "MMseqs2", "yours": "BinPLM (Swiss-Prot)", "atlas_only": "Atlas 1 shard",
         "mm_atlas": "MMseqs2 + Atlas 1 shard", "atlas20_only": "Atlas 20 shards", "mm_atlas20": "MMseqs2 + Atlas 20 shards"}
a1 = json.loads((E / "atlas_msa.json").read_text())
a20 = json.loads((E / "atlas20_msa.json").read_text())

L = [f"# Atlas 20 shards (~{a20['proteins'] / 1e6:.0f}M proteins) vs 1 shard (0.5M): {len(qs)} fresh hard proteins", "",
     "Same method as the 1-shard test (binary index -> top 100 -> 150M embedding alignment -> chance cutoff); only the "
     "database is bigger. AlphaFold: ColabFold, 1 model, 3 recycles, no templates.", "",
     "| MSA | mean TM | median TM | correct folds (TM>0.5) |", "|---|---|---|---|"]
for p in tm:
    t = np.array([tm[p][q] for q in qs])
    L.append(f"| {NAMES[p]} | {t.mean():.3f} | {np.median(t):.3f} | {(t > 0.5).sum()}/{len(qs)} |")
L += ["", "| comparison | mean TM change | better | worse | sign p | permutation p |", "|---|---|---|---|---|---|"]
for a, b in (("atlas20_only", "atlas_only"), ("atlas20_only", "mmseqs2"), ("atlas20_only", "yours"),
             ("mm_atlas20", "mm_atlas"), ("mm_atlas20", "mmseqs2")):
    d = np.array([tm[a][q] - tm[b][q] for q in qs])
    w, l = int((d > 0.05).sum()), int((d < -0.05).sum())
    L.append(f"| {NAMES[a]} vs {NAMES[b]} | {d.mean():+.3f} | {w} | {l} | {sign_p(w, l):.2f} | {perm_p(d):.3f} |")
L += ["", "| protein | rows 1 shard | rows 20 shards (from new shards) | " + " | ".join(NAMES[p] for p in tm) + " |",
      "|---|---|---|" + "---|" * len(tm)]
for q in qs:
    L.append(f"| {q} | {a1['accepted'].get(q, 0)} | {a20['accepted'].get(q, 0)} ({a20['from_new_shards'].get(q, 0)}) | "
             + " | ".join(f"{tm[p][q]:.2f}" for p in tm) + " |")
L += ["", "Time per protein, 1 shard: " + ", ".join(f"{k}: {v:.2f}s" for k, v in a1["seconds_per_protein"].items()),
      "", "Time per protein, 20 shards: " + ", ".join(f"{k}: {v:.2f}s" for k, v in a20["seconds_per_protein"].items())]
(RESULTS / "atlas20.md").write_text("\n".join(L) + "\n")
log("\n".join(L))
