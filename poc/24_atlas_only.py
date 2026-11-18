"""Atlas only: the protein + the accepted Atlas rows, nothing from MMseqs2 or Swiss-Prot.
Same Atlas rows as 22_atlas_msa.py (taken from the mm_atlas MSAs), same AlphaFold settings.
Run in the 'cf' env.  Writes results/atlas_only.md."""
import glob
import json
import subprocess
import time
from math import comb

import numpy as np

from common import DATA, RESULTS

E = DATA / "e2e"
P = "atlas_only"
LOG = open(RESULTS / "atlas_only.log", "a")


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


# ---------------------------------------------------------------- build the MSAs
(E / "a3m" / P).mkdir(parents=True, exist_ok=True)
rows = {}
for f in sorted((E / "a3m" / "mm_atlas").glob("*.a3m")):
    mm, both = (E / "a3m" / "mmseqs2" / f.name).read_text(), f.read_text()
    assert both.startswith(mm)
    add = both[len(mm):]                                   # the Atlas rows appended by 22_atlas_msa.py
    query = "\n".join(mm.splitlines()[:2]) + "\n"          # >id + sequence
    (E / "a3m" / P / f.name).write_text(query + add)
    rows[f.stem] = add.count(">")
log(f"built {len(rows)} Atlas-only MSAs")

# ---------------------------------------------------------------- fold (batch, then retry missing singly)
t0 = time.time()
cf = ["--num-models", "1", "--num-recycle", "3", "--model-type", "alphafold2_ptm"]
subprocess.run(["colabfold_batch", str(E / "a3m" / P), str(E / "out" / P), *cf], capture_output=True, text=True)
for q in rows:
    if not glob.glob(str(E / "out" / P / f"{q}_unrelaxed_rank_001*.pdb")):
        d = E / "retry_atlas" / P / q
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{q}.a3m").write_text((E / "a3m" / P / f"{q}.a3m").read_text())
        subprocess.run(["colabfold_batch", str(d), str(E / "out" / P), *cf], capture_output=True, text=True)
log(f"folded {P} in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- score
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
for p in ("mm_atlas", P):
    tm[p] = {}
    for q in qs:
        pdb = sorted(glob.glob(str(E / "out" / p / f"{q}_unrelaxed_rank_001*.pdb")))
        if pdb:
            tm[p][q] = tm_score(pdb[0], E / "ref" / f"{q}.ent")
qs = [q for q in qs if all(q in tm[p] for p in tm)]
NAMES = {"jackhmmer": "JackHMMER", "mmseqs2": "MMseqs2", "yours": "BinPLM (Swiss-Prot)", "mm_atlas": "MMseqs2 + Atlas",
         P: "Atlas only"}

L = [f"# Atlas only (no MMseqs2, no Swiss-Prot): {len(qs)} fresh hard proteins", "",
     "MSA = the protein + Atlas relatives (one 500k-protein shard; binary index -> top 100 -> embedding alignment -> "
     "chance cutoff). Proteins with 0 accepted rows fold from the single sequence. AlphaFold: ColabFold, 1 model, "
     "3 recycles, no templates.", "",
     "| MSA | mean TM | median TM | correct folds (TM>0.5) |", "|---|---|---|---|"]
for p in tm:
    t = np.array([tm[p][q] for q in qs])
    L.append(f"| {NAMES[p]} | {t.mean():.3f} | {np.median(t):.3f} | {(t > 0.5).sum()}/{len(qs)} |")
L += ["", "| comparison | mean TM change | better | worse | sign p | permutation p |", "|---|---|---|---|---|---|"]
for b in ("jackhmmer", "mmseqs2", "yours", "mm_atlas"):
    d = np.array([tm[P][q] - tm[b][q] for q in qs])
    w, l = int((d > 0.05).sum()), int((d < -0.05).sum())
    L.append(f"| Atlas only vs {NAMES[b]} | {d.mean():+.3f} | {w} | {l} | {sign_p(w, l):.2f} | {perm_p(d):.3f} |")
L += ["", "| protein | Atlas rows | " + " | ".join(NAMES[p] for p in tm) + " |", "|---|---|" + "---|" * len(tm)]
for q in qs:
    L.append(f"| {q} | {rows.get(q, 0)} | " + " | ".join(f"{tm[p][q]:.2f}" for p in tm) + " |")
(RESULTS / "atlas_only.md").write_text("\n".join(L) + "\n")
log("\n".join(L))
