"""Early DHR report (as soon as the 19 DHR folds exist): DHR vs JackHMMER, MMseqs2, BinPLM (all Swiss-Prot).
Run in the 'cf' env.  Writes results/dhr_early.md."""
import glob
import json
import subprocess
from math import comb

import numpy as np

from common import DATA, RESULTS

E = DATA / "e2e"
qs = sorted({k.split("|")[1] for k in json.loads((RESULTS / "e2e.json").read_text())["accuracy"]})


def pdb(p, q):
    f = sorted(glob.glob(str(E / "out" / p / f"{q}_unrelaxed_rank_001*.pdb")))
    return f[0] if f else None


def tm(f, q):
    if not f:
        return np.nan
    o = subprocess.run(["USalign", str(f), str(E / "ref" / f"{q}.ent"), "-outfmt", "2"], capture_output=True, text=True).stdout
    return float([l for l in o.splitlines() if l and not l.startswith("#")][0].split("\t")[3])


def sign_p(w, l):
    n = w + l
    return min(1.0, sum(comb(n, k) for k in range(max(w, l), n + 1)) / 2 ** n * 2) if n else 1.0


def perm_p(d, rng=np.random.default_rng(0)):
    return float(np.mean([abs((d * rng.choice([-1, 1], len(d))).mean()) >= abs(d.mean()) - 1e-12 for _ in range(20000)]))


P = ["jackhmmer", "mmseqs2", "yours", "dhr"]
NM = {"jackhmmer": "JackHMMER", "mmseqs2": "MMseqs2", "yours": "BinPLM", "dhr": "DHR"}
T = {p: {q: tm(pdb(p, q), q) for q in qs} for p in P}
qs = [q for q in qs if all(np.isfinite(T[p][q]) for p in P)]
dh = json.loads((E / "dhr.json").read_text())
rows = lambda p, q: sum(1 for l in (E / "a3m" / p / f"{q}.a3m").read_text().splitlines() if l.startswith(">")) - 1
L = [f"# DHR vs BinPLM (both on Swiss-Prot): {len(qs)} fresh hard proteins", "",
     "| MSA (Swiss-Prot) | mean TM | median TM | correct folds (TM>0.5) | MSA rows (median) |", "|---|---|---|---|---|"]
for p in P:
    t = np.array([T[p][q] for q in qs])
    L.append(f"| {NM[p]} | {t.mean():.3f} | {np.median(t):.3f} | {(t > 0.5).sum()}/{len(qs)} | "
             f"{int(np.median([rows(p, q) for q in qs]))} |")
L += ["", "| comparison | mean TM change | better | worse | sign p | permutation p |", "|---|---|---|---|---|---|"]
for a, b in (("yours", "dhr"), ("dhr", "jackhmmer"), ("dhr", "mmseqs2")):
    d = np.array([T[a][q] - T[b][q] for q in qs])
    w, l = int((d > 0.05).sum()), int((d < -0.05).sum())
    L.append(f"| {NM[a]} vs {NM[b]} | {d.mean():+.3f} | {w} | {l} | {sign_p(w, l):.2f} | {perm_p(d):.3f} |")
L += ["", "| protein | JackHMMER | MMseqs2 | BinPLM | DHR | rows: BinPLM / DHR |", "|---|---|---|---|---|---|"]
for q in qs:
    L.append(f"| {q} | " + " | ".join(f"{T[p][q]:.2f}" for p in P) + f" | {rows('yours', q)} / {rows('dhr', q)} |")
sec = {k: np.median([dh[q]["seconds"][k] for q in qs]) for k in dh[qs[0]]["seconds"]}
L += ["", "DHR time per protein (median): " + ", ".join(f"{k} {v:.2f}s" for k, v in sec.items())]
(RESULTS / "dhr_early.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
