"""Fold and compare: DHR (Swiss-Prot), big-database MMseqs2 (ColabFold server: UniRef30 + ColabFoldDB), and big-database
MMseqs2 + BinPLM Swiss-Prot rows / + Atlas 20-shard rows.  Same 19 fresh hard proteins, same AlphaFold settings.
Run in the 'cf' env.  Writes results/dhr_bigdb.md."""
import glob
import json
import shutil
import subprocess
import time
from math import comb

import numpy as np

from common import DATA, RESULTS

E = DATA / "e2e"
CF = ["--num-models", "1", "--num-recycle", "3", "--model-type", "alphafold2_ptm"]
LOG = open(RESULTS / "dhr_bigdb.log", "a")
qs = sorted({k.split("|")[1] for k in json.loads((RESULTS / "e2e.json").read_text())["accuracy"]})


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


def rows_of(a3m_text):
    """all sequences after the query (header+row pairs), skipping a leading '#' line."""
    L = [l for l in a3m_text.splitlines() if l and not l.startswith("#")]
    return "\n".join(L[2:]) + ("\n" if len(L) > 2 else "")


# ---------------------------------------------------------------- build the big-database MSAs
for d in ("bigdb", "bigdb_yours", "bigdb_atlas"):
    (E / "a3m" / d).mkdir(parents=True, exist_ok=True)
for q in qs:
    big = (E / "bigdb" / "msa" / f"{q}.a3m").read_text().replace("\x00", "")
    big = big if big.endswith("\n") else big + "\n"
    (E / "a3m" / "bigdb" / f"{q}.a3m").write_text(big)
    (E / "a3m" / "bigdb_yours" / f"{q}.a3m").write_text(big + rows_of((E / "a3m" / "yours" / f"{q}.a3m").read_text()))
    (E / "a3m" / "bigdb_atlas" / f"{q}.a3m").write_text(big + rows_of((E / "a3m" / "atlas20_only" / f"{q}.a3m").read_text()))


def pdb(p, q):
    f = sorted(glob.glob(str(E / "out" / p / f"{q}_unrelaxed_rank_001*.pdb")))
    return f[0] if f else None


NEW = ["dhr", "bigdb", "bigdb_yours", "bigdb_atlas"]
for p in NEW:
    t0 = time.time()
    if not all(pdb(p, q) for q in qs):
        subprocess.run(["colabfold_batch", str(E / "a3m" / p), str(E / "out" / p), *CF], capture_output=True, text=True)
        for q in qs:
            if not pdb(p, q):
                d = E / "retry_big" / p / q
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy(E / "a3m" / p / f"{q}.a3m", d)
                subprocess.run(["colabfold_batch", str(d), str(E / "out" / p), *CF], capture_output=True, text=True)
    log(f"folded {p} in {time.time() - t0:.0f}s")


def tm(f, q):
    if not f:
        return np.nan
    o = subprocess.run(["USalign", str(f), str(E / "ref" / f"{q}.ent"), "-outfmt", "2"], capture_output=True, text=True).stdout
    return float([l for l in o.splitlines() if l and not l.startswith("#")][0].split("\t")[3])


def nrows(p, q):
    return sum(1 for l in (E / "a3m" / p / f"{q}.a3m").read_text().splitlines() if l.startswith(">")) - 1


def sign_p(w, l):
    n = w + l
    return min(1.0, sum(comb(n, k) for k in range(max(w, l), n + 1)) / 2 ** n * 2) if n else 1.0


def perm_p(d, rng=np.random.default_rng(0)):
    return float(np.mean([abs((d * rng.choice([-1, 1], len(d))).mean()) >= abs(d.mean()) - 1e-12 for _ in range(20000)]))


P = ["jackhmmer", "mmseqs2", "yours", "dhr", "atlas20_only", "bigdb", "bigdb_yours", "bigdb_atlas"]
NM = {"jackhmmer": "JackHMMER (Swiss-Prot)", "mmseqs2": "MMseqs2 (Swiss-Prot)", "yours": "BinPLM (Swiss-Prot)",
      "dhr": "DHR (Swiss-Prot)", "atlas20_only": "Atlas 20 shards only", "bigdb": "MMseqs2, big databases (ColabFold)",
      "bigdb_yours": "big-db MMseqs2 + BinPLM rows", "bigdb_atlas": "big-db MMseqs2 + Atlas rows"}
T = {p: {q: tm(pdb(p, q), q) for q in qs} for p in P}
qs = [q for q in qs if all(np.isfinite(T[p][q]) for p in P)]
dh = json.loads((E / "dhr.json").read_text())
L = [f"# DHR and big-database baseline: {len(qs)} fresh hard proteins", "",
     "DHR: official weights, Swiss-Prot, top 2000 -> JackHMMER (DHR flags). Big databases: ColabFold MSA server "
     "(UniRef30 + ColabFoldDB). AlphaFold: ColabFold, 1 model, 3 recycles, no templates.", "",
     "| MSA | mean TM | median TM | correct folds (TM>0.5) | MSA rows (median) |", "|---|---|---|---|---|"]
for p in P:
    t = np.array([T[p][q] for q in qs])
    L.append(f"| {NM[p]} | {t.mean():.3f} | {np.median(t):.3f} | {(t > 0.5).sum()}/{len(qs)} | "
             f"{int(np.median([nrows(p, q) for q in qs]))} |")
L += ["", "| comparison | mean TM change | better | worse | sign p | permutation p |", "|---|---|---|---|---|---|"]
for a, b in (("yours", "dhr"), ("dhr", "jackhmmer"), ("dhr", "mmseqs2"), ("bigdb", "mmseqs2"), ("bigdb", "yours"),
             ("bigdb_yours", "bigdb"), ("bigdb_atlas", "bigdb"), ("atlas20_only", "bigdb")):
    d = np.array([T[a][q] - T[b][q] for q in qs])
    w, l = int((d > 0.05).sum()), int((d < -0.05).sum())
    L.append(f"| {NM[a]} vs {NM[b]} | {d.mean():+.3f} | {w} | {l} | {sign_p(w, l):.2f} | {perm_p(d):.3f} |")
L += ["", "| protein | " + " | ".join(NM[p] for p in P) + " | rows: DHR / big-db |", "|---|" + "---|" * (len(P) + 1)]
for q in qs:
    L.append(f"| {q} | " + " | ".join(f"{T[p][q]:.2f}" for p in P) + f" | {dh[q]['rows']} / {nrows('bigdb', q)} |")
sec = {k: np.median([dh[q]["seconds"][k] for q in qs]) for k in dh[qs[0]]["seconds"]}
L += ["", "DHR time per protein (median): " + ", ".join(f"{k} {v:.2f}s" for k, v in sec.items()),
      "Big-database MSAs came from the public ColabFold server (shared queue), so their wall time is not comparable."]
(RESULTS / "dhr_bigdb.md").write_text("\n".join(L) + "\n")
log("\n".join(L))
