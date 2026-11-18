"""Fold the end-to-end MSAs (4 pipelines x 20 fresh proteins), time AlphaFold per protein from ColabFold's
own log, score against the real structures, and write results/e2e.md.  Run in the 'cf' env."""
import glob
import json
import re
import subprocess
import time
from datetime import datetime
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

from common import DATA, RESULTS

E = DATA / "e2e"
PIPES = ["jackhmmer", "mmseqs2", "yours", "hybrid"]
NAMES = {"jackhmmer": "JackHMMER (AlphaFold-style)", "mmseqs2": "MMseqs2 (ColabFold-style)",
         "yours": "BinPLM (combined)", "hybrid": "Hybrid (MMseqs2 + BinPLM)"}
LOG = open(RESULTS / "e2e.log", "a")


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


info = json.loads((E / "times.json").read_text())
T, depth = info["per_protein"], info["depth"]
qs = list(T)

# ---------------------------------------------------------------- fold
for p in PIPES:
    t0 = time.time()
    if not (E / "out" / p).exists():
        subprocess.run(["colabfold_batch", str(E / "a3m" / p), str(E / "out" / p), "--num-models", "1", "--num-recycle", "3",
                        "--model-type", "alphafold2_ptm"], capture_output=True, text=True)
    for q in [] if (E / "retry" / p).exists() else qs:                                              # a crash takes the batch down: retry missing singly
        if not glob.glob(str(E / "out" / p / f"{q}_unrelaxed_rank_001*.pdb")):
            d = E / "retry" / p / q
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{q}.a3m").write_text((E / "a3m" / p / f"{q}.a3m").read_text())
            subprocess.run(["colabfold_batch", str(d), str(E / "out" / p), "--num-models", "1", "--num-recycle", "3",
                            "--model-type", "alphafold2_ptm"], capture_output=True, text=True)
    log(f"folded {p} in {time.time() - t0:.0f}s")


def fold_times(p):
    """Per-protein AlphaFold wall time = gap between consecutive finished predictions in the batch
    (ColabFold runs them one after another). The first protein's gap includes start-up, so it is dropped."""
    done = sorted((Path(f).stat().st_mtime, Path(f).name.split(".done")[0])
                  for f in glob.glob(str(E / "out" / p / "*.done.txt")))
    return {q: {"wall": t - t0, "model": np.nan} for (t0, _), (t, q) in zip(done, done[1:])}


def tm_score(pred, ref):
    o = subprocess.run(["USalign", str(pred), str(ref), "-outfmt", "2"], capture_output=True, text=True).stdout
    return float([l for l in o.splitlines() if l and not l.startswith("#")][0].split("\t")[3])


def neff(a3m):
    rows = [l for l in a3m.splitlines() if l and not l.startswith(">")]
    rows = ["".join(c for c in r if not c.islower()) for r in rows]
    if len(rows) < 2:
        return float(len(rows))
    A = np.array([list(r) for r in rows])
    X = np.stack([(A == c) for c in "ACDEFGHIKLMNPQRSTVWY"], -1).reshape(len(rows), -1).astype(np.float32)
    nong = (A != "-").sum(1).astype(np.float32)
    S = X @ X.T / np.maximum(np.minimum.outer(nong, nong), 1)
    return float((1 / (S >= 0.8).sum(1)).sum())


R, FT = {}, {}
for p in PIPES:
    FT[p] = fold_times(p)
    for q in qs:
        pdb = sorted(glob.glob(str(E / "out" / p / f"{q}_unrelaxed_rank_001*.pdb")))
        js = sorted(glob.glob(str(E / "out" / p / f"{q}_scores_rank_001*.json")))
        if pdb and js:
            R[(p, q)] = {"tm": tm_score(pdb[0], E / "ref" / f"{q}.ent"),
                         "plddt": float(np.mean(json.loads(Path(js[0]).read_text())["plddt"])),
                         "neff": neff((E / "a3m" / p / f"{q}.a3m").read_text())}
        else:
            log(f"  {p} {q}: no prediction (ColabFold failed)")
ok = [q for q in qs if all((p, q) in R for p in PIPES)]


def stages(p, q):
    t = T[q]
    g = lambda *ks: sum(t.get(k, 0.0) for k in ks)
    mm = (g("mmseqs2: search + align (3 rounds)"), g("mmseqs2: build MSA (ColabFold filters)"))
    if p == "jackhmmer":
        s = {"prep": 0, "search": g("jackhmmer: search + align"), "align": 0, "msa": g("jackhmmer: build MSA")}
    elif p == "mmseqs2":
        s = {"prep": 0, "search": mm[0], "align": 0, "msa": mm[1]}
    else:
        k = "yours" if p == "yours" else "hybrid-part"
        s = {"prep": g(f"{k}: embed query"), "search": g(f"{k}: window index search"),
             "align": g(f"{k}: calibration (200 random)", f"{k}: align 200 candidates"), "msa": g(f"{k}: build MSA rows")}
        if p == "hybrid":
            s["search"] += mm[0]
            s["msa"] += mm[1]
    s["to_msa"] = sum(s.values())
    f = FT[p].get(q, {})
    s["fold"], s["fold_model"] = f.get("wall", np.nan), f.get("model", np.nan)
    s["total"] = s["to_msa"] + s["fold"]
    return s


def sign_p(w, l):
    n = w + l
    return min(1.0, sum(comb(n, k) for k in range(max(w, l), n + 1)) / 2 ** n * 2) if n else 1.0


def perm_p(d, rng=np.random.default_rng(0)):
    return float(np.mean([abs((d * rng.choice([-1, 1], len(d))).mean()) >= abs(d.mean()) - 1e-12 for _ in range(20000)]))


S = {p: [stages(p, q) for q in ok] for p in PIPES}
md = lambda p, k: np.nanmedian([s[k] for s in S[p]])
L = [f"# End-to-end: sequence -> MSA -> AlphaFold structure, {len(ok)} fresh hard proteins", "",
     "Database: Swiss-Prot (576k proteins) for every pipeline. AlphaFold: ColabFold, 1 model, 3 recycles, no templates.",
     "Each protein run start to finish, one at a time, after a warm-up. JackHMMER & MMseqs2: 6 CPU threads, each tool's own",
     "alignment/MSA output. BinPLM: GPU + CPU. Hybrid times are sequential (MMseqs2, then BinPLM).", "",
     "## Time per protein (median seconds)", "",
     "| pipeline | query prep | search | align + calibrate | build MSA | **sequence -> MSA** | AlphaFold | **total** |",
     "|---|---|---|---|---|---|---|---|"]
for p in PIPES:
    L.append(f"| {NAMES[p]} | {md(p, 'prep'):.2f} | {md(p, 'search'):.2f} | {md(p, 'align'):.2f} | {md(p, 'msa'):.2f} | "
             f"**{md(p, 'to_msa'):.2f}** | {md(p, 'fold'):.0f} | **{md(p, 'total'):.0f}** |")
L += ["", "JackHMMER and MMseqs2 align inside their search step. One-time setup (not in the table): JackHMMER none; "
      "MMseqs2 index 21 s; BinPLM: Swiss-Prot embedding ~1.7 h GPU + window index ~1.5 min + 256-bit codes ~4 min; "
      f"ESM-2 model load {info['setup']['esm_model_load_s']:.0f} s per session.", "",
      "## Accuracy", "",
      "| pipeline | mean TM | median TM | correct folds (TM>0.5) | mean pLDDT | MSA rows (median) | Neff 80% (median) |",
      "|---|---|---|---|---|---|---|"]
for p in PIPES:
    tm = np.array([R[(p, q)]["tm"] for q in ok])
    L.append(f"| {NAMES[p]} | {tm.mean():.3f} | {np.median(tm):.3f} | {(tm > 0.5).sum()}/{len(ok)} | "
             f"{np.mean([R[(p, q)]['plddt'] for q in ok]):.1f} | {np.median([depth[q][p] for q in ok]):.0f} | "
             f"{np.median([R[(p, q)]['neff'] for q in ok]):.1f} |")
L += ["", "## Head to head (TM-score; better/worse = difference > 0.05)", "",
      "| A vs B | mean TM difference (A - B) | A better | B better | sign-test p | permutation p |", "|---|---|---|---|---|---|"]
for a, b in combinations(PIPES, 2):
    d = np.array([R[(a, q)]["tm"] - R[(b, q)]["tm"] for q in ok])
    w, l = int((d > 0.05).sum()), int((d < -0.05).sum())
    L.append(f"| {NAMES[a]} vs {NAMES[b]} | {d.mean():+.3f} | {w} | {l} | {sign_p(w, l):.2f} | {perm_p(d):.3f} |")
L += ["", "## Per protein (TM-score)", "", "| protein | length | " + " | ".join(PIPES) + " | rows: " + " / ".join(PIPES) + " |",
      "|---|---|" + "---|" * len(PIPES) + "---|"]
qlen = dict(l.split("\t")[:3:2] for l in (E / "queries.tsv").read_text().splitlines()[1:])
for q in ok:
    L.append(f"| {q} | {qlen.get(q, '')} | " + " | ".join(f"{R[(p, q)]['tm']:.2f}" for p in PIPES) + " | "
             + " / ".join(str(depth[q][p]) for p in PIPES) + " |")
(RESULTS / "e2e.md").write_text("\n".join(L) + "\n")
(RESULTS / "e2e.json").write_text(json.dumps({"stages": {p: dict(zip(ok, S[p])) for p in PIPES},
                                              "accuracy": {f"{p}|{q}": R[(p, q)] for p in PIPES for q in ok}}, indent=1))
log("\n".join(L))
