"""Test B report: every SCOPe ranking scored (a) on all queries it covers and (b) on the SAME fixed 1000-query subset
(the pLM-BLAST queries), with memory per protein and time.  Same metric definitions as phase1/05_evaluate.py.
Writes poc/results/B_benchmark.md."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("p1common", "/mnt/c/Folding/phase1/common.py")
p1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p1)
meta = p1.load_meta()
fam, sf, fold = meta["fam"], meta["sf"], meta["fold"]
N = len(fam)
SUB = np.sort(np.random.default_rng(7).choice(N, 1000, replace=False))


def counts(lbl):
    _, inv, c = np.unique(lbl, return_inverse=True, return_counts=True)
    return c[inv]


n_fam = counts(fam) - 1
n_sf_all = counts(sf) - 1
n_sf = n_sf_all - n_fam
n_fold = counts(fold) - 1 - n_sf_all


def evaluate(path, queries):
    z = np.load(path)
    idx, off = z["idx"], z["off"]
    s_fam, s_sf, s_fold, rr = [], [], [], []
    for q in queries:
        r = idx[off[q]:off[q + 1]]
        r = r[r != q]
        same_fam, same_sf, same_fold = fam[r] == fam[q], sf[r] == sf[q], fold[r] == fold[q]
        fp = np.flatnonzero(~same_fold)
        cut = fp[0] if len(fp) else len(r)
        if n_fam[q]:
            s_fam.append(same_fam[:cut].sum() / n_fam[q])
        if n_sf[q]:
            s_sf.append((same_sf & ~same_fam)[:cut].sum() / n_sf[q])
            rr.append((same_sf & ~same_fam)[:1000].sum() / n_sf[q])
        if n_fold[q]:
            s_fold.append((same_fold & ~same_sf)[:cut].sum() / n_fold[q])
    return np.mean(s_fam), np.mean(s_sf), np.mean(s_fold), np.mean(rr), float(z["bytes_per_protein"]), float(z["seconds"])


KEEP = ["window-64-256", "twostage-K1000", "residue-center-diag", "protein-pca-256", "jackhmmer", "mmseqs2_iter3",
        "dhr", "plmsearch_nopfam", "plmblast"]
NOTE = {"window-64-256": "yours: window index (search only)", "twostage-K1000": "yours: window + embedding alignment",
        "residue-center-diag": "yours: full per-residue codes (slow upper bound)", "protein-pca-256": "yours: 32 B/protein",
        "jackhmmer": "letters", "mmseqs2_iter3": "letters (ColabFold-style)", "dhr": "DHR", "plmsearch_nopfam": "PLMSearch",
        "plmblast": "pLM-BLAST (ProtT5 per residue)"}
L = ["# Test B: remote-relative search on SCOPe 2.08 (15,177 domains): BinPLM vs pLM-BLAST, PLMSearch, DHR", "",
     "Sensitivity = share of a query's true relatives ranked above its first wrong-fold hit (standard SCOPe benchmark).",
     "Superfamily = remote relatives (same superfamily, different family). Subset = the same 1000 random queries for "
     "every method (pLM-BLAST was only run on these).", "",
     "| method | what | superfamily sens. (subset) | family sens. (subset) | fold sens. (subset) | remote recall@1000 (subset) "
     "| superfamily sens. (all queries) | bytes / protein | time (s) |", "|---|---|---|---|---|---|---|---|---|"]
rows = []
for m in KEEP:
    f = p1.RANKINGS / f"{m}.npz"
    if not f.exists():
        continue
    z = np.load(f)
    covered = np.flatnonzero(np.diff(z["off"]) > 0)
    sub = evaluate(f, SUB)
    full = evaluate(f, covered)[1] if len(covered) > 2000 else float("nan")
    rows.append((sub[1], f"| {m} | {NOTE[m]} | **{sub[1]:.3f}** | {sub[0]:.3f} | {sub[2]:.3f} | {sub[3]:.3f} | "
                         f"{full:.3f} | {sub[4]:,.0f} | {sub[5]:.0f} |"))
L += [r for _, r in sorted(rows, key=lambda x: -x[0])]
L += ["", "Time = total for the whole benchmark as each method was run (pLM-BLAST: 1000 queries only; not directly "
      "comparable). Bytes/protein = stored search data per database protein."]
out = Path("/mnt/c/Folding/poc/results/B_benchmark.md")
out.write_text("\n".join(L) + "\n")
print("\n".join(L))
