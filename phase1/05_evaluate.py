"""Score every ranking against the SCOPe hierarchy and write results/results.md + .csv.

Standard SCOPe homology benchmark definitions (as used by MMseqs2 / Foldseek papers), self excluded:
  family TP      = same family
  superfamily TP = same superfamily, different family   (remote homologs)
  fold TP        = same fold, different superfamily     (very remote / analogs)
  FP             = different fold
Sensitivity = fraction of a query's TPs ranked above its first FP, averaged over queries with >=1 TP.
Recall@k   = fraction of same-superfamily homologs (any family) in the top k, regardless of FPs.
             This is the "oversample then align" view relevant to building MSAs.
"""
import csv

import numpy as np

from common import RANKINGS, RESULTS, load_meta

meta = load_meta()
fam, sf, fold = meta["fam"], meta["sf"], meta["fold"]
N = len(fam)


def counts(lbl):
    _, inv, c = np.unique(lbl, return_inverse=True, return_counts=True)
    return c[inv]


n_fam = counts(fam) - 1
n_sf_all = counts(sf) - 1
n_sf = n_sf_all - n_fam
n_fold = counts(fold) - 1 - n_sf_all


def evaluate(path):
    z = np.load(path)
    idx, off = z["idx"], z["off"]
    s_fam, s_sf, s_fold, r100, r1000, rr1000 = [], [], [], [], [], []
    for q in range(N):
        r = idx[off[q]:off[q + 1]]
        r = r[r != q]
        same_fam = fam[r] == fam[q]
        same_sf = sf[r] == sf[q]
        same_fold = fold[r] == fold[q]
        fp = np.flatnonzero(~same_fold)
        cut = fp[0] if len(fp) else len(r)
        if n_fam[q]:
            s_fam.append(same_fam[:cut].sum() / n_fam[q])
        if n_sf[q]:
            s_sf.append((same_sf & ~same_fam)[:cut].sum() / n_sf[q])
            rr1000.append((same_sf & ~same_fam)[:1000].sum() / n_sf[q])
        if n_fold[q]:
            s_fold.append((same_fold & ~same_sf)[:cut].sum() / n_fold[q])
        if n_sf_all[q]:
            r100.append(same_sf[:100].sum() / n_sf_all[q])
            r1000.append(same_sf[:1000].sum() / n_sf_all[q])
    return {
        "method": path.stem,
        "sens_family": np.mean(s_fam), "sens_superfamily": np.mean(s_sf), "sens_fold": np.mean(s_fold),
        "recall@100": np.mean(r100), "recall@1000": np.mean(r1000), "remote_recall@1000": np.mean(rr1000),
        "bytes_per_protein": float(z["bytes_per_protein"]), "seconds": float(z["seconds"]),
    }


rows = sorted((evaluate(p) for p in RANKINGS.glob("*.npz")), key=lambda r: -r["sens_superfamily"])
cols = list(rows[0].keys())
with open(RESULTS / "results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

hdr = ("| Method | Family sens. | **Superfamily sens.** | Fold sens. | Recall@100 | Recall@1000 "
       "| Remote recall@1000 | Bytes/protein | Time (s) |")
lines = ["# Phase 1 results: SCOPe 2.08 40%, ESM-2 150M", "",
         f"{N} domains; queries with >=1 TP: family {int((n_fam > 0).sum())}, "
         f"superfamily {int((n_sf > 0).sum())}, fold {int((n_fold > 0).sum())}.", "",
         hdr, "|" + "---|" * 9]
for r in rows:
    lines.append(f"| {r['method']} | {r['sens_family']:.3f} | **{r['sens_superfamily']:.3f}** | "
                 f"{r['sens_fold']:.3f} | {r['recall@100']:.3f} | {r['recall@1000']:.3f} | "
                 f"{r['remote_recall@1000']:.3f} | {r['bytes_per_protein']:,.0f} | {r['seconds']:.0f} |")
(RESULTS / "results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
