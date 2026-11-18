"""Test A, stage 2: how deep is each candidate's big-database MSA (ColabFold server)?
Neff = number of effective sequences (each sequence weighted by 1 / number of sequences >=80% identical to it), the
standard measure; AlphaFold accuracy drops sharply below ~30.  Pre-registered rule: shallow = Neff < 30; if fewer than
10 qualify, take the 20 lowest-Neff proteins instead.  Writes e2e/dark/depth.tsv and e2e/dark/shallow.txt."""
from pathlib import Path

import numpy as np

from common import DATA, read_fasta

D = DATA / "e2e" / "dark"
ids, _ = read_fasta(D / "candidates.fasta")


def neff(path, cap=4000):
    rows = [l for l in Path(path).read_text().replace("\x00", "").splitlines() if l and not l.startswith((">", "#"))]
    rows = ["".join(c for c in r if not c.islower()) for r in rows]
    n = len(rows)
    if n > cap:
        return n, float("inf")                           # certainly deep
    A = np.array([list(r) for r in rows])
    X = np.stack([(A == c) for c in "ACDEFGHIKLMNPQRSTVWY"], -1).reshape(n, -1).astype(np.float32)
    nong = (A != "-").sum(1).astype(np.float32)
    S = X @ X.T / np.maximum(np.minimum.outer(nong, nong), 1)
    return n, float((1 / (S >= 0.8).sum(1)).sum())


out, res = [], {}
for q in ids:
    f = D / "msa" / f"{q}.a3m"
    if f.exists():
        res[q] = neff(f)
        out.append(f"{q}\t{res[q][0]}\t{res[q][1]:.1f}")
(D / "depth.tsv").write_text("sid\trows\tneff80\n" + "\n".join(out) + "\n")
shallow = [q for q, (n, e) in res.items() if e < 30]
rule = "Neff < 30"
if len(shallow) < 10:
    shallow = sorted(res, key=lambda q: res[q][1])[:20]
    rule = "fewer than 10 had Neff < 30 -> the 20 lowest-Neff"
(D / "shallow.txt").write_text("\n".join(shallow) + "\n")
ne = np.array([e for _, e in res.values()])
print(f"screened {len(res)}/{len(ids)}; Neff<30: {(ne < 30).sum()}, <100: {(ne < 100).sum()}, "
      f">=100: {(ne >= 100).sum()}; shallow set ({rule}): {len(shallow)}")
print("DEPTH DONE")
