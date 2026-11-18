"""Test A preview check 1: refold the 7 preview proteins with 3 more random seeds (1,2,3) per MSA version, to see how
much TM-score moves by chance. Run in the 'cf' env. Writes results/A_preview_seeds.md."""
import glob, re, subprocess
import numpy as np
from common import DATA, RESULTS
E = DATA / "e2e"; D = E / "dark"
SETS = ["bigdb", "bigdb_yours", "bigdb_atlas", "bigdb_both"]
qs = (D / "shallow_preview.txt").read_text().split()
CF = ["--num-models", "1", "--num-recycle", "3", "--model-type", "alphafold2_ptm", "--random-seed", "1", "--num-seeds", "3"]
for p in SETS:
    subprocess.run(["colabfold_batch", str(D / "a3m" / p), str(D / "seeds" / p), *CF], capture_output=True, text=True)
tm = lambda f, q: float([l for l in subprocess.run(["USalign", f, str(E / "ref" / f"{q}.ent"), "-outfmt", "2"],
                         capture_output=True, text=True).stdout.splitlines() if l and not l.startswith("#")][0].split("\t")[3])
T = {}
for p in SETS:
    for q in qs:
        f0 = sorted(glob.glob(str(D / "out" / p / f"{q}_unrelaxed_rank_001*.pdb")))
        fs = sorted(glob.glob(str(D / "seeds" / p / f"{q}_unrelaxed_rank_*seed*.pdb")))
        T[p, q] = [tm(f0[0], q)] + [tm(f, q) for f in fs]
L = ["# Test A preview: seed check (TM-score for seed 0 = original, then seeds 1-3)", "",
     "| protein | " + " | ".join(SETS) + " |", "|---|" + "---|" * len(SETS)]
for q in qs:
    L.append(f"| {q} | " + " | ".join(" / ".join(f"{t:.2f}" for t in T[p, q]) + f" (mean {np.mean(T[p, q]):.2f})" for p in SETS) + " |")
L += ["", "| MSA | mean TM over all seeds | correct folds (of 7 x seeds) |", "|---|---|---|"]
for p in SETS:
    a = np.concatenate([T[p, q] for q in qs])
    L.append(f"| {p} | {a.mean():.3f} | {(a > 0.5).sum()}/{len(a)} |")
(RESULTS / "A_preview_seeds.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
