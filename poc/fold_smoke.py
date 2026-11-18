"""Smoke test of the Part 3 chain on one query: phmmer MSA -> a3m -> ColabFold -> US-align."""
import os
import glob
import json
import subprocess
import sys
import time

sys.path.insert(0, "/mnt/c/Folding/poc")
sys.argv = ["x"]
import importlib.util
spec = importlib.util.spec_from_file_location("fold", "/mnt/c/Folding/poc/07_fold.py")
fold = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fold)                      # loads helpers only (main is guarded)

from common import DATA, DB_FASTA
q = "d1kr7a_"
seq = fold.qseq[q]
T = DATA / "fold_smoke"
(T / "in").mkdir(parents=True, exist_ok=True)
(T / "q.fa").write_text(f">{q}\n{seq}\n")
subprocess.run([os.path.expanduser("~") + "/miniforge3/envs/fold/bin/phmmer", "--cpu", "4", "-E", "1e-3", "--incE", "1e-3", "--incdomE", "1e-3", "-o", "/dev/null",
                "-A", str(T / "q.sto"), str(T / "q.fa"), str(DB_FASTA)], check=True)
rows = [r for r in fold.read_sto_rows(T / "q.sto") if len(r) == len(seq)]
print(f"MSA rows: {len(rows)}")
(T / "in" / f"{q}__msa.a3m").write_text(f">{q}\n{seq}\n" + "".join(f">h{i}\n{r}\n" for i, r in enumerate(rows)))
(T / "in" / f"{q}__single.a3m").write_text(f">{q}\n{seq}\n")
t = time.time()
r = subprocess.run(["colabfold_batch", str(T / "in"), str(T / "out"), "--num-models", "1", "--num-recycle", "3",
                    "--model-type", "alphafold2_ptm"], capture_output=True, text=True)
print(f"colabfold exit {r.returncode} in {time.time() - t:.0f}s")
if r.returncode:
    print(r.stderr[-3000:])
for c in ("msa", "single"):
    pdb = sorted(glob.glob(str(T / "out" / f"{q}__{c}_unrelaxed_rank_001*.pdb")))
    js = sorted(glob.glob(str(T / "out" / f"{q}__{c}_scores_rank_001*.json")))
    import numpy as np
    print(c, "TM-score vs real structure:", fold.tm_score(pdb[0], fold.ref_structure(q)),
          "| mean pLDDT:", round(float(np.mean(json.loads(open(js[0]).read())["plddt"])), 1))
