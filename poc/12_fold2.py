"""Step 5-6: fold the 20 hardest proteins with 6 different MSAs and score against the real structure.

  A jackhmmer_af2       AlphaFold's own search
  B mmseqs_cf           ColabFold's search                       (main baseline)
  C mmseqs_plus_emb     B + relatives found & aligned by the binary index / embedding aligner   (the test)
  D emb_only            only the embedding relatives
  E mmseqs_plus_decoy   B + the same number of look-alike random proteins, aligned the same way (control)
  F single_sequence     no MSA (floor)
ColabFold, 1 model, 3 recycles, no templates (same as 07_fold.py). Folds that are identical to an
already-run MSA are reused. Run in the 'cf' env.
"""
import glob
import hashlib
import json
import shutil
import subprocess
import time
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

from common import DATA, Q_FASTA, RESULTS, read_fasta

N_Q = int(__import__("os").environ.get("N_Q", 20))
CONDS = ["jackhmmer_af2", "mmseqs_cf", "mmseqs_plus_emb", "emb_only", "mmseqs_plus_decoy", "single_sequence"]
VERIFY, EMB = DATA / "verify_work", DATA / "embmsa"
F = DATA / "fold"
A3M, OUT, REF, OLD = F / "a3m2", F / "out2", F / "ref", F / "out"
LOG = open(RESULTS / "align_fold.log", "a")


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


q_ids, q_seqs = read_fasta(Q_FASTA)
qseq = dict(zip(q_ids, q_seqs))
summ = json.loads((RESULTS / "poc_results.json").read_text())["per_query"]
ref_neff = {r["q"]: r["neff"] for r in summ["phmmer_max"]}
hard = sorted(q_ids, key=lambda q: ref_neff[q])[:N_Q]


def sto_rows(path, L):
    if not path.exists() or not path.stat().st_size:
        return []
    seqs, rf = defaultdict(str), ""
    for line in open(path):
        if line.startswith("#=GC RF"):
            rf += line.split()[-1]
        elif line.strip() and not line.startswith(("#", "//")):
            n, s = line.split()
            seqs[n] += s
    keep = [i for i, c in enumerate(rf) if c == "x"]
    rows = ["".join(s[i] for i in keep).upper().replace(".", "-") for s in seqs.values()]
    return [r for r in rows if len(r) == L]


def tm_score(pred, ref):
    out = subprocess.run(["USalign", str(pred), str(ref), "-outfmt", "2"], capture_output=True, text=True).stdout
    return float([l for l in out.splitlines() if l and not l.startswith("#")][0].split("\t")[3])


def find(q, c):
    for d in (OUT / q, OUT / f"{q}_{c}", OLD / q):
        pdb = sorted(glob.glob(str(d / f"{q}__{c}_unrelaxed_rank_001*.pdb")))
        js = sorted(glob.glob(str(d / f"{q}__{c}_scores_rank_001*.json")))
        if pdb and js:
            return pdb[0], js[0]
    return None


def colabfold(indir, outdir):
    return subprocess.run(["colabfold_batch", str(indir), str(outdir), "--num-models", "1", "--num-recycle", "3",
                           "--model-type", "alphafold2_ptm"], capture_output=True, text=True)


# ---- build MSAs; identical MSAs are folded once
msas, alias = {}, {}
for q in hard:
    L, e = len(qseq[q]), json.loads((EMB / f"{q}.json").read_text())
    jk, mm = sto_rows(VERIFY / f"jackhmmer_af2.{q}.sto", L), sto_rows(VERIFY / f"mmseqs_cf.{q}.sto", L)
    rows = {"jackhmmer_af2": jk, "mmseqs_cf": mm, "mmseqs_plus_emb": mm + e["rows_emb"], "emb_only": e["rows_emb"],
            "mmseqs_plus_decoy": mm + e["rows_decoy"], "single_sequence": []}
    seen = {}
    for c in CONDS:
        txt = f">{q}\n{qseq[q]}\n" + "".join(f">h{i}\n{r}\n" for i, r in enumerate(rows[c]))
        h = hashlib.md5(txt.encode()).hexdigest()
        msas[(q, c)] = (txt, len(rows[c]))
        alias[(q, c)] = seen.setdefault(h, c)

# ---- fold what's missing
t_all = time.time()
for n, q in enumerate(hard):
    todo = [c for c in CONDS if alias[(q, c)] == c and not find(q, c)]
    if todo:
        d = A3M / q
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        for c in todo:
            (d / f"{q}__{c}.a3m").write_text(msas[(q, c)][0])
        t0 = time.time()
        r = colabfold(d, OUT / q)
        for c in todo:                                   # a crash takes out the whole batch: retry singly
            if not find(q, c):
                d1 = A3M / f"{q}_{c}"
                shutil.rmtree(d1, ignore_errors=True)
                d1.mkdir(parents=True)
                (d1 / f"{q}__{c}.a3m").write_text(msas[(q, c)][0])
                r1 = colabfold(d1, OUT / f"{q}_{c}")
                if not find(q, c):
                    log(f"  {q} {c}: colabfold FAILED ({r1.returncode})")
        log(f"  [{n + 1}/{len(hard)}] {q}: folded {len(todo)} MSAs in {time.time() - t0:.0f}s")

# ---- score
rows = []
for q in hard:
    ref = REF / f"{q}.ent"
    for c in CONDS:
        hit = find(q, alias[(q, c)])
        if not hit:
            continue
        rows.append({"q": q, "condition": c, "msa_rows": msas[(q, c)][1], "same_as": alias[(q, c)],
                     "tm": tm_score(hit[0], ref), "plddt": float(np.mean(json.loads(Path(hit[1]).read_text())["plddt"]))})
(RESULTS / "fold2_results.json").write_text(json.dumps(rows, indent=1))

by = defaultdict(dict)
for x in rows:
    by[x["q"]][x["condition"]] = x
qs = [q for q in hard if all(c in by[q] for c in CONDS)]


def sign_p(w, l):
    n = w + l
    return sum(comb(n, k) for k in range(max(w, l), n + 1)) / 2 ** n * 2 if n else 1.0


lines = [f"# Folding test 2: {len(qs)} hardest proteins, 6 MSAs each (ColabFold, 1 model, no templates)", "",
         "TM-score vs the real structure (>0.5 = correct fold). Wins/losses vs MMseqs2 (B) count differences > 0.05.", "",
         "| MSA | mean TM | median TM | correct folds | mean pLDDT | mean MSA rows | better than B | worse than B | sign-test p |",
         "|---|---|---|---|---|---|---|---|---|"]
for c in CONDS:
    tm = np.array([by[q][c]["tm"] for q in qs])
    b = np.array([by[q]["mmseqs_cf"]["tm"] for q in qs])
    w, l = int((tm - b > 0.05).sum()), int((b - tm > 0.05).sum())
    lines.append(f"| {c} | {tm.mean():.3f} | {np.median(tm):.3f} | {(tm > 0.5).sum()}/{len(qs)} | "
                 f"{np.mean([by[q][c]['plddt'] for q in qs]):.1f} | {np.mean([by[q][c]['msa_rows'] for q in qs]):.0f} | "
                 f"{w} | {l} | {sign_p(w, l):.2f} |")
lines += ["", "| protein | MMseqs2 rows | +emb rows | TM B | TM C (+emb) | TM E (+decoy) | TM D (emb only) | TM F (single) |",
          "|---|---|---|---|---|---|---|---|"]
for q in qs:
    g = lambda c: by[q][c]["tm"]
    lines.append(f"| {q} | {by[q]['mmseqs_cf']['msa_rows']} | "
                 f"{by[q]['mmseqs_plus_emb']['msa_rows'] - by[q]['mmseqs_cf']['msa_rows']} | {g('mmseqs_cf'):.2f} | "
                 f"{g('mmseqs_plus_emb'):.2f} | {g('mmseqs_plus_decoy'):.2f} | {g('emb_only'):.2f} | {g('single_sequence'):.2f} |")
(RESULTS / "fold2_results.md").write_text("\n".join(lines) + "\n")
log("\n".join(lines))
log(f"folding done in {(time.time() - t_all) / 60:.0f} min")
