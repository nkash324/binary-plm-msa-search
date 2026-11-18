"""Part 3: does a better MSA give a better predicted structure?

For the 40 hardest queries (shallowest exhaustive-reference MSA), fold each with AlphaFold2
(ColabFold, 1 model, 3 recycles, no templates) using the MSA from each search method, plus
single-sequence as a floor, and score against the real experimental structure (TM-score, US-align).

Runs in the 'cf' conda env. Resumable: finished folds are skipped.
Usage: python 07_fold.py [n_queries]
"""
import glob
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import DATA, Q_FASTA, RESULTS, read_fasta

N_HARD = int(sys.argv[1]) if len(sys.argv) > 1 else 40
CONDITIONS = ["jackhmmer_af2", "mmseqs_cf", "emb_twostage", "hybrid_mmseqs_emb", "single_sequence"]
VERIFY = DATA / "verify_work"
FOLD = DATA / "fold"
A3M, OUT, REF = FOLD / "a3m", FOLD / "out", FOLD / "ref"
for d in (A3M, OUT, REF):
    d.mkdir(parents=True, exist_ok=True)
LOG = open(FOLD / "fold.log", "a")


def log(msg):
    print(msg, flush=True)
    LOG.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    LOG.flush()


q_ids, q_seqs = read_fasta(Q_FASTA)
qseq = dict(zip(q_ids, q_seqs))


def read_sto_rows(path):
    seqs, rf = defaultdict(str), ""
    for line in open(path):
        if line.startswith("#=GC RF"):
            rf += line.split()[-1]
        elif line.strip() and not line.startswith(("#", "//")):
            name, s = line.split()
            seqs[name] += s
    keep = [i for i, c in enumerate(rf) if c == "x"]
    return ["".join(s[i] for i in keep).upper().replace(".", "-") for s in seqs.values()]


def ref_structure(sid):
    p = REF / f"{sid}.ent"                    # extracted by get_structures.sh (SCOPe bulk download)
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def tm_score(pred, ref):
    out = subprocess.run(["USalign", str(pred), str(ref), "-outfmt", "2"], capture_output=True, text=True).stdout
    row = [l for l in out.splitlines() if l and not l.startswith("#")][0].split("\t")
    return float(row[3])          # TM-score normalized by the reference (chain 2)


if __name__ == "__main__":
    summ = json.loads((RESULTS / "poc_results.json").read_text())["per_query"]
    ref_neff = {r["q"]: r["neff"] for r in summ["phmmer_max"]}
    hard = sorted(q_ids, key=lambda q: ref_neff[q])[:N_HARD]
    log(f"{len(hard)} hardest queries (reference Neff {ref_neff[hard[0]]:.1f}..{ref_neff[hard[-1]]:.1f})")

    rows = []
    for n, q in enumerate(hard):
        qdir_in, qdir_out = A3M / q, OUT / q
        qdir_in.mkdir(exist_ok=True)
        depth = {}
        for c in CONDITIONS:
            sto = VERIFY / f"{c}.{q}.sto"
            msa = read_sto_rows(sto) if c != "single_sequence" and sto.exists() and sto.stat().st_size else []
            msa = [r for r in msa if len(r) == len(qseq[q])]
            depth[c] = len(msa)
            (qdir_in / f"{q}__{c}.a3m").write_text(f">{q}\n{qseq[q]}\n" + "".join(f">h{i}\n{r}\n" for i, r in enumerate(msa)))
        t0 = time.time()
        r = subprocess.run(["colabfold_batch", str(qdir_in), str(qdir_out), "--num-models", "1", "--num-recycle", "3",
                            "--model-type", "alphafold2_ptm"], capture_output=True, text=True)
        if r.returncode:
            log(f"  {q}: colabfold FAILED ({r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode})")
        try:
            ref = ref_structure(q)
        except Exception as e:                       # network hiccup: score later
            log(f"  {q}: reference download failed: {e}")
            ref = None
        for c in CONDITIONS:
            pdb = sorted(glob.glob(str(qdir_out / f"{q}__{c}_unrelaxed_rank_001*.pdb")))
            js = sorted(glob.glob(str(qdir_out / f"{q}__{c}_scores_rank_001*.json")))
            if not pdb or ref is None:
                continue
            rows.append({"q": q, "condition": c, "msa_rows": depth[c], "ref_neff": ref_neff[q],
                         "tm": tm_score(pdb[0], ref), "plddt": float(np.mean(json.loads(Path(js[0]).read_text())["plddt"]))})
        done = {x["condition"]: x["tm"] for x in rows if x["q"] == q}
        log(f"  [{n + 1}/{len(hard)}] {q} L={len(qseq[q])} {time.time() - t0:.0f}s  TM: " +
            "  ".join(f"{c}={done[c]:.2f}" for c in CONDITIONS if c in done))
        (RESULTS / "fold_results.json").write_text(json.dumps(rows, indent=1))

    # ---- summary
    by = defaultdict(dict)
    for x in rows:
        by[x["q"]][x["condition"]] = x
    qs = [q for q in by if all(c in by[q] for c in CONDITIONS)]
    lines = [f"# Folding test: {len(qs)} hardest queries, AlphaFold2 (ColabFold, 1 model, no templates)", "",
             "TM-score vs the real structure (>0.5 = correct fold). 'Better/worse vs JackHMMER' counts differences > 0.05.", "",
             "| MSA from | mean TM | median TM | correct folds (TM>0.5) | mean pLDDT | mean MSA rows | better vs JackHMMER | worse |",
             "|---|---|---|---|---|---|---|---|"]
    for c in CONDITIONS:
        tm = np.array([by[q][c]["tm"] for q in qs])
        base = np.array([by[q]["jackhmmer_af2"]["tm"] for q in qs])
        lines.append(f"| {c} | {tm.mean():.3f} | {np.median(tm):.3f} | {(tm > 0.5).sum()}/{len(qs)} | "
                     f"{np.mean([by[q][c]['plddt'] for q in qs]):.1f} | {np.mean([by[q][c]['msa_rows'] for q in qs]):.0f} | "
                     f"{(tm - base > 0.05).sum()} | {(base - tm > 0.05).sum()} |")
    (RESULTS / "fold_results.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
