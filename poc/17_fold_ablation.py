"""Does each speed-up change AlphaFold accuracy?  For every config, fold the 20 development proteins with
MMseqs2 rows + that config's embedding rows (condition C), same ColabFold settings as before.
If a config produced exactly the same MSA as the original run, the original fold is reused.
Run in the 'cf' env. Writes results/ablation.md (speed + relatives + accuracy, all configs)."""
import glob
import json
import shutil
import subprocess
import time
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

from common import DATA, Q_FASTA, RESULTS, read_fasta

F = DATA / "fold"
ABL, OUTA, A3MA = DATA / "abl", F / "out_abl", F / "a3m_abl"
LOG = open(RESULTS / "ablation.log", "a")
CFGS = ["base", "k200", "bits256", "bits128", "precal", "cal200", "combined"]


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


q_ids, q_seqs = read_fasta(Q_FASTA)
qseq = dict(zip(q_ids, q_seqs))
summ = json.loads((RESULTS / "poc_results.json").read_text())["per_query"]
ref_neff = {r["q"]: r["neff"] for r in summ["phmmer_max"]}
hard = sorted(q_ids, key=lambda q: ref_neff[q])[:20]
prev = {(x["q"], x["condition"]): x for x in json.loads((RESULTS / "fold2_results.json").read_text())}
orig = {q: json.loads((DATA / "embmsa" / f"{q}.json").read_text()) for q in hard}


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
    return [r for r in ("".join(s[i] for i in keep).upper().replace(".", "-") for s in seqs.values()) if len(r) == L]


def tm_score(pred, ref):
    out = subprocess.run(["USalign", str(pred), str(ref), "-outfmt", "2"], capture_output=True, text=True).stdout
    return float([l for l in out.splitlines() if l and not l.startswith("#")][0].split("\t")[3])


def find(d, q):
    pdb = sorted(glob.glob(str(d / f"{q}*_unrelaxed_rank_001*.pdb")))
    js = sorted(glob.glob(str(d / f"{q}*_scores_rank_001*.json")))
    return (pdb[0], js[0]) if pdb and js else None


def colabfold(i, o):
    subprocess.run(["colabfold_batch", str(i), str(o), "--num-models", "1", "--num-recycle", "3",
                    "--model-type", "alphafold2_ptm"], capture_output=True, text=True)


mm = {q: sto_rows(DATA / "verify_work" / f"mmseqs_cf.{q}.sto", len(qseq[q])) for q in hard}
tm = defaultdict(dict)
for cfg in CFGS:
    t0, nfold = time.time(), 0
    todo = []
    for q in hard:
        rows = json.loads((ABL / cfg / f"{q}.json").read_text())["rows_emb"]
        if not rows and (q, "mmseqs_cf") in prev:                               # no extra rows = MMseqs2 alone
            tm[cfg][q] = prev[(q, "mmseqs_cf")]["tm"]
            continue
        if rows == orig[q]["rows_emb"] and (q, "mmseqs_plus_emb") in prev:      # identical MSA -> reuse
            tm[cfg][q] = prev[(q, "mmseqs_plus_emb")]["tm"]
            continue
        todo.append((q, rows))
    if todo:
        d = A3MA / cfg
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        for q, rows in todo:
            (d / f"{q}.a3m").write_text(f">{q}\n{qseq[q]}\n" + "".join(f">h{i}\n{r}\n" for i, r in enumerate(mm[q] + rows)))
        colabfold(d, OUTA / cfg)
        for q, rows in todo:                                 # retry singly after a crash
            if not find(OUTA / cfg, q):
                d1 = A3MA / f"{cfg}_{q}"
                shutil.rmtree(d1, ignore_errors=True)
                d1.mkdir(parents=True)
                shutil.copy(d / f"{q}.a3m", d1)
                colabfold(d1, OUTA / cfg)
        for q, _ in todo:
            hit = find(OUTA / cfg, q)
            if hit:
                tm[cfg][q] = tm_score(hit[0], F / "ref" / f"{q}.ent")
                nfold += 1
            else:
                log(f"  {cfg} {q}: colabfold FAILED")
    log(f"{cfg}: folded {nfold}, reused {len(hard) - len(todo)} ({time.time() - t0:.0f}s)")

# ---------------------------------------------------------------- report
A = json.loads((RESULTS / "ablation.json").read_text())
qs = [q for q in hard if all(q in tm[c] for c in CFGS) and (q, "mmseqs_cf") in prev]
B = np.array([prev[(q, "mmseqs_cf")]["tm"] for q in qs])
base = np.array([tm["base"][q] for q in qs])


def sign_p(w, l):
    n = w + l
    return min(1.0, sum(comb(n, k) for k in range(max(w, l), n + 1)) / 2 ** n * 2) if n else 1.0


CAL = {"per_query": "per query, 2000 random", "pre": "pre-computed (one formula)", "small": "per query, 200 random + fitted tail"}
lines = ["# Speed-ups: what each one costs and buys", "",
         f"20 development proteins. Window search {A['_search_s']:.2f} s/protein in every config (identical). "
         "Data is read the same way in every config (memory-mapped file + OS cache); only the method's own settings change.", "",
         "## Speed (seconds per protein, search + alignment)", "",
         "| config | candidates | code size | chance level | cold | warm |", "|---|---|---|---|---|---|"]
for c in CFGS:
    r = A[c]
    lines.append(f"| {c} | {r['K']} | {r['bits']} bits ({r['bytes_per_aa']} B/aa) | {CAL[r['calibration']]} | "
                 f"{r['cold_total']:.2f} | {r['warm_total']:.2f} |")
lines += ["", "Warm-run breakdown (s/protein):", "", "| config | " + " | ".join(sorted({k for c in CFGS for k in A[c]['warm']})) + " |",
          "|---|" + "---|" * len({k for c in CFGS for k in A[c]['warm']})]
keys = sorted({k for c in CFGS for k in A[c]["warm"]})
for c in CFGS:
    lines.append(f"| {c} | " + " | ".join(f"{A[c]['warm'].get(k, 0):.2f}" for k in keys) + " |")
lines += ["", "## Accuracy", "",
          f"Alignment accuracy = F1 on 148 held-out SCOPe distant pairs (structure = truth). "
          f"TM = AlphaFold with MMseqs2 + embedding relatives, {len(qs)} proteins; MMseqs2 alone = {B.mean():.3f}.", "",
          "| config | alignment F1 | relatives accepted | of base's kept | new vs base | mean TM | correct folds | vs base: better / worse | p |",
          "|---|---|---|---|---|---|---|---|---|"]
f1 = {b: json.loads((DATA / ("emb_align_params.json" if b == 640 else f"emb_align_params_{b}.json")).read_text())["embedding"]["f1"]
      for b in (640, 256, 128)}
for c in CFGS:
    t = np.array([tm[c][q] for q in qs])
    w, l = int((t - base > 0.05).sum()), int((base - t > 0.05).sum())
    r = A[c]
    lines.append(f"| {c} | {f1[r['bits']]:.3f} | {r['accepted']} | {100 * r['kept_from_base']:.0f}% | {r['new_vs_base']} | "
                 f"{t.mean():.3f} | {(t > 0.5).sum()}/{len(qs)} | {w} / {l} | {sign_p(w, l):.2f} |")
(RESULTS / "ablation.md").write_text("\n".join(lines) + "\n")
json.dump({c: tm[c] for c in CFGS}, open(RESULTS / "ablation_tm.json", "w"), indent=1)
log("\n".join(lines))
