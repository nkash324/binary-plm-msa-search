"""Step 1b: re-tune the embedding-SW bias/gap WITHOUT SCOPe (the old values were tuned on 150 SCOPe40 pairs, which
are inside the benchmark set).

Pairs: random Swiss-Prot proteins (80-500 aa) and one MMseqs2 hit each (3 iterations, e < 1e-3) at 15-30% identity.
Swiss-Prot entries >= 90% identical (over >= 50% length) to any SCOPe40 domain are excluded on both sides.
Truth: US-align of the two AlphaFold DB models (pairs with TM >= 0.5 and >= 20 aligned residues).
Grid = the one in phase1/09_align_check.py; tuned on half the pairs, reported on the other half.
Writes poc/emb_align_params_sprot.json and results/retune.md.  Run in the 'fold' env (needs network)."""
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import emb_align as ea
import structref as sr
from common import DATA, DB_FASTA, PHASE1, RESULTS, read_fasta

N_Q, N_PAIRS = int(os.environ.get("N_Q", 3000)), int(os.environ.get("N_PAIRS", 400))
W = DATA / "retune_work"
W.mkdir(exist_ok=True)
rng = np.random.default_rng(11)
log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def sh(*a):
    subprocess.run([str(x) for x in a], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


db = DATA / "mmseqs_work" / "db"
ids, seqs = read_fasta(DB_FASTA)
id2i = {d: i for i, d in enumerate(ids)}
off = np.load(DATA / "offsets.npy")
bits = np.load(DATA / "res_bits.npy", mmap_mode="r")
t0 = time.time()

# near-copies of SCOPe40 domains
if not (W / "scope_ident.m8").exists():
    sh("mmseqs", "createdb", PHASE1 / "seqs.fasta", W / "scope")
    sh("mmseqs", "search", W / "scope", db, W / "sres", W / "tmp", "--min-seq-id", 0.9, "-c", 0.5, "-s", 4,
       "--threads", 3, "--split-memory-limit", "2G")
    sh("mmseqs", "convertalis", W / "scope", db, W / "sres", W / "scope_ident.m8")
excl = {l.split("\t")[1] for l in open(W / "scope_ident.m8")}
log(f"{len(excl)} Swiss-Prot entries are near-copies of SCOPe40 domains (excluded) ({time.time() - t0:.0f}s)")

ok = [i for i, s in enumerate(seqs) if 80 <= len(s) <= 500 and ids[i] not in excl]
qs = sorted(rng.choice(ok, N_Q, replace=False))
if not (W / "pairs.m8").exists():
    (W / "q.fa").write_text("".join(f">{ids[i]}\n{seqs[i]}\n" for i in qs))
    sh("mmseqs", "createdb", W / "q.fa", W / "q")
    sh("mmseqs", "search", W / "q", db, W / "res", W / "tmp", "--num-iterations", 3, "-s", 7.5, "-e", 1e-3,
       "--threads", 3, "--split-memory-limit", "2G")
    sh("mmseqs", "convertalis", W / "q", db, W / "res", W / "pairs.m8", "--format-output", "query,target,fident")
cands = {}
for l in open(W / "pairs.m8"):
    q, t, f = l.split("\t")
    f = float(f)
    if q != t and t not in excl and 0.15 <= f <= 0.30 and 80 <= len(seqs[id2i[t]]) <= 500:
        cands.setdefault(q, []).append(t)
qk = sorted(cands)
rng.shuffle(qk)
pairs = [(q, cands[q][rng.integers(len(cands[q]))]) for q in qk[:N_PAIRS]]
log(f"{len(cands)} queries with a 15-30% identity hit; using {len(pairs)} pairs ({time.time() - t0:.0f}s)")

accs = sorted({a for p in pairs for a in p})
with ThreadPoolExecutor(2) as ex:
    got = dict(zip(accs, ex.map(sr.afdb_pdb, accs)))
log(f"AFDB models: {sum(v is not None for v in got.values())}/{len(accs)} ({time.time() - t0:.0f}s)")

keep, refs = [], []
for a, b in pairs:
    if got[a] is None or got[b] is None:
        continue
    ref, tm = sr.reference(got[a], got[b], seqs[id2i[a]], seqs[id2i[b]])
    if tm >= 0.5 and len(ref) >= 20:
        keep.append((a, b))
        refs.append(ref)
log(f"{len(keep)} pairs with TM >= 0.5 and >= 20 aligned residues")

code = lambda a: ea.pm1(bits[off[id2i[a]]:off[id2i[a] + 1]])
E = {a: code(a) for p in keep for a in p}
BIASES, GAPS = (0.0, 0.025, 0.05, 0.1, 0.15, 0.2), (0.03, 0.06, 0.1, 0.25, 0.5)
res = {}
for bias in BIASES:
    for gap in GAPS:
        res[(bias, gap)] = [sr.score(ea.sw_many(E[a], [E[b]], bias, gap, keep=True)[1][0], r)
                            for (a, b), r in zip(keep, refs)]
bl = [sr.score(sr.blosum_pairs(seqs[id2i[a]], seqs[id2i[b]]), r) for (a, b), r in zip(keep, refs)]
tune, test = list(range(0, len(keep), 2)), list(range(1, len(keep), 2))
mf1 = lambda r, ks: float(np.mean([r[k][2] for k in ks]))
best = max(res, key=lambda p: mf1(res[p], tune))
old = json.loads((DATA / "emb_align_params.json").read_text())
summ = lambda r: {m: float(np.mean([r[k][n] for k in test])) for n, m in enumerate(("sensitivity", "precision", "f1"))}
out = {"bits": 640, "bias": best[0], "gap": best[1], "pairs": len(keep), "test_pairs": len(test),
       "new": summ(res[best]), "old_params": summ(res[(old["bias"], old["gap"])]), "blosum": summ(bl),
       "source": "Swiss-Prot/AFDB pairs, no SCOPe"}
(DATA / "emb_align_params_sprot.json").write_text(json.dumps(out, indent=1))
L = ["# Step 1b: leak-free embedding-SW parameters (Swiss-Prot / AFDB pairs, no SCOPe)", "",
     f"{len(keep)} pairs (15-30% identity MMseqs2 hits, TM >= 0.5); tuned on {len(tune)}, tested on {len(test)}.", "",
     "| aligner | F1 | sensitivity | precision |", "|---|---|---|---|",
     f"| embedding SW, re-tuned (bias {best[0]}, gap {best[1]}) | {out['new']['f1']:.3f} | "
     f"{out['new']['sensitivity']:.3f} | {out['new']['precision']:.3f} |",
     f"| embedding SW, old SCOPe-tuned (bias {old['bias']}, gap {old['gap']}) | {out['old_params']['f1']:.3f} | "
     f"{out['old_params']['sensitivity']:.3f} | {out['old_params']['precision']:.3f} |",
     f"| BLOSUM62 SW | {out['blosum']['f1']:.3f} | {out['blosum']['sensitivity']:.3f} | {out['blosum']['precision']:.3f} |"]
(RESULTS / "retune.md").write_text("\n".join(L) + "\n")
log("\n".join(L))
shutil.rmtree(W / "tmp", ignore_errors=True)
