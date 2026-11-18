"""Step 4: is the embedding aligner actually better at lining up DISTANT relatives?

300 random SCOPe pairs from the same superfamily but different families (remote homologs), excluding the
superfamilies of the PoC test proteins. The truth is the structural superposition (US-align): residue
pairs closer than 5 A. We compare:
  BLOSUM62 Smith-Waterman (classic letter-based alignment, gap open 11 / extend 1)
  Embedding Smith-Waterman (ESM-2 binary codes), with bias/gap tuned on half the pairs, tested on the other half.
Writes ~/folding-data/poc/emb_align_params.json with the tuned settings and a go/no-go verdict.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices

sys.path.append("/mnt/c/Folding/poc")
import emb_align as ea                                                   # noqa: E402
from common import DATA, load_meta, read_fasta                           # noqa: E402

N_PAIRS = int(os.environ.get("N_PAIRS", 300))
USALIGN = os.path.expanduser("~") + "/miniforge3/envs/cf/bin/USalign"
POC = Path.home() / "folding-data" / "poc"
TGZ = POC / "download" / "pdbstyle-sel-gs-bib-40-2.08.tgz"
ENT = DATA / "ent"
ENT.mkdir(exist_ok=True)
rng = np.random.default_rng(1)


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


meta = load_meta()
sid, fam, sf = meta["sid"], meta["fam"], meta["sf"]
_, seqs = read_fasta(DATA / "seqs.fasta")
off = np.load(DATA / "offsets.npy")
seqs = [s[:off[i + 1] - off[i]] for i, s in enumerate(seqs)]   # phase 1 truncated at 1022
BITS = int(os.environ.get("BITS", 640))                 # 640 = original codes; 256 / 128 = PCA-compressed
SUF = "" if BITS == 640 else f"_{BITS}"
res_bits = np.load(DATA / ("res_bits.npy" if BITS == 640 else f"res_bits{BITS}.npy"), mmap_mode="r")

# exclude the PoC test proteins' superfamilies
qsf = {".".join(l.split("\t")[1].split(".")[:3]) for l in open(POC / "queries.tsv").read().splitlines()[1:]}
sccs = meta["sccs"]
ok = np.array([".".join(s.split(".")[:3]) not in qsf and len(q) <= 500 for s, q in zip(sccs, seqs)])
pairs = set()
idx_by_sf = {}
for i in np.flatnonzero(ok):
    idx_by_sf.setdefault(sf[i], []).append(i)
groups = [g for g in idx_by_sf.values() if len(set(fam[g])) > 1]
while len(pairs) < N_PAIRS:
    g = groups[rng.integers(len(groups))]
    a, b = rng.choice(g, 2, replace=False)
    if fam[a] != fam[b]:
        pairs.add((min(a, b), max(a, b)))
pairs = sorted(pairs)
need = sorted({sid[i] for p in pairs for i in p})

# ---- structures
missing = [s for s in need if not (ENT / f"{s}.ent").exists()]
if missing:
    listing = POC / "download" / "all_paths.txt"
    if not listing.exists():
        subprocess.run(f"tar -tzf {TGZ} > {listing}", shell=True, check=True)
    want = {f"{s}.ent" for s in missing}
    paths = [p for p in listing.read_text().split() if p.rsplit("/", 1)[-1] in want]
    (ENT / "paths.txt").write_text("\n".join(paths) + "\n")
    subprocess.run(["tar", "-xzf", str(TGZ), "-C", str(ENT), "--strip-components=3", "-T", str(ENT / "paths.txt")],
                   check=True)
log(f"{len(pairs)} remote-homolog pairs, {len(need)} structures")

blosum = Align.PairwiseAligner(mode="local", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                               open_gap_score=-11, extend_gap_score=-1)
glob_al = Align.PairwiseAligner(mode="global", match_score=2, mismatch_score=-1, open_gap_score=-3,
                                extend_gap_score=-0.5)
STD = set("ACDEFGHIKLMNPQRSTVWY")
clean = lambda s: "".join(c if c in STD else "X" for c in s)


def blocks_to_pairs(al):
    out = []
    for (a0, a1), (b0, b1) in zip(*al.aligned):
        out += [(a0 + k, b0 + k) for k in range(a1 - a0)]
    return out


def atom_to_seqres(atom, seqres):
    m = {}
    for i, j in blocks_to_pairs(glob_al.align(atom, seqres)[0]):
        m[i] = j
    return m


def reference(a, b):
    out = subprocess.run([USALIGN, str(ENT / f"{sid[a]}.ent"), str(ENT / f"{sid[b]}.ent")],
                         capture_output=True, text=True).stdout.splitlines()
    k = next((i for i, l in enumerate(out) if l.startswith('(":"')), None)
    if k is None:                       # structure missing or unreadable: skip this pair
        return set()
    s1, mk, s2 = out[k + 1], out[k + 2], out[k + 3]
    ma, mb = atom_to_seqres(s1.replace("-", ""), seqs[a]), atom_to_seqres(s2.replace("-", ""), seqs[b])
    ia = ib = 0
    ref = set()
    for c1, m, c2 in zip(s1, mk.ljust(len(s1)), s2):
        if c1 != "-" and c2 != "-" and m == ":" and ia in ma and ib in mb:
            ref.add((ma[ia], mb[ib]))
        ia += c1 != "-"
        ib += c2 != "-"
    return ref


def score(pred, ref):
    pred = set(pred)
    hit = len(pred & ref)
    return hit / max(len(ref), 1), (hit / len(pred) if pred else 0.0)


t0 = time.time()
refs = [reference(a, b) for a, b in pairs]
keep = [k for k, r in enumerate(refs) if len(r) >= 20]
log(f"structural reference alignments in {time.time() - t0:.0f}s ({len(keep)} pairs with >=20 aligned residues)")

bl = [score(blocks_to_pairs(blosum.align(clean(seqs[pairs[k][0]]), clean(seqs[pairs[k][1]]))[0]), refs[k])
      for k in keep]

E = {i: ea.pm1(res_bits[off[i]:off[i + 1]]) for p in pairs for i in p}
BIASES, GAPS = (0.0, 0.025, 0.05, 0.1, 0.15, 0.2), (0.03, 0.06, 0.1, 0.25, 0.5)
emb = {}
for bias in BIASES:
    for gap in GAPS:
        emb[(bias, gap)] = [score(ea.sw_many(E[pairs[k][0]], [E[pairs[k][1]]], bias, gap, keep=True)[1][0], refs[k])
                            for k in keep]
log(f"embedding alignments ({len(BIASES) * len(GAPS)} settings) in {time.time() - t0:.0f}s")

f1 = lambda s, p: 2 * s * p / (s + p) if s + p else 0.0
tune, test = keep[0::2], keep[1::2]
pos = {k: n for n, k in enumerate(keep)}
mean_f1 = lambda res, ks: np.mean([f1(*res[pos[k]]) for k in ks])
best = max(emb, key=lambda p: mean_f1(emb[p], tune))
summ = lambda res, ks: {"sensitivity": float(np.mean([res[pos[k]][0] for k in ks])),
                        "precision": float(np.mean([res[pos[k]][1] for k in ks])),
                        "f1": float(mean_f1(res, ks))}
r_emb, r_bl = summ(emb[best], test), summ(bl, test)
go = r_emb["sensitivity"] >= r_bl["sensitivity"] + 0.05 and r_emb["f1"] >= r_bl["f1"] + 0.05
out = {"bits": BITS, "bias": best[0], "gap": best[1], "test_pairs": len(test), "embedding": r_emb, "blosum": r_bl, "go": bool(go)}
(POC / f"emb_align_params{SUF}.json").write_text(json.dumps(out, indent=1))
lines = ["# Alignment check: distant relatives (same superfamily, different family), SCOPe", "",
         f"Truth = structural superposition (US-align, residues < 5 A). {len(test)} held-out pairs.", "",
         "| aligner | correct residue pairs found (sensitivity) | pairs that are correct (precision) | F1 |",
         "|---|---|---|---|",
         f"| BLOSUM62 Smith-Waterman (letters) | {r_bl['sensitivity']:.3f} | {r_bl['precision']:.3f} | {r_bl['f1']:.3f} |",
         f"| Embedding Smith-Waterman, {BITS}-bit codes (bias {best[0]}, gap {best[1]}) | {r_emb['sensitivity']:.3f} | "
         f"{r_emb['precision']:.3f} | {r_emb['f1']:.3f} |", "",
         f"Verdict: {'GO - embedding aligner clearly better' if go else 'NO-GO - not clearly better, stop here'}"]
Path(f"/mnt/c/Folding/poc/results/align_check{SUF}.md").write_text("\n".join(lines) + "\n")
log("\n".join(lines))
