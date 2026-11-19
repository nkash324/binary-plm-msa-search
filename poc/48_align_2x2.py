"""Alignment correctness, 2x2 (retrieval source x aligner).

Question: does the per-residue EMBEDDING aligner line residues up better than a LETTER aligner, separately from which
search found the relative?  Truth everywhere = structural superposition (US-align), residue pairs < 5 A.

Part A (SCOPe distant pairs): the 148 held-out same-superfamily / different-family pairs of phase1/09_align_check.py
  (same random draw), scored for
    BLOSUM62 Smith-Waterman (gap 11/1)
    MMseqs2's own alignment (single-sequence, forced: --exhaustive-search, e-value cut off disabled)
    embedding Smith-Waterman, 256-bit codes (bias/gap tuned in 09, unchanged)
Part B (e2e targets): the 19 fresh hard proteins vs the relatives each search actually returned in the e2e run
    (a) MMseqs2 hits = rows of e2e/a3m/mmseqs2 (all), (b) BinPLM hits = rows of e2e/a3m/yours (<= 40 per target, random).
  Relative structures = AlphaFold DB models of the Swiss-Prot entries (cached in poc/afdb). Each pair scored for
    BLOSUM SW, embedding SW, and (MMseqs2 hits only) MMseqs2's own profile alignment from the a3m.
  Pairs are reported (i) where the relative really has the query's fold (US-align TM >= 0.5, normalised by the query)
  and (ii) all pairs with >= 20 structurally aligned residues.
Writes poc/results/align_2x2.md and align_2x2.json.  Run in the 'fold' env; needs network for AFDB models.
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from Bio import Align
from Bio.Align import substitution_matrices

sys.path.insert(0, "/mnt/c/Folding/poc")
import emb_align as ea                                                   # noqa: E402
from common import DATA, DB_FASTA, PHASE1, RESULTS, read_fasta           # noqa: E402

USALIGN = os.path.expanduser("~") + "/miniforge3/envs/cf/bin/USalign"
E = DATA / "e2e"
AFDB = DATA / "afdb"
AFDB.mkdir(exist_ok=True)
W = DATA / "align2x2_work"
W.mkdir(exist_ok=True)
MAX_YOURS = int(os.environ.get("MAX_YOURS", 40))
torch.cuda.set_per_process_memory_fraction(0.5)
PAUSE = DATA / "PAUSE"


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def wait_pause():
    while PAUSE.exists():
        time.sleep(10)


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


def blosum_pairs(a, b):
    return blocks_to_pairs(blosum.align(clean(a), clean(b))[0])


def atom_to_seq(atom, seq):
    if atom == seq:
        return {i: i for i in range(len(seq))}
    return dict(blocks_to_pairs(glob_al.align(clean(atom), clean(seq))[0]))


def reference(fa, fb, seqa, seqb):
    """US-align a (chain 1) onto b; residue pairs < 5 A in sequence coordinates, and TM normalised by a."""
    out = subprocess.run([USALIGN, str(fa), str(fb)], capture_output=True, text=True).stdout.splitlines()
    k = next((i for i, l in enumerate(out) if l.startswith('(":"')), None)
    tm = next((float(l.split()[1]) for l in out if l.startswith("TM-score=") and "Structure_1" in l), np.nan)
    if k is None:
        return set(), np.nan
    s1, mk, s2 = out[k + 1], out[k + 2], out[k + 3]
    ma, mb = atom_to_seq(s1.replace("-", ""), seqa), atom_to_seq(s2.replace("-", ""), seqb)
    ia = ib = 0
    ref = set()
    for c1, m, c2 in zip(s1, mk.ljust(len(s1)), s2):
        if c1 != "-" and c2 != "-" and m == ":" and ia in ma and ib in mb:
            ref.add((ma[ia], mb[ib]))
        ia += c1 != "-"
        ib += c2 != "-"
    return ref, tm


def score(pred, ref):
    pred = set(pred)
    hit = len(pred & ref)
    s, p = hit / max(len(ref), 1), (hit / len(pred) if pred else 0.0)
    return s, p, (2 * s * p / (s + p) if s + p else 0.0)


def summ(rows):
    a = np.array(rows) if rows else np.zeros((0, 3))
    return {"n": len(a), "sens": float(a[:, 0].mean()) if len(a) else np.nan,
            "prec": float(a[:, 1].mean()) if len(a) else np.nan, "f1": float(a[:, 2].mean()) if len(a) else np.nan}


def mmseqs_pairs(qseqs, tseqs, tag):
    """MMseqs2's own pairwise alignment for each (query i, target i), prefilter off, no e-value cut."""
    d = W / f"mm_{tag}"
    d.mkdir(exist_ok=True)
    (d / "q.fa").write_text("".join(f">q{i}\n{s}\n" for i, s in enumerate(qseqs)))
    (d / "t.fa").write_text("".join(f">t{i}\n{s}\n" for i, s in enumerate(tseqs)))
    subprocess.run(["mmseqs", "easy-search", d / "q.fa", d / "t.fa", d / "res.m8", d / "tmp", "--exhaustive-search", "1",
                    "-e", "1e9", "--min-ungapped-score", "0", "--max-seqs", "100000", "-a", "--threads", "3",
                    "--format-output", "query,target,qstart,tstart,qaln,taln"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = {}
    for l in open(d / "res.m8"):
        q, t, qs, ts, qa, ta = l.rstrip("\n").split("\t")
        if q[1:] != t[1:] or q in out:
            continue
        i, j, pr = int(qs) - 1, int(ts) - 1, []
        for a, b in zip(qa, ta):
            if a != "-" and b != "-":
                pr.append((i, j))
            i += a != "-"
            j += b != "-"
        out[q] = pr
    return [out.get(f"q{i}", []) for i in range(len(qseqs))]


R = {}
# ================================================================ Part A: SCOPe distant pairs
spec = importlib.util.spec_from_file_location("p1common", "/mnt/c/Folding/phase1/common.py")
p1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p1)
meta = p1.load_meta()
sid, fam, sf, sccs = meta["sid"], meta["fam"], meta["sf"], meta["sccs"]
_, s1 = read_fasta(PHASE1 / "seqs.fasta")
off1 = np.load(PHASE1 / "offsets.npy")
s1 = [s[:off1[i + 1] - off1[i]] for i, s in enumerate(s1)]
rb1 = np.load(PHASE1 / "res_bits256.npy", mmap_mode="r")
PRM1 = json.loads((DATA / "emb_align_params_256.json").read_text())
qsf = {".".join(l.split("\t")[1].split(".")[:3]) for l in (DATA / "queries.tsv").read_text().splitlines()[1:]}
ok = np.array([".".join(c.split(".")[:3]) not in qsf and len(q) <= 500 for c, q in zip(sccs, s1)])
rng, by_sf, pairs = np.random.default_rng(1), {}, set()
for i in np.flatnonzero(ok):
    by_sf.setdefault(sf[i], []).append(i)
groups = [g for g in by_sf.values() if len(set(fam[g])) > 1]
while len(pairs) < 300:
    g = groups[rng.integers(len(groups))]
    a, b = rng.choice(g, 2, replace=False)
    if fam[a] != fam[b]:
        pairs.add((min(a, b), max(a, b)))
pairs = sorted(pairs)
ENT1 = PHASE1 / "ent"
t0 = time.time()
refs = [reference(ENT1 / f"{sid[a]}.ent", ENT1 / f"{sid[b]}.ent", s1[a], s1[b])[0] for a, b in pairs]
keep = [k for k, r in enumerate(refs) if len(r) >= 20]
test = keep[1::2]                                       # the held-out half, exactly as in 09
log(f"Part A: {len(test)} held-out pairs (references {time.time() - t0:.0f}s)")
mmA = mmseqs_pairs([s1[pairs[k][0]] for k in test], [s1[pairs[k][1]] for k in test], "A")
resA = {"blosum": [], "mmseqs": [], "emb": []}
for n, k in enumerate(test):
    a, b = pairs[k]
    resA["blosum"].append(score(blosum_pairs(s1[a], s1[b]), refs[k]))
    resA["mmseqs"].append(score(mmA[n], refs[k]))
    Qa, Tb = ea.pm1(rb1[off1[a]:off1[a + 1]]), ea.pm1(rb1[off1[b]:off1[b + 1]])
    resA["emb"].append(score(ea.sw_many(Qa, [Tb], PRM1["bias"], PRM1["gap"], keep=True)[1][0], refs[k]))
R["A"] = {k: summ(v) for k, v in resA.items()}
R["A"]["mmseqs_no_alignment"] = int(sum(len(x) == 0 for x in mmA))
hasA = [n for n, x in enumerate(mmA) if x]               # pairs MMseqs2 did align: all three aligners on them
R["A_aligned"] = {k: summ([v[n] for n in hasA]) for k, v in resA.items()}
log(f"Part A: {json.dumps(R['A'])}")

# ================================================================ Part B: e2e targets vs each search's hits
db_ids, db_seqs = read_fasta(DB_FASTA)
id2idx = {d: i for i, d in enumerate(db_ids)}
off = np.load(DATA / "offsets.npy")
R256 = np.load(DATA / "res_bits256.npy", mmap_mode="r")
PRM = json.loads((DATA / "emb_align_params_256.json").read_text())
pca = np.load(DATA / "pca_res.npz")
gmu, gP = torch.from_numpy(pca["mu"]).cuda(), torch.from_numpy(pca["P"][:, :256].copy()).cuda()
mu_res = torch.from_numpy(np.load(DATA / "mu.npz")["res"]).cuda()
from transformers import AutoTokenizer, EsmModel                                   # noqa: E402
tok = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
esm = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D", add_pooling_layer=False,
                               attn_implementation="sdpa").eval().cuda().half()


@torch.no_grad()
def embed256(seq):
    enc = tok([seq], return_tensors="pt")
    r = esm(input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda()).last_hidden_state.float()
    sgn = r[0, 1:len(seq) + 1] - mu_res >= 0
    return (((sgn.half() * 2 - 1).float() - gmu) @ gP > 0).half() * 2 - 1


def read_a3m(path):
    ids, rows = read_fasta(path)
    return list(zip(ids[1:], rows[1:]))


def a3m_row_pairs(row, tseq):
    """(query_pos, target_pos) from an a3m row: uppercase/'-' = query columns, lowercase = insertions."""
    segment = "".join(c.upper() for c in row if c != "-")
    j = tseq.find(segment)
    if j < 0:
        return None
    i, pr = 0, []
    for c in row:
        if c == "-":
            i += 1
        elif c.islower():
            j += 1
        else:
            pr.append((i, j))
            i += 1
            j += 1
    return pr


def get(url, tries=4):
    for t in range(tries):                                  # AFDB answers 403/5xx when hit too fast: back off
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:                                   # noqa: BLE001
            pass
        time.sleep(2 * 2 ** t)
    return None


def afdb_pdb(acc):
    f = AFDB / f"{acc}.pdb"
    if f.exists() and f.stat().st_size:
        return f
    j = get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}")
    d = None
    try:
        d = get(json.loads(j)[0]["pdbUrl"]) if j else None
    except Exception:                                       # noqa: BLE001
        pass
    if not d:                                               # failures are not cached
        return None
    f.write_text(d)
    return f


qids, qseqs = read_fasta(E / "queries.fasta")
QS = [(q, s) for q, s in zip(qids, qseqs) if (E / "a3m" / "mmseqs2" / f"{q}.a3m").exists()]
sel = {}
for q, _ in QS:
    mm = read_a3m(E / "a3m" / "mmseqs2" / f"{q}.a3m")
    yo = read_a3m(E / "a3m" / "yours" / f"{q}.a3m")
    r = np.random.default_rng(0)
    yo = [yo[i] for i in sorted(r.choice(len(yo), min(MAX_YOURS, len(yo)), replace=False))] if yo else []
    sel[q] = {"mmseqs": mm, "yours": yo}
accs = sorted({a for v in sel.values() for k in v.values() for a, _ in k if a in id2idx})
t0 = time.time()
with ThreadPoolExecutor(2) as ex:
    got = dict(zip(accs, ex.map(afdb_pdb, accs)))
log(f"Part B: {len(QS)} targets, {len(accs)} relatives, AFDB models for {sum(v is not None for v in got.values())} "
    f"({time.time() - t0:.0f}s)")

PB = []
for q, seq in QS:
    wait_pause()
    Q = embed256(seq)
    for src in ("mmseqs", "yours"):
        for acc, row in sel[q][src]:
            if acc not in id2idx or got.get(acc) is None:
                continue
            p = id2idx[acc]
            tseq = db_seqs[p]
            ref, tm = reference(E / "ref" / f"{q}.ent", got[acc], seq, tseq)
            if len(ref) < 20:
                continue
            T = ea.pm1(np.array(R256[off[p]:off[p + 1]]))
            rec = {"q": q, "acc": acc, "src": src, "tm": tm, "nref": len(ref),
                   "blosum": score(blosum_pairs(seq, tseq), ref),
                   "emb": score(ea.sw_many(Q, [T], PRM["bias"], PRM["gap"], keep=True)[1][0], ref)}
            if src == "mmseqs":
                own = a3m_row_pairs(row, tseq)
                rec["own"] = score(own, ref) if own is not None else None
            PB.append(rec)
    log(f"  {q}: {sum(r['q'] == q for r in PB)} scored pairs")

R["B"] = {}
for subset, cond in (("same_fold", lambda r: r["tm"] >= 0.5), ("all", lambda r: True)):
    for src in ("mmseqs", "yours"):
        rs = [r for r in PB if r["src"] == src and cond(r)]
        d = {al: summ([r[al] for r in rs]) for al in ("blosum", "emb")}
        if src == "mmseqs":
            d["own"] = summ([r["own"] for r in rs if r.get("own") is not None])
        d["targets"] = len({r["q"] for r in rs})
        R["B"][f"{subset}/{src}"] = d
(RESULTS / "align_2x2.json").write_text(json.dumps({"summary": R, "pairs_B": PB}, indent=1, default=float))


def cell(d):
    return f"{d['f1']:.3f} ({d['sens']:.2f} / {d['prec']:.2f})" if d["n"] else "-"


L = ["# Alignment correctness, 2x2: retrieval source x aligner", "",
     "Truth = structural superposition (US-align), residue pairs closer than 5 A. Cells: **F1** (sensitivity / "
     "precision) of aligned residue pairs, averaged over pairs. Embedding SW = 256-bit PCA residue codes, bias "
     f"{PRM['bias']}, gap {PRM['gap']} (tuned earlier on the other half of the SCOPe pairs). BLOSUM SW = BLOSUM62, gap "
     "open 11 / extend 1.", "",
     f"## A. SCOPe distant pairs (same superfamily, different family; {R['A']['emb']['n']} held-out pairs)", "",
     "| aligner | F1 | sensitivity | precision |", "|---|---|---|---|"]
for k, name in (("blosum", "BLOSUM62 Smith-Waterman"), ("mmseqs", "MMseqs2 own alignment (pairwise, forced)"),
                ("emb", "Embedding Smith-Waterman")):
    d = R["A"][k]
    L.append(f"| {name} | {d['f1']:.3f} | {d['sens']:.3f} | {d['prec']:.3f} |")
L += ["", f"MMseqs2 returned no alignment for {R['A']['mmseqs_no_alignment']} of the pairs even with the e-value cut "
      "off and ungapped prefilter disabled (scored as 0 found). On the "
      f"{R['A_aligned']['emb']['n']} pairs it did align, F1: BLOSUM SW {R['A_aligned']['blosum']['f1']:.3f}, MMseqs2 "
      f"{R['A_aligned']['mmseqs']['f1']:.3f}, embedding SW {R['A_aligned']['emb']['f1']:.3f}.", "",
      f"## B. {len(QS)} e2e targets vs the relatives each search returned", "",
      "Relative structures: AlphaFold DB models of the Swiss-Prot entries. MMseqs2 hits = all rows of the e2e MMseqs2 "
      f"MSA; BinPLM hits = up to {MAX_YOURS} random rows per target of the e2e BinPLM MSA. 'MMseqs2 own' = the "
      "alignment MMseqs2 wrote into its MSA (3-iteration profile search).", ""]
for subset, title in (("same_fold", "Relatives that really share the query's fold (TM >= 0.5)"),
                      ("all", "All relatives with >= 20 structurally aligned residues")):
    m, y = R["B"][f"{subset}/mmseqs"], R["B"][f"{subset}/yours"]
    L += [f"### {title}", "",
          "| retrieval \\ aligner | pairs | letters: BLOSUM SW | letters: MMseqs2 own | embedding SW |",
          "|---|---|---|---|---|",
          f"| (a) MMseqs2 hits | {m['blosum']['n']} ({m['targets']} targets) | {cell(m['blosum'])} | {cell(m['own'])} | "
          f"{cell(m['emb'])} |",
          f"| (b) BinPLM hits | {y['blosum']['n']} ({y['targets']} targets) | {cell(y['blosum'])} | - | {cell(y['emb'])} |",
          ""]
L += ["Caveat: each hit set was accepted by its own aligner (MMseqs2 hits by MMseqs2's alignment score, BinPLM hits by "
      "the embedding SW score), so the diagonal cells have a selection advantage. The off-diagonal cells (BLOSUM on "
      "BinPLM hits, embedding SW on MMseqs2 hits) are the cleaner test of the aligner alone."]
(RESULTS / "align_2x2.md").write_text("\n".join(L) + "\n")
log("\n".join(L))
