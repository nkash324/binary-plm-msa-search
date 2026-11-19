"""MSAs for the 2x2 (retrieval source x aligner) on the 19 e2e targets. No folding here.

Retrieval (candidates):
  mm     = every Swiss-Prot hit of an MMseqs2 ColabFold-style search (3 iterations, -s 8, -e 0.1, max 10000)
  binplm = the 200 best candidates of the BinPLM window index (exactly as in 18_e2e.py)
Aligner (scores each candidate and writes its row):
  blosum = BLOSUM62 Smith-Waterman, gap open 11 / extend 1 (letters)
  emb    = embedding Smith-Waterman, 256-bit residue codes (bias/gap from emb_align_params_256.json)
Acceptance: the SAME calibrated chance cutoff in every cell. For each target and aligner: score the same 200 random
Swiss-Prot proteins (none of them a candidate of either search), fit score ~ a + b*log(length), z-score, fit a Gumbel
tail to the null z (90th/99th percentiles) and accept candidates above the 1-in-2000 chance level (as in 18_e2e.py).
Rows: query length, matched columns only (no insertions), same for all cells.
Writes ~/folding-data/poc/msa_2x2/<cell>/<target>.a3m, results/msa_2x2.{md,json}.  Run in the 'fold' env.
"""
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from Bio import Align
from Bio.Align import substitution_matrices

sys.path.insert(0, "/mnt/c/Folding/poc")
import emb_align as ea                                                   # noqa: E402
from common import DATA, DB_FASTA, RESULTS, read_fasta                   # noqa: E402

E = DATA / "e2e"
OUT = DATA / "msa_2x2"
CELLS = ["mm_blosum", "mm_emb", "binplm_blosum", "binplm_emb"]
for c in CELLS:
    (OUT / c).mkdir(parents=True, exist_ok=True)
torch.cuda.set_per_process_memory_fraction(0.5)
PAUSE = DATA / "PAUSE"


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def sh(*a):
    subprocess.run([str(x) for x in a], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


qids, qseqs = read_fasta(E / "queries.fasta")
QS = [(q, s) for q, s in zip(qids, qseqs) if (E / "a3m" / "mmseqs2" / f"{q}.a3m").exists()]
db_ids, db_seqs = read_fasta(DB_FASTA)
id2idx = {d: i for i, d in enumerate(db_ids)}
NP = len(db_ids)
off = np.load(DATA / "offsets.npy")
lens = np.diff(off)

# ================================================================ retrieval (a): MMseqs2, all targets in one batch
W = DATA / "msa2x2_work"
shutil.rmtree(W, ignore_errors=True)
W.mkdir()
(W / "q.fa").write_text("".join(f">{q}\n{s}\n" for q, s in QS))
t0 = time.time()
db = DATA / "mmseqs_work" / "db"
sh("mmseqs", "createdb", W / "q.fa", W / "q")
sh("mmseqs", "search", W / "q", db, W / "res", W / "tmp", "--num-iterations", 3, "-s", 8, "-e", 0.1,
   "--max-seqs", 10000, "-a", "--threads", 3)
sh("mmseqs", "convertalis", W / "q", db, W / "res", W / "res.m8")
mm_cand = defaultdict(list)
for l in open(W / "res.m8"):
    q, t = l.split("\t")[:2]
    if t in id2idx and id2idx[t] not in mm_cand[q]:
        mm_cand[q].append(id2idx[t])
log(f"MMseqs2 search: {time.time() - t0:.0f}s, hits per target median {np.median([len(mm_cand[q]) for q, _ in QS])}")

# ================================================================ retrieval (b): BinPLM window index (as 18_e2e.py)
R256 = np.load(DATA / "res_bits256.npy", mmap_mode="r")
PRM = json.loads((DATA / "emb_align_params_256.json").read_text())
BIAS, GAP = PRM["bias"], PRM["gap"]
pca = np.load(DATA / "pca_res.npz")
gmu, gP = torch.from_numpy(pca["mu"]).cuda(), torch.from_numpy(pca["P"][:, :256].copy()).cuda()
z = np.load(DATA / "win64_256.npz")
wcodes = np.ascontiguousarray(z["codes"]).view(np.uint64)
owner, WMU, WPM = z["owner"], z["mu"], z["P"]
null_pool = np.random.default_rng(42).choice(NP, 2000, replace=False)
mu_res = torch.from_numpy(np.load(DATA / "mu.npz")["res"]).cuda()
from transformers import AutoTokenizer, EsmModel                                   # noqa: E402
tok = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
esm = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D", add_pooling_layer=False,
                               attn_implementation="sdpa").eval().cuda().half()
SH8 = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], device="cuda", dtype=torch.uint8)


@torch.no_grad()
def embed(seq):
    enc = tok([seq], return_tensors="pt")
    r = esm(input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda()).last_hidden_state.float()
    r = r[0, 1:len(seq) + 1]
    sgn = r - mu_res >= 0
    b640 = (sgn.view(-1, 80, 8).to(torch.uint8) * SH8).sum(-1, dtype=torch.uint8).cpu().numpy()
    Q256 = ((sgn.half() * 2 - 1).float() - gmu) @ gP > 0
    return b640, (Q256.half() * 2 - 1)


def window_search(b640):
    X = np.unpackbits(b640, axis=1).astype(np.float32) * 2 - 1
    L, Wn, B, TOPW = len(X), 64, 256, 2000
    starts = sorted(set(list(range(0, max(1, L - Wn + 1), Wn // 4)) + [max(0, L - Wn)]))
    qc = np.ascontiguousarray(np.packbits((np.stack([X[a:a + Wn].mean(0) for a in starts]) - WMU) @ WPM > 0,
                                          axis=1)).view(np.uint64)
    score = np.zeros(NP, np.float32)
    for lo in range(0, len(wcodes), 500_000):
        c = wcodes[lo:lo + 500_000]
        h = np.zeros((len(qc), len(c)), np.uint16)
        for w in range(c.shape[1]):
            h += np.bitwise_count(c[None, :, w] ^ qc[:, None, w])
        i = np.argpartition(h, TOPW, axis=1)[:, :TOPW]
        zz = (B / 2 - np.take_along_axis(h, i, 1).astype(np.float32)) / (np.sqrt(B) / 2)
        for j in range(len(qc)):
            o = np.argsort(-zz[j])
            pp, first = np.unique(owner[lo + i[j][o]], return_index=True)
            score[pp] += np.maximum(zz[j][o][first], 0)
    return score


# ================================================================ aligners
blosum = Align.PairwiseAligner(mode="local", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                               open_gap_score=-11, extend_gap_score=-1)
STD = set("ACDEFGHIKLMNPQRSTVWY")
clean = lambda s: "".join(c if c in STD else "X" for c in s)


class Blosum:
    def __init__(self, seq):
        self.q = clean(seq)

    def scores(self, ps):
        return np.array([blosum.score(self.q, clean(db_seqs[p])) for p in ps], float)

    def pairs(self, ps):
        out = []
        for p in ps:
            al = blosum.align(self.q, clean(db_seqs[p]))[0]
            out.append([(a0 + k, b0 + k) for (a0, a1), (b0, _) in zip(*al.aligned) for k in range(a1 - a0)])
        return out


class Emb:
    def __init__(self, Q):
        self.Q = Q

    def _T(self, ps):
        return [ea.pm1(np.array(R256[off[p]:off[p + 1]])) for p in ps]

    def scores(self, ps):
        return ea.sw_many(self.Q, self._T(ps), BIAS, GAP)[0] if len(ps) else np.zeros(0)

    def pairs(self, ps):
        return ea.sw_many(self.Q, self._T(ps), BIAS, GAP, keep=True)[1] if len(ps) else []


def calibrate(al, null):
    """Same recipe as 18_e2e.py: length regression on the null, Gumbel tail, 1-in-2000 chance level."""
    ns = al.scores(null)
    A = np.c_[np.ones(len(null)), np.log(lens[null])]
    coef, *_ = np.linalg.lstsq(A, ns, rcond=None)
    sd = np.std(ns - A @ coef)
    zf = lambda s, ps: (s - np.c_[np.ones(len(ps)), np.log(lens[ps])] @ coef) / sd
    q90, q99 = np.quantile(zf(ns, null), [0.9, 0.99])
    beta = (q99 - q90) / (np.log(-np.log(0.9)) - np.log(-np.log(0.99)))
    return zf, q90 + beta * (-np.log(-np.log(1 - 1 / 2000)) + np.log(-np.log(0.9)))


def neff(rows):
    if len(rows) < 2:
        return float(len(rows))
    A = np.array([list(r) for r in rows])
    X = np.stack([(A == c) for c in "ACDEFGHIKLMNPQRSTVWY"], -1).reshape(len(rows), -1).astype(np.float32)
    nong = (A != "-").sum(1).astype(np.float32)
    S = X @ X.T / np.maximum(np.minimum.outer(nong, nong), 1)
    return float((1 / (S >= 0.8).sum(1)).sum())


# ================================================================ build
D = {}
for n, (q, seq) in enumerate(QS):
    while PAUSE.exists():
        time.sleep(10)
    t0 = time.time()
    b640, Q = embed(seq)
    sc = window_search(b640)
    bc = np.argsort(-sc)[:200]
    cand = {"mm": np.array(mm_cand[q], int), "binplm": bc[sc[bc] > 0]}
    used = set(cand["mm"].tolist()) | set(cand["binplm"].tolist())
    null = np.array([p for p in null_pool if p not in used])[:200]
    D[q] = {"L": len(seq), "candidates": {k: len(v) for k, v in cand.items()},
            "overlap_candidates": len(set(cand["mm"].tolist()) & set(cand["binplm"].tolist()))}
    acc = {}
    for aname, al in (("blosum", Blosum(seq)), ("emb", Emb(Q))):
        zf, thr = calibrate(al, null)
        D[q][f"thr_{aname}"] = float(thr)
        for rname, ps in cand.items():
            cell = f"{rname}_{aname}"
            keep = ps[zf(al.scores(ps), ps) > thr] if len(ps) else ps
            rows = []
            for p, pr in zip(keep, al.pairs(keep)):
                r = ["-"] * len(seq)
                for i, j in pr:
                    r[i] = db_seqs[p][j]
                rows.append("".join(r))
            (OUT / cell / f"{q}.a3m").write_text(f">{q}\n{seq}\n" + "".join(
                f">{db_ids[p]}\n{r}\n" for p, r in zip(keep, rows)))
            acc[cell] = set(keep.tolist())
            D[q][cell] = {"rows": len(rows), "neff80": neff([seq] + rows)}
    D[q]["overlap_mm_emb_vs_binplm_emb"] = len(acc["mm_emb"] & acc["binplm_emb"])
    log(f"[{n + 1}/{len(QS)}] {q} L={len(seq)} cand mm {len(cand['mm'])} / binplm {len(cand['binplm'])}; rows "
        + ", ".join(f"{c} {D[q][c]['rows']}" for c in CELLS) + f" ({time.time() - t0:.0f}s)")
shutil.rmtree(W, ignore_errors=True)

# reference: the native MSAs from the e2e run (own acceptance rules, own alignments)
for q, seq in QS:
    for nat in ("mmseqs2", "yours"):
        _, rows = read_fasta(E / "a3m" / nat / f"{q}.a3m")
        rows = ["".join(c for c in r if not c.islower()) for r in rows]
        D[q][f"native_{nat}"] = {"rows": len(rows) - 1, "neff80": neff(rows)}
(RESULTS / "msa_2x2.json").write_text(json.dumps(D, indent=1))

qs = [q for q, _ in QS]
med = lambda c, k: float(np.median([D[q][c][k] for q in qs]))
mean = lambda c, k: float(np.mean([D[q][c][k] for q in qs]))
L = [f"# MSAs for the 2x2 (retrieval x aligner), {len(QS)} e2e targets, Swiss-Prot", "",
     "Every cell uses the same acceptance rule: per target and aligner, 200 random Swiss-Prot proteins (not candidates "
     "of either search) set a length-corrected chance level (Gumbel tail); candidates scoring above the 1-in-2000 "
     "level are kept. Rows: matched query columns only. Neff = number of effective sequences at 80% identity "
     "(query included). Files: ~/folding-data/poc/msa_2x2/<cell>/<target>.a3m", "",
     f"Candidates per target (median): MMseqs2 {np.median([D[q]['candidates']['mm'] for q in qs]):.0f}, "
     f"BinPLM {np.median([D[q]['candidates']['binplm'] for q in qs]):.0f} (shared "
     f"{np.median([D[q]['overlap_candidates'] for q in qs]):.0f}).", "",
     "| cell | retrieval | aligner | rows median | rows mean | Neff80 median | Neff80 mean | targets with 0 rows |",
     "|---|---|---|---|---|---|---|---|"]
names = {"mm_blosum": ("MMseqs2 hits", "BLOSUM SW"), "mm_emb": ("MMseqs2 hits", "embedding SW"),
         "binplm_blosum": ("BinPLM top 200", "BLOSUM SW"), "binplm_emb": ("BinPLM top 200", "embedding SW"),
         "native_mmseqs2": ("(ref) native MMseqs2 MSA", "MMseqs2 own"),
         "native_yours": ("(ref) e2e BinPLM MSA", "embedding SW")}
for c in CELLS + ["native_mmseqs2", "native_yours"]:
    L.append(f"| {c} | {names[c][0]} | {names[c][1]} | {med(c, 'rows'):.0f} | {mean(c, 'rows'):.1f} | "
             f"{med(c, 'neff80'):.1f} | {mean(c, 'neff80'):.1f} | {sum(D[q][c]['rows'] == 0 for q in qs)} |")
L += ["", "## Per target (rows)", "",
      "| target | L | cand mm / binplm | " + " | ".join(CELLS) + " | native mm / BinPLM |",
      "|---|---|---|" + "---|" * len(CELLS) + "---|"]
for q in qs:
    d = D[q]
    L.append(f"| {q} | {d['L']} | {d['candidates']['mm']} / {d['candidates']['binplm']} | "
             + " | ".join(str(d[c]["rows"]) for c in CELLS)
             + f" | {d['native_mmseqs2']['rows']} / {d['native_yours']['rows']} |")
(RESULTS / "msa_2x2.md").write_text("\n".join(L) + "\n")
log("\n".join(L))
