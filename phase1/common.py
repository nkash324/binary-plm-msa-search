"""Shared paths and helpers for the Phase 1 SCOPe benchmark."""
import os
from pathlib import Path

import numpy as np

# Bulky data lives on the Linux filesystem (fast); results go to C:\Folding (visible in Windows).
DATA = Path(os.environ.get("FOLD_DATA", Path.home() / "folding-data" / "phase1"))
RANKINGS = DATA / "rankings"
RESULTS = Path(__file__).resolve().parent / "results"
for d in (DATA, RANKINGS, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

SCOPE_URL = ("https://scop.berkeley.edu/downloads/scopeseq-2.08/"
             "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa")

ESM_MODEL = "esm2_t30_150M_UR50D"   # 150M params, 640-dim, 30 layers (fits the 4 GB WSL cap)
ESM_LAYER = 30
EMB_DIM = 640
MAX_LEN = 1022                       # ESM-2 positional limit (excl. BOS/EOS)
TOP_N = 1000                         # ranked targets kept per query


def read_fasta(path):
    ids, seqs, cur = [], [], []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
                ids.append(line[1:].split()[0])
            elif line:
                cur.append(line)
    if cur:
        seqs.append("".join(cur))
    return ids, seqs


def load_meta():
    """Returns dict with sid, sccs, and integer label codes for family / superfamily / fold."""
    sid, sccs = [], []
    with open(DATA / "meta.tsv") as f:
        next(f)
        for line in f:
            _, s, c, _ = line.rstrip("\n").split("\t")
            sid.append(s)
            sccs.append(c)
    fam = [c for c in sccs]
    sf = [".".join(c.split(".")[:3]) for c in sccs]
    fold = [".".join(c.split(".")[:2]) for c in sccs]

    def codes(xs):
        _, inv = np.unique(np.array(xs), return_inverse=True)
        return inv.astype(np.int32)

    return {"sid": sid, "sccs": sccs, "fam": codes(fam), "sf": codes(sf), "fold": codes(fold)}


def save_ranking(name, ranked_lists, seconds, bytes_per_protein, note=""):
    """ranked_lists: list (one per query, in index order) of int arrays of target indices, best first."""
    off = np.zeros(len(ranked_lists) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(r) for r in ranked_lists])
    idx = (np.concatenate(ranked_lists) if off[-1] else np.zeros(0)).astype(np.int32)
    np.savez(RANKINGS / f"{name}.npz", idx=idx, off=off, seconds=seconds,
             bytes_per_protein=bytes_per_protein, note=note)
    print(f"[saved] {name}: {seconds:.1f}s, {bytes_per_protein:.0f} B/protein", flush=True)
