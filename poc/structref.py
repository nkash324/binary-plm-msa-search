"""Shared helpers: structural reference alignments (US-align), alignment scoring, AFDB model download.
Same logic as poc/48_align_2x2.py."""
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices

from common import DATA

USALIGN = os.path.expanduser("~") + "/miniforge3/envs/cf/bin/USalign"
AFDB = DATA / "afdb"
AFDB.mkdir(exist_ok=True)
STD = set("ACDEFGHIKLMNPQRSTVWY")
clean = lambda s: "".join(c if c in STD else "X" for c in s)
blosum = Align.PairwiseAligner(mode="local", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                               open_gap_score=-11, extend_gap_score=-1)
glob_al = Align.PairwiseAligner(mode="global", match_score=2, mismatch_score=-1, open_gap_score=-3,
                                extend_gap_score=-0.5)


def blocks_to_pairs(al):
    out = []
    for (a0, a1), (b0, _) in zip(*al.aligned):
        out += [(a0 + k, b0 + k) for k in range(a1 - a0)]
    return out


def blosum_pairs(a, b):
    return blocks_to_pairs(blosum.align(clean(a), clean(b))[0])


def atom_to_seq(atom, seq):
    if atom == seq:
        return {i: i for i in range(len(seq))}
    return dict(blocks_to_pairs(glob_al.align(clean(atom), clean(seq))[0]))


def reference(fa, fb, seqa, seqb):
    """US-align a (chain 1) onto b: residue pairs < 5 A in sequence coordinates, and TM normalised by a."""
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


def _get(url, tries=4):
    for t in range(tries):                  # AFDB answers 403/5xx when hit too fast: back off
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:                   # noqa: BLE001
            pass
        time.sleep(2 * 2 ** t)
    return None


def afdb_pdb(acc):
    f = AFDB / f"{acc}.pdb"
    if f.exists() and f.stat().st_size:
        return f
    j = _get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}")
    d = None
    try:
        d = _get(json.loads(j)[0]["pdbUrl"]) if j else None
    except Exception:                       # noqa: BLE001
        pass
    if not d:                               # failures are not cached
        return None
    f.write_text(d)
    return f
