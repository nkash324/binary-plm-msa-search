"""Proof of concept: a realistic MSA-search workload, scaled to fit this PC.

Database: UniProt Swiss-Prot (~576k real proteins, ~210M residues).
Queries:  100 SCOPe domains (known structures), random over classes a-d.
"""
import os

from pathlib import Path

DATA = Path(os.environ.get("POC_DATA", Path.home() / "folding-data" / "poc"))
PHASE1 = Path(os.environ.get("FOLD_DATA", Path.home() / "folding-data" / "phase1"))
RESULTS = Path(__file__).resolve().parent / "results"
for d in (DATA, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

ESM_MODEL = "esm2_t30_150M_UR50D"
EMB_DIM = 640
MAX_LEN = 1022


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

DB_FASTA = DATA / "sprot.fasta"
Q_FASTA = DATA / "queries.fasta"
N_QUERIES = 100
K_PREFILTER = int(os.environ.get("POC_K1", 5000))   # stage 1 keeps this many proteins (per-protein Hamming)
K_ALIGN = 2000         # stage 2 keeps this many for alignment (the MSA candidate list)
E_VERIFY = 1e-3        # uniform alignment/verification cutoff (E-value at full-DB size)

# Keep the PC usable while experiments run.
CPU_WORKERS = int(os.environ.get("POC_CPU", 3))        # of the 6 CPUs WSL is allowed to use
GPU_REST = float(os.environ.get("POC_GPU_REST", 1.0))   # idle time per unit of GPU work (1.0 -> ~50% busy)
