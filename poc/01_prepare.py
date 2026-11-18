"""Decompress Swiss-Prot (accession as ID) and pick the query set."""
import gzip

import numpy as np

from common import DATA, DB_FASTA, N_QUERIES, PHASE1, Q_FASTA, read_fasta

n = 0
with gzip.open(DATA / "download" / "uniprot_sprot.fasta.gz", "rt") as fin, open(DB_FASTA, "w") as fout:
    for line in fin:
        if line.startswith(">"):
            fout.write(">" + line.split("|")[1] + "\n")      # >sp|P12345|NAME_HUMAN ... -> >P12345
            n += 1
        else:
            fout.write(line)
print(f"{n:,} database proteins -> {DB_FASTA}")

ids, seqs = read_fasta(PHASE1 / "seqs.fasta")
sccs = {}
with open(PHASE1 / "meta.tsv") as f:
    next(f)
    for line in f:
        _, s, c, _ = line.rstrip("\n").split("\t")
        sccs[s] = c
ok = [i for i, (s, q) in enumerate(zip(ids, seqs))
      if sccs[s][0] in "abcd" and 60 <= len(q) <= 400 and set(q) <= set("ACDEFGHIKLMNPQRSTVWY")]
pick = np.sort(np.random.default_rng(0).choice(ok, N_QUERIES, replace=False))
with open(Q_FASTA, "w") as f, open(DATA / "queries.tsv", "w") as g:
    g.write("sid\tsccs\tlength\n")
    for i in pick:
        f.write(f">{ids[i]}\n{seqs[i]}\n")
        g.write(f"{ids[i]}\t{sccs[ids[i]]}\t{len(seqs[i])}\n")
print(f"{N_QUERIES} queries -> {Q_FASTA}")
