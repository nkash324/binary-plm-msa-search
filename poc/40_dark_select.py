"""Test A, stage 1: pick candidate 'dark' proteins.
Pool = SCOPe 2.08 domains (classes a-d, 60-400 aa, standard letters, known structure) outside the superfamilies used
for development/tuning, and not used in any earlier test.  1000 random -> local MMseqs2 vs Swiss-Prot (ColabFold-style)
-> the 200 with the fewest Swiss-Prot relatives go to the ColabFold server screen.
Run in the 'fold' env.  Writes e2e/dark/candidates.fasta."""
import importlib.util
import shutil
import subprocess
from collections import defaultdict

import numpy as np

from common import DATA, PHASE1, read_fasta

E = DATA / "e2e"
D = E / "dark"
D.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("p1common", "/mnt/c/Folding/phase1/common.py")
p1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p1)
meta = p1.load_meta()
sid, sccs, fam, sf = meta["sid"], meta["sccs"], meta["fam"], meta["sf"]
_, p1seqs = read_fasta(PHASE1 / "seqs.fasta")
sf_of = lambda c: ".".join(c.split(".")[:3])

# exclusions exactly as in 18_e2e.py (dev queries' superfamilies + aligner-tuning pairs), plus every protein used so far
poc_q = [l.split("\t") for l in (DATA / "queries.tsv").read_text().splitlines()[1:]]
excl_sf = {sf_of(c) for _, c, _ in poc_q}
off1 = np.load(PHASE1 / "offsets.npy")
s1 = [s[:off1[i + 1] - off1[i]] for i, s in enumerate(p1seqs)]
ok1 = np.array([sf_of(c) not in excl_sf and len(q) <= 500 for c, q in zip(sccs, s1)])
r1, by_sf, pairs = np.random.default_rng(1), {}, set()
for i in np.flatnonzero(ok1):
    by_sf.setdefault(sf[i], []).append(i)
groups = [g for g in by_sf.values() if len(set(fam[g])) > 1]
while len(pairs) < 300:
    g = groups[r1.integers(len(groups))]
    a, b = r1.choice(g, 2, replace=False)
    if fam[a] != fam[b]:
        pairs.add((min(a, b), max(a, b)))
excl_sf |= {sf_of(sccs[i]) for p in pairs for i in p}
used = {s for s, _, _ in poc_q} | set(read_fasta(E / "queries.fasta")[0])
STD = set("ACDEFGHIKLMNPQRSTVWY")
pool = [i for i in range(len(sid)) if sccs[i][0] in "abcd" and 60 <= len(p1seqs[i]) <= 400
        and set(p1seqs[i]) <= STD and sid[i] not in used and sf_of(sccs[i]) not in excl_sf
        and (PHASE1 / "ent" / f"{sid[i]}.ent").exists()]
print(f"fresh pool: {len(pool)} domains")
rng = np.random.default_rng(456)                         # different seed from the earlier test
pick = sorted(rng.choice(pool, 1000, replace=False))

W = D / "pre"
shutil.rmtree(W, ignore_errors=True)
W.mkdir()
(W / "pool.fa").write_text("".join(f">{sid[i]}\n{p1seqs[i]}\n" for i in pick))
sh = lambda *a: subprocess.run([str(x) for x in a], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
sh("mmseqs", "createdb", W / "pool.fa", W / "q")
sh("mmseqs", "search", W / "q", DATA / "mmseqs_work" / "db", W / "res", W / "tmp", "--num-iterations", 3, "-s", 8,
   "-e", 0.1, "--max-seqs", 10000, "--threads", 6)
sh("mmseqs", "convertalis", W / "q", DATA / "mmseqs_work" / "db", W / "res", W / "res.m8")
hits = defaultdict(set)
for l in open(W / "res.m8"):
    c = l.split("\t")
    hits[c[0]].add(c[1])
order = sorted(pick, key=lambda i: (len(hits[sid[i]]), rng.random()))
cand = order[:200]
print("Swiss-Prot relatives of the 200 picked: median", int(np.median([len(hits[sid[i]]) for i in cand])),
      "max", max(len(hits[sid[i]]) for i in cand))
(D / "candidates.fasta").write_text("".join(f">{sid[i]}\n{p1seqs[i]}\n" for i in cand))
(D / "candidates.tsv").write_text("sid\tsccs\tlength\tswissprot_hits\n" + "".join(
    f"{sid[i]}\t{sccs[i]}\t{len(p1seqs[i])}\t{len(hits[sid[i]])}\n" for i in cand))
for i in cand:                                           # reference structures for scoring later
    shutil.copy(PHASE1 / "ent" / f"{sid[i]}.ent", E / "ref" / f"{sid[i]}.ent")
shutil.rmtree(W, ignore_errors=True)
print("CANDIDATES DONE")
