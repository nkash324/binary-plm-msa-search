"""Download SCOPe 2.08 (40% identity subset) and write clean FASTA + labels."""
import collections
import urllib.request

import numpy as np

from common import DATA, SCOPE_URL

raw = DATA / "scope40_raw.fa"
if not raw.exists():
    print("Downloading", SCOPE_URL)
    urllib.request.urlretrieve(SCOPE_URL, raw)

entries, cur = [], None
with open(raw) as f:
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            parts = line[1:].split()
            cur = [parts[0], parts[1], []]
            entries.append(cur)
        elif line and cur is not None:
            cur[2].append(line)

kept = []
for sid, sccs, chunks in entries:
    if sccs[0] not in "abcdefg":     # true structural classes only
        continue
    kept.append((sid, sccs, "".join(chunks).upper()))

with open(DATA / "seqs.fasta", "w") as fa, open(DATA / "meta.tsv", "w") as mt:
    mt.write("idx\tsid\tsccs\tlength\n")
    for i, (sid, sccs, seq) in enumerate(kept):
        fa.write(f">{sid}\n{seq}\n")
        mt.write(f"{i}\t{sid}\t{sccs}\t{len(seq)}\n")

lens = np.array([len(s) for _, _, s in kept])
print(f"Domains kept: {len(kept)} (of {len(entries)})")
print("By class:", dict(sorted(collections.Counter(c[0] for _, c, _ in kept).items())))
print(f"Length: mean {lens.mean():.0f}, median {np.median(lens):.0f}, max {lens.max()}, "
      f">1022: {(lens > 1022).sum()}, total residues {lens.sum():,}")
