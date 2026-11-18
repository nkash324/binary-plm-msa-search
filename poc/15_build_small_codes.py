"""Smaller per-residue codes: PCA of the 640-dim +/-1 residue vectors, keep the top 256 (and top 128)
directions, take the sign. Same recipe as the window index. One pass over the 16.7 GB file.
Writes res_bits256/128.npy (+ q_ versions) for Swiss-Prot, and the same for the SCOPe (phase 1) set,
using one shared PCA basis (fitted on Swiss-Prot residues)."""
import time
from pathlib import Path

import numpy as np
import torch

import emb_align as ea
from common import DATA

P1 = Path.home() / "folding-data" / "phase1"
rng = np.random.default_rng(0)
torch.cuda.set_per_process_memory_fraction(0.5)
t0 = time.time()

src = np.load(DATA / "res_bits.npy", mmap_mode="r")
rows = np.sort(rng.choice(len(src), 200_000, replace=False))
X = np.unpackbits(np.asarray(src[rows]), axis=1).astype(np.float32) * 2 - 1
mu = X.mean(0)
w, v = np.linalg.eigh(np.cov((X - mu).T))
Pm = v[:, np.argsort(-w)[:256]].astype(np.float32)
np.savez(DATA / "pca_res.npz", mu=mu, P=Pm, explained=np.sort(w)[::-1] / w.sum())
gmu, gP = torch.from_numpy(mu).cuda(), torch.from_numpy(Pm).cuda()
print(f"PCA fitted ({time.time() - t0:.0f}s); top-256 keeps {100 * np.sort(w)[::-1][:256].sum() / w.sum():.0f}% "
      f"of variance, top-128 keeps {100 * np.sort(w)[::-1][:128].sum() / w.sum():.0f}%", flush=True)


def convert(src_path, out_prefix):
    s = np.load(src_path, mmap_mode="r")
    outs = {b: np.lib.format.open_memmap(out_prefix.parent / f"{out_prefix.name}{b}.npy", mode="w+",
                                         dtype=np.uint8, shape=(len(s), b // 8)) for b in (256, 128)}
    for a in range(0, len(s), 250_000):
        Z = ((ea.pm1(np.asarray(s[a:a + 250_000])).float() - gmu) @ gP > 0).cpu().numpy()
        for b, o in outs.items():
            o[a:a + len(Z)] = np.packbits(Z[:, :b], axis=1)
    for o in outs.values():
        o.flush()
    print(f"  {src_path.name}: {len(s):,} residues ({time.time() - t0:.0f}s)", flush=True)


convert(DATA / "q_res_bits.npy", DATA / "q_res_bits")
convert(P1 / "res_bits.npy", P1 / "res_bits")
convert(DATA / "res_bits.npy", DATA / "res_bits")
print("done", flush=True)
