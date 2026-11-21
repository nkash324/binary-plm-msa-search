"""Step 3b/3c: residue codebooks ("embedding letters") fitted on Swiss-Prot ONLY, and group IDs for every residue.

Codebooks are fitted directly on the +/-1 640-bit residue codes (the only per-residue form Swiss-Prot has), so SCOPe
and Swiss-Prot are assigned the same way. Centroid c_a = mean of its members' +/-1 vectors; a residue goes to the
centroid with the largest dot product. Score table for the cheap first pass:
    table[a, b] = c_a . c_b / 640  (= the expected 640-bit cell score of a residue from group a vs one from group b)
Variants: K = 20, 256, 1024 on all 640 bits; PQ4 = 4 sub-codebooks of 256 on 160-bit slices (cell score = sum).
Writes poc/codebooks.npz, poc/res_cb{K}.npy & phase1/res_cb{K}_sp.npy (uint8/uint16 per residue; PQ: 4 bytes).
Run in the 'fold' env."""
import time

import numpy as np
import torch

import esw
from common import DATA, PHASE1

torch.cuda.set_per_process_memory_fraction(0.6)
rng = np.random.default_rng(5)
log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)
sp = np.load(DATA / "res_bits.npy", mmap_mode="r")
sc = np.load(PHASE1 / "res_bits_sp.npy", mmap_mode="r")
NS = 600_000
rows = np.sort(rng.choice(len(sp), NS, replace=False))
X = esw.unpack(np.asarray(sp[rows]))                                     # (600k, 640) +/-1 half on GPU
log(f"sample of {NS:,} Swiss-Prot residues")


def kmeans(X, K, iters=25, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g, device="cuda")[:K]].float()
    for it in range(iters):
        a = torch.cat([(X[s:s + 200_000] @ C.T.half()).argmax(1) for s in range(0, len(X), 200_000)])
        S = torch.zeros_like(C)
        for s in range(0, len(X), 200_000):
            S.index_add_(0, a[s:s + 200_000], X[s:s + 200_000].float())
        n = torch.bincount(a, minlength=K).float()
        empty = n == 0
        C = torch.where(empty[:, None], C, S / n.clamp(min=1)[:, None])
        if empty.any():                                                  # re-seed empty groups
            C[empty] = X[torch.randint(len(X), (int(empty.sum()),), generator=g, device="cuda")].float()
    return C, a


CB = {}
for K in (20, 256, 1024):
    t = time.time()
    C, a = kmeans(X, K)
    CB[f"C{K}"] = C.cpu().numpy()
    CB[f"T{K}"] = (C @ C.T).cpu().numpy() / 640
    fit = float(sum((X[s:s + 200_000].float() * C[a[s:s + 200_000]]).sum() for s in range(0, len(X), 200_000)) / len(X) / 640)
    log(f"K={K}: fitted in {time.time() - t:.0f}s, mean member-centroid cosine {fit:.3f}")
PQ_M = 4
W = 640 // PQ_M
for m in range(PQ_M):
    C, _ = kmeans(X[:, m * W:(m + 1) * W].contiguous(), 256, seed=m)
    CB[f"PQ{m}"] = C.cpu().numpy()
    CB[f"PQT{m}"] = (C @ C.T).cpu().numpy() / 640                       # slice tables sum to the full score
log("PQ4 fitted")
np.savez(DATA / "codebooks.npz", **CB)

for name, bits, dst in (("SCOPe", sc, PHASE1), ("Swiss-Prot", sp, DATA)):     # one pass over the codes each
    t = time.time()
    suffix = "_sp" if name == "SCOPe" else ""
    outs = {K: np.lib.format.open_memmap(dst / f"res_cb{K}{suffix}.npy", mode="w+", shape=(len(bits),),
                                         dtype=np.uint8 if K <= 256 else np.uint16) for K in (20, 256, 1024)}
    pq = np.lib.format.open_memmap(dst / f"res_pq4{suffix}.npy", mode="w+", shape=(len(bits), PQ_M), dtype=np.uint8)
    Cg = {K: torch.from_numpy(CB[f"C{K}"]).cuda().half() for K in outs}
    Pg = [torch.from_numpy(CB[f"PQ{m}"]).cuda().half() for m in range(PQ_M)]
    for s in range(0, len(bits), 1_000_000):
        Y = esw.unpack(np.asarray(bits[s:s + 1_000_000]))
        for K, o in outs.items():
            o[s:s + len(Y)] = (Y @ Cg[K].T).argmax(1).cpu().numpy().astype(o.dtype)
        pq[s:s + len(Y)] = torch.stack([(Y[:, m * W:(m + 1) * W] @ Pg[m].T).argmax(1) for m in range(PQ_M)],
                                       1).cpu().numpy().astype(np.uint8)
    for o in list(outs.values()) + [pq]:
        o.flush()
    del outs, pq
    log(f"{name}: {len(bits):,} residues assigned ({time.time() - t:.0f}s)")
