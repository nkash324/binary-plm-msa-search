"""Exhaustive embedding Smith-Waterman: many queries against a whole database of per-residue codes.

Same scoring and recurrence as emb_align.sw (cell score = cosine of the +/-1 codes minus a bias, linear gap), but:
  - the database is streamed from a memory-mapped packed-bit file in contiguous chunks (RAM stays small),
  - each chunk is unpacked once on the GPU and reused for every query (database outer loop, queries inner loop),
  - only the best local score per (query, target) is kept (traceback: use emb_align.sw_many on the few you keep).
"""
import time

import numpy as np
import torch

DEV = "cuda"
_SH = None


def unpack(bits):
    """packed uint8 (n, D/8) -> +/-1 half (n, D) on the GPU."""
    global _SH
    if _SH is None:
        _SH = torch.arange(7, -1, -1, device=DEV, dtype=torch.uint8)
    b = torch.from_numpy(np.ascontiguousarray(bits)).to(DEV)
    return ((b.unsqueeze(-1) >> _SH) & 1).reshape(b.shape[0], -1).half() * 2 - 1


@torch.no_grad()
def _sw_block(Q, R, starts, tl, bias, gap):
    """Q (Lq, D) half; R (n_res + 1, D) half chunk residues, last row = zeros (padding);
    starts/tl: int64 tensors (B,) start row and length of each target in R. Returns best scores (B,)."""
    Lq, D = Q.shape
    B, Lm = len(tl), int(tl.max())
    j = torch.arange(Lm, device=DEV)
    idx = starts[:, None] + j[None, :]
    pad = j[None, :] >= tl[:, None]
    idx = torch.where(pad, R.shape[0] - 1, idx)
    T = R[idx.reshape(-1)]                                                 # (B*Lm, D)
    S = (Q @ T.T).float().view(Lq, B, Lm) / D - bias
    S.masked_fill_(pad[None], -1e4)
    jg = gap * torch.arange(Lm + 1, device=DEV, dtype=torch.float32)
    prev = torch.zeros(B, Lm + 1, device=DEV)
    zero = torch.zeros(B, 1, device=DEV)
    best = torch.zeros(B, device=DEV)
    for i in range(Lq):
        t = torch.clamp(torch.maximum(prev[:, :-1] + S[i], prev[:, 1:] - gap), min=0)
        cur = torch.cummax(torch.cat([zero, t], 1) + jg, 1).values - jg
        best = torch.maximum(best, cur.amax(1))
        prev = cur
    return best


def search(Qs, bits, off, bias, gap, targets=None, chunk_res=1_000_000, cells=120_000_000, pause=None, log=None):
    """Qs: list of (Lq, D) +/-1 half GPU tensors. bits: (n_res, D/8) uint8 (np.memmap ok); off: (N+1,) offsets.
    targets: sorted protein indices to score (default all). Returns float32 (len(Qs), len(targets))."""
    N = len(off) - 1
    targets = np.arange(N) if targets is None else np.asarray(targets)
    out = np.zeros((len(Qs), len(targets)), np.float32)
    lens = np.diff(off)[targets]
    k, t0 = 0, time.time()
    while k < len(targets):                       # chunk = consecutive targets up to chunk_res residues
        n, tot = 0, 0
        while k + n < len(targets) and (n == 0 or tot + lens[k + n] <= chunk_res):
            tot += lens[k + n]
            n += 1
        ts = targets[k:k + n]
        if pause is not None:
            while pause.exists():
                time.sleep(10)
        contiguous = ts[-1] - ts[0] == n - 1
        if contiguous:
            raw = np.asarray(bits[off[ts[0]]:off[ts[-1] + 1]])
            st = off[ts] - off[ts[0]]
        else:
            raw = np.concatenate([np.asarray(bits[off[p]:off[p + 1]]) for p in ts])
            st = np.r_[0, np.cumsum(lens[k:k + n])[:-1]]
        R = torch.cat([unpack(raw), torch.zeros(1, raw.shape[1] * 8, device=DEV, dtype=torch.half)])
        order = np.argsort(lens[k:k + n], kind="stable")
        tl_all, st_all = lens[k:k + n][order], st[order]
        for qi, Q in enumerate(Qs):
            a = 0
            while a < n:                          # length-sorted sub-batches under the cell budget
                b = a + 1
                while b < n and (b + 1 - a) * tl_all[b] * len(Q) <= cells:
                    b += 1
                sc = _sw_block(Q, R, torch.from_numpy(st_all[a:b]).to(DEV), torch.from_numpy(tl_all[a:b]).to(DEV),
                               bias, gap)
                out[qi, k + order[a:b]] = sc.cpu().numpy()
                a = b
        del R
        k += n
        if log is not None:
            log(f"    esw: {k}/{len(targets)} targets, {time.time() - t0:.0f}s")
    return out
