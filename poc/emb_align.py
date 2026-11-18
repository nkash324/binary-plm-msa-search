"""Embedding Smith-Waterman: local alignment where the score for pairing query residue i with target
residue j is the similarity of their ESM-2 binary codes (cosine of the +/-1 vectors) minus a bias.
Linear gap penalty. Batched on the GPU; traceback on the CPU for the alignments we keep."""
import numpy as np
import torch

DEV = "cuda"
_SH = None


def pm1(bits):
    """packed uint8 (n, D/8) numpy -> +/-1 half tensor (n, D) on GPU."""
    global _SH
    if _SH is None:
        _SH = torch.arange(7, -1, -1, device=DEV, dtype=torch.uint8)
    b = torch.from_numpy(np.ascontiguousarray(bits)).to(DEV)
    return (((b.unsqueeze(-1) >> _SH) & 1).reshape(b.shape[0], -1).half() * 2 - 1)


def sw(Q, Ts, bias, gap, keep=False):
    """Q: (Lq, D) +/-1 half; Ts: list of (Lt, D) +/-1 half. Returns scores (np) and, if keep,
    a list of (H, S) numpy matrices for traceback."""
    Lq, D = Q.shape
    B, Lm = len(Ts), max(len(t) for t in Ts)
    T = torch.zeros(B, Lm, D, device=DEV, dtype=torch.half)
    pad = torch.ones(B, Lm, device=DEV, dtype=torch.bool)
    for k, t in enumerate(Ts):
        T[k, :len(t)] = t
        pad[k, :len(t)] = False
    S = torch.einsum("id,bjd->bij", Q, T).float() / D - bias
    S.masked_fill_(pad[:, None, :], -1e4)
    jg = gap * torch.arange(Lm + 1, device=DEV, dtype=torch.float32)
    prev = torch.zeros(B, Lm + 1, device=DEV)
    zero = torch.zeros(B, 1, device=DEV)
    best = torch.zeros(B, device=DEV)
    Hs = [prev] if keep else None
    for i in range(Lq):
        t = torch.clamp(torch.maximum(prev[:, :-1] + S[:, i], prev[:, 1:] - gap), min=0)
        cur = torch.cummax(torch.cat([zero, t], 1) + jg, 1).values - jg
        best = torch.maximum(best, cur.amax(1))
        prev = cur
        if keep:
            Hs.append(cur)
    out = best.cpu().numpy()
    if not keep:
        return out, None
    H = torch.stack(Hs, 1).cpu().numpy()
    S = S.cpu().numpy()
    return out, [(H[k, :, :len(t) + 1], S[k, :, :len(t)]) for k, t in enumerate(Ts)]


def traceback(H, S, gap):
    """Aligned (query_pos, target_pos) pairs of the best local alignment."""
    i, j = np.unravel_index(np.argmax(H), H.shape)
    pairs = []
    while i > 0 and j > 0 and H[i, j] > 1e-6:
        h = H[i, j]
        if abs(h - (H[i - 1, j - 1] + S[i - 1, j - 1])) < 1e-3:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif abs(h - (H[i - 1, j] - gap)) < 1e-3:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def sw_many(Q, Ts, bias, gap, keep=False, cells=20_000_000):
    """Batches targets by length so each GPU batch stays small. Returns scores and (optionally) pairs."""
    order = np.argsort([len(t) for t in Ts])
    scores = np.zeros(len(Ts))
    pairs = [None] * len(Ts)
    k = 0
    while k < len(order):
        L = len(Ts[order[k]])
        n = 1
        while k + n < len(order) and (n + 1) * len(Ts[order[k + n]]) * max(len(Q), 1) <= cells:
            n += 1
        idx = order[k:k + n]
        s, mats = sw(Q, [Ts[i] for i in idx], bias, gap, keep)
        scores[idx] = s
        if keep:
            for i, (H, S) in zip(idx, mats):
                pairs[i] = traceback(H, S, gap)
        k += n
    return scores, pairs
