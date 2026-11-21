"""Fused Triton kernel for exhaustive embedding Smith-Waterman on packed 640-bit codes.

Same score and recurrence as emb_align.sw / esw.search:
  cell score  = (D - 2 * Hamming(q_i, t_j)) / D - bias          (= cosine of the +/-1 codes minus a bias)
  H[i, j]     = max(0, H[i-1, j-1] + s, H[i-1, j] - gap, H[i, j-1] - gap)   (linear gap; row via a max-scan)
One program per target protein; the target's codes stay in registers, the previous DP row in a small scratch
buffer (L1/L2). Targets longer than MAXJ fall back to the PyTorch engine (esw.search)."""
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

import esw

DEV = "cuda"
W = 10                       # 640 bits = 10 x 64-bit words
MAXJ = 4096


@triton.jit
def _max_op(a, b):
    return tl.maximum(a, b)


@triton.jit
def _sw_kernel(q_ptr, Lq, t_ptr, tstart_ptr, tlen_ptr, scratch_ptr, out_ptr, bias, gap,
               NW: tl.constexpr, BLOCK_J: tl.constexpr):
    pid = tl.program_id(0)
    ts = tl.load(tstart_ptr + pid)
    tlen = tl.load(tlen_ptr + pid)
    j = tl.arange(0, BLOCK_J)
    m = j < tlen
    # target words in registers: NW vectors of BLOCK_J int64
    tw0 = tl.load(t_ptr + (ts + j) * NW + 0, mask=m, other=0)
    tw1 = tl.load(t_ptr + (ts + j) * NW + 1, mask=m, other=0)
    tw2 = tl.load(t_ptr + (ts + j) * NW + 2, mask=m, other=0)
    tw3 = tl.load(t_ptr + (ts + j) * NW + 3, mask=m, other=0)
    tw4 = tl.load(t_ptr + (ts + j) * NW + 4, mask=m, other=0)
    tw5 = tl.load(t_ptr + (ts + j) * NW + 5, mask=m, other=0)
    tw6 = tl.load(t_ptr + (ts + j) * NW + 6, mask=m, other=0)
    tw7 = tl.load(t_ptr + (ts + j) * NW + 7, mask=m, other=0)
    tw8 = tl.load(t_ptr + (ts + j) * NW + 8, mask=m, other=0)
    tw9 = tl.load(t_ptr + (ts + j) * NW + 9, mask=m, other=0)
    jg = gap * j.to(tl.float32)
    sbase = scratch_ptr + pid * (BLOCK_J + 1)          # scratch[0] = H[i-1, -1] = 0 ; scratch[1 + j] = H[i-1, j]
    tl.store(sbase + 1 + j, tl.zeros([BLOCK_J], tl.float32))
    tl.store(sbase, 0.0)
    best = tl.zeros([BLOCK_J], tl.float32)
    D = NW * 64
    for i in range(Lq):
        qb = q_ptr + i * NW
        ham = libdevice.popc(tl.load(qb + 0) ^ tw0) + libdevice.popc(tl.load(qb + 1) ^ tw1)
        ham += libdevice.popc(tl.load(qb + 2) ^ tw2) + libdevice.popc(tl.load(qb + 3) ^ tw3)
        ham += libdevice.popc(tl.load(qb + 4) ^ tw4) + libdevice.popc(tl.load(qb + 5) ^ tw5)
        ham += libdevice.popc(tl.load(qb + 6) ^ tw6) + libdevice.popc(tl.load(qb + 7) ^ tw7)
        ham += libdevice.popc(tl.load(qb + 8) ^ tw8) + libdevice.popc(tl.load(qb + 9) ^ tw9)
        s = (D - 2 * ham).to(tl.float32) / D - bias
        s = tl.where(m, s, -1e4)
        tl.debug_barrier()
        diag = tl.load(sbase + j)                      # H[i-1, j-1]
        up = tl.load(sbase + 1 + j)                    # H[i-1, j]
        t = tl.maximum(tl.maximum(diag + s, up - gap), 0.0)
        cur = tl.associative_scan(t + jg, 0, _max_op) - jg
        best = tl.maximum(best, tl.where(m, cur, 0.0))
        tl.debug_barrier()
        tl.store(sbase + 1 + j, cur)
    tl.store(out_ptr + pid, tl.max(best, 0))


@triton.jit
def _lut_kernel(q_ptr, Lq, t_ptr, tstart_ptr, tlen_ptr, tab_ptr, scratch_ptr, out_ptr, bias, gap,
                K: tl.constexpr, M: tl.constexpr, BLOCK_J: tl.constexpr):
    """Same DP; cell score = sum over m of table_m[q_sym, t_sym] - bias (codebook / product-quantised letters)."""
    pid = tl.program_id(0)
    ts = tl.load(tstart_ptr + pid)
    tlen = tl.load(tlen_ptr + pid)
    j = tl.arange(0, BLOCK_J)
    m = j < tlen
    jg = gap * j.to(tl.float32)
    sbase = scratch_ptr + pid * (BLOCK_J + 1)
    tl.store(sbase + 1 + j, tl.zeros([BLOCK_J], tl.float32))
    tl.store(sbase, 0.0)
    best = tl.zeros([BLOCK_J], tl.float32)
    for i in range(Lq):
        s = tl.zeros([BLOCK_J], tl.float32)
        for mm in tl.static_range(M):
            tb = tl.load(t_ptr + (ts + j) * M + mm, mask=m, other=0).to(tl.int32)
            qa = tl.load(q_ptr + i * M + mm).to(tl.int32)
            s += tl.load(tab_ptr + mm * K * K + qa * K + tb, mask=m, other=0.0)
        s = tl.where(m, s - bias, -1e4)
        tl.debug_barrier()
        diag = tl.load(sbase + j)
        up = tl.load(sbase + 1 + j)
        t = tl.maximum(tl.maximum(diag + s, up - gap), 0.0)
        cur = tl.associative_scan(t + jg, 0, _max_op) - jg
        best = tl.maximum(best, tl.where(m, cur, 0.0))
        tl.debug_barrier()
        tl.store(sbase + 1 + j, cur)
    tl.store(out_ptr + pid, tl.max(best, 0))


def search_lut(Qsym, sym, off, table, bias, gap, targets=None, chunk_res=8_000_000, maxj=4096):
    """Codebook SW. Qsym: list of (Lq, M) int arrays of query group IDs; sym: (n_res, M) or (n_res,) group IDs of the
    database (memmap ok); table: (M, K, K) float32 (the slices' tables sum to the cell score). Targets longer than
    maxj are cut to their first maxj residues (first-pass filter only)."""
    N = len(off) - 1
    targets = np.arange(N) if targets is None else np.asarray(targets)
    lens = np.minimum(np.diff(off)[targets], maxj)
    M, K = table.shape[0], table.shape[1]
    tab = torch.from_numpy(np.ascontiguousarray(table, np.float32)).to(DEV)
    Qg = [torch.from_numpy(np.ascontiguousarray(np.asarray(q).reshape(len(q), M), np.int32)).to(DEV) for q in Qsym]
    out = np.zeros((len(Qsym), len(targets)), np.float32)
    k = 0
    while k < len(targets):
        n, tot = 0, 0
        while k + n < len(targets) and (n == 0 or tot + np.diff(off)[targets[k + n]] <= chunk_res):
            tot += np.diff(off)[targets[k + n]]
            n += 1
        ts = targets[k:k + n]
        if ts[-1] - ts[0] == n - 1:
            raw = np.asarray(sym[off[ts[0]]:off[ts[-1] + 1]])
            st = off[ts] - off[ts[0]]
        else:
            raw = np.concatenate([np.asarray(sym[off[p]:off[p + 1]]) for p in ts])
            st = np.r_[0, np.cumsum(np.diff(off)[ts])[:-1]]
        T = torch.from_numpy(np.ascontiguousarray(raw.reshape(len(raw), M)).astype(np.int32)).to(DEV)
        L = lens[k:k + n]
        prev = 0
        for hi in (64, 128, 256, 512, 1024, 2048, 4096):
            if hi > maxj:
                break
            sel = np.flatnonzero((L > prev) & (L <= hi))
            prev = hi
            if not len(sel):
                continue
            st_t, tl_t = torch.from_numpy(st[sel]).to(DEV), torch.from_numpy(L[sel]).to(DEV)
            scratch = torch.empty(len(sel) * (hi + 1), device=DEV, dtype=torch.float32)
            res = torch.empty(len(sel), device=DEV, dtype=torch.float32)
            for qi, Q in enumerate(Qg):
                _lut_kernel[(len(sel),)](Q, Q.shape[0], T, st_t, tl_t, tab, scratch, res, float(bias), float(gap),
                                         K=K, M=M, BLOCK_J=hi, num_warps=4 if hi <= 256 else (8 if hi <= 1024 else 16))
                out[qi, k + sel] = res.cpu().numpy()
        del T
        k += n
    return out


def _words(bits):
    """packed uint8 (n, 80) -> int64 (n, 10), byte order preserved (Hamming is order-independent)."""
    return np.ascontiguousarray(bits).view(np.int64)


def search(Qbits, bits, off, bias, gap, targets=None, chunk_res=4_000_000, pause=None, log=None):
    """Qbits: list of packed uint8 (Lq, 80) query codes. Same output as esw.search."""
    import time
    N = len(off) - 1
    targets = np.arange(N) if targets is None else np.asarray(targets)
    lens = np.diff(off)[targets]
    out = np.zeros((len(Qbits), len(targets)), np.float32)
    long_ = np.flatnonzero(lens > MAXJ)
    Qw = [torch.from_numpy(_words(q)).to(DEV) for q in Qbits]
    k = 0
    while k < len(targets):
        n, tot = 0, 0
        while k + n < len(targets) and (n == 0 or tot + lens[k + n] <= chunk_res):
            tot += lens[k + n]
            n += 1
        ts = targets[k:k + n]
        if pause is not None:
            while pause.exists():
                time.sleep(10)
        if ts[-1] - ts[0] == n - 1:
            raw = np.asarray(bits[off[ts[0]]:off[ts[-1] + 1]])
            st = off[ts] - off[ts[0]]
        else:
            raw = np.concatenate([np.asarray(bits[off[p]:off[p + 1]]) for p in ts])
            st = np.r_[0, np.cumsum(lens[k:k + n])[:-1]]
        T = torch.from_numpy(_words(raw)).to(DEV)
        L = lens[k:k + n]
        for lo, hi in ((0, 64), (64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096)):
            sel = np.flatnonzero((L > lo) & (L <= hi)) if lo else np.flatnonzero(L <= hi)
            if not len(sel):
                continue
            st_t = torch.from_numpy(st[sel]).to(DEV)
            tl_t = torch.from_numpy(L[sel]).to(DEV)
            scratch = torch.empty(len(sel) * (hi + 1), device=DEV, dtype=torch.float32)
            res = torch.empty(len(sel), device=DEV, dtype=torch.float32)
            for qi, Q in enumerate(Qw):
                _sw_kernel[(len(sel),)](Q, Q.shape[0], T, st_t, tl_t, scratch, res, float(bias), float(gap),
                                        NW=W, BLOCK_J=hi, num_warps=4 if hi <= 256 else (8 if hi <= 1024 else 16))
                out[qi, k + sel] = res.cpu().numpy()
        del T
        k += n
        if log is not None:
            log(f"    esw_triton: {k}/{len(targets)} targets")
    if len(long_):                                     # long targets: PyTorch engine
        Qs = [esw.unpack(q) for q in Qbits]
        out[:, long_] = esw.search(Qs, bits, off, bias, gap, targets=targets[long_], pause=pause)
    return out
