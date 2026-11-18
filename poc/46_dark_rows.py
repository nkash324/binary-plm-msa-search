"""Test A, stage 3: for the shallow proteins, add relatives found by BinPLM on top of the big-database MSA.
  BinPLM : BinPLM Swiss-Prot pipeline, combined settings, unchanged (window index -> 200 candidates -> embedding alignment
          -> chance cutoff), query = the domain (as in every earlier test)
  atlas : full Atlas index (all downloaded shards, ~75M proteins): ESM-2 3B vector -> 256 bits -> top 100 -> sequences
          from the Atlas API -> same embedding alignment and cutoff
MSAs written (e2e/dark/a3m/<set>/<id>.a3m): bigdb (server MSA as is), bigdb_yours, bigdb_atlas, bigdb_both.
Run in the 'fold' env (needs ~7 GB WSL RAM for the 3B model)."""
import os
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

import emb_align as ea
from common import DATA, DB_FASTA, RESULTS, read_fasta

E, A = DATA / "e2e", Path.home() / "folding-data" / "atlas"
D = E / "dark"
LOG = open(RESULTS / "overnight.log", "a")


def log(m):
    m = f"{time.strftime('%H:%M')} A: {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


shallow = [l.strip() for l in (D / os.environ.get("A_SHALLOW", "shallow.txt")).read_text().split() if l.strip()]
cids, cseqs = read_fasta(D / "candidates.fasta")
QSEQ = dict(zip(cids, cseqs))
log(f"building rows for {len(shallow)} shallow proteins")

# ---------------------------------------------------------------- Atlas: 3B query codes come from 46a_atlas_qvec.py
import subprocess, sys                                     # noqa: E401,E402  (separate process: 3B memory fully freed)
r = subprocess.run([sys.executable, "46a_atlas_qvec.py"], capture_output=True, text=True)
if r.returncode:
    log("Atlas query vectors FAILED: " + r.stderr.strip().splitlines()[-1])
    raise SystemExit(1)
log(r.stdout.strip().splitlines()[-1])
qz = np.load(D / "atlas_qcode.npz")
qcode = {q: np.ascontiguousarray(qz[q]).view(np.uint64) for q in shallow}
names = [u.rsplit("/", 1)[1][:-4] for u in (RESULTS / "atlas_all_shards.txt").read_text().split()]
names = [n for n in names if (A / "idx20" / f"{n}.ids.txt").exists()]
cand, base = {q: [] for q in shallow}, 0
for si, n in enumerate(names):                             # stream shard by shard: never hold all codes or ids at once
    c = np.ascontiguousarray(np.load(A / "idx20" / f"{n}.codes.npy")).view(np.uint64)
    for q in shallow:
        h = np.bitwise_count(c ^ qcode[q]).sum(1)
        k = min(100, len(h) - 1)
        t = np.argpartition(h, k)[:k + 1] if len(h) > 100 else np.arange(len(h))
        cand[q] += [(int(h[i]), si, int(i)) for i in t]
        cand[q] = sorted(cand[q])[:100]
    base += len(c)
    del c
ids = {}                                                   # (shard, row) -> Atlas id, only for the kept hits
need = {}
for q in shallow:
    cand[q] = [(si, i) for _, si, i in cand[q]]
    for si, i in cand[q]:
        need.setdefault(si, set()).add(i)
for si, rows_ in need.items():
    sid_n = (A / "idx20" / f"{names[si]}.ids.txt").read_text().split()
    for i in rows_:
        ids[(si, i)] = sid_n[i]
    del sid_n
log(f"Atlas index: {len(names)} shards, {base:,} proteins searched")

cache_f = A / "seqs.json"
cache = json.loads(cache_f.read_text())


def fetch(i):
    if ids[i] in cache:
        return ids[i], cache[ids[i]]
    for k in range(4):
        try:
            with urllib.request.urlopen(f"https://api.esmatlas.com/fetchSequence/{ids[i]}", timeout=60) as r:
                return ids[i], json.loads(r.read())["sequence"]
        except Exception:                                   # noqa: BLE001
            time.sleep(2 * (k + 1))
    return ids[i], None


with ThreadPoolExecutor(8) as ex:
    for k, s in ex.map(fetch, sorted({i for c in cand.values() for i in c})):
        if s:
            cache[k] = s
cache_f.write_text(json.dumps(cache))

# ---------------------------------------------------------------- shared pieces of BinPLM pipeline (as 18_e2e.py)
db_ids, db_seqs = read_fasta(DB_FASTA)
NP = len(db_ids)
off = np.load(DATA / "offsets.npy")
lens = np.diff(off)
R256 = np.load(DATA / "res_bits256.npy", mmap_mode="r")
PRM = json.loads((DATA / "emb_align_params_256.json").read_text())
BIAS, GAP = PRM["bias"], PRM["gap"]
pca = np.load(DATA / "pca_res.npz")
gmu, gP = torch.from_numpy(pca["mu"]).cuda(), torch.from_numpy(pca["P"][:, :256].copy()).cuda()
z = np.load(DATA / "win64_256.npz")
wcodes = np.ascontiguousarray(z["codes"]).view(np.uint64)
owner, WMU, WPM = z["owner"], z["mu"], z["P"]
null_pool = np.random.default_rng(42).choice(NP, 2000, replace=False)
mu_res = torch.from_numpy(np.load(DATA / "mu.npz")["res"]).cuda()
tok = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
esm = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D", add_pooling_layer=False,
                               attn_implementation="sdpa").eval().cuda().half()
SH8 = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], device="cuda", dtype=torch.uint8)
sp = lambda p: ea.pm1(np.array(R256[off[p]:off[p + 1]]))


@torch.no_grad()
def embed(seq):
    seq = seq[:1022]
    enc = tok([seq], return_tensors="pt")
    r = esm(input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda()).last_hidden_state.float()
    sgn = r[0, 1:len(seq) + 1] - mu_res >= 0
    b640 = (sgn.view(-1, 80, 8).to(torch.uint8) * SH8).sum(-1, dtype=torch.uint8).cpu().numpy()
    return b640, (((sgn.half() * 2 - 1).float() - gmu) @ gP > 0).half() * 2 - 1


def window_search(b640):
    X = np.unpackbits(b640, axis=1).astype(np.float32) * 2 - 1
    L, W, B, TOPW = len(X), 64, 256, 2000
    starts = sorted(set(list(range(0, max(1, L - W + 1), W // 4)) + [max(0, L - W)]))
    qc = np.ascontiguousarray(np.packbits((np.stack([X[a:a + W].mean(0) for a in starts]) - WMU) @ WPM > 0,
                                          axis=1)).view(np.uint64)
    score = np.zeros(NP, np.float32)
    for lo in range(0, len(wcodes), 500_000):
        c = wcodes[lo:lo + 500_000]
        h = np.zeros((len(qc), len(c)), np.uint16)
        for w in range(c.shape[1]):
            h += np.bitwise_count(c[None, :, w] ^ qc[:, None, w])
        i = np.argpartition(h, TOPW, axis=1)[:, :TOPW]
        zz = (B / 2 - np.take_along_axis(h, i, 1).astype(np.float32)) / (np.sqrt(B) / 2)
        for j in range(len(qc)):
            o = np.argsort(-zz[j])
            pp, first = np.unique(owner[lo + i[j][o]], return_index=True)
            score[pp] += np.maximum(zz[j][o][first], 0)
    return score


def chance(Q, avoid):
    null = np.array([p for p in null_pool if p not in avoid])[:200]
    ns, _ = ea.sw_many(Q, [sp(p) for p in null], BIAS, GAP)
    Al = np.c_[np.ones(len(null)), np.log(lens[null])]
    coef, *_ = np.linalg.lstsq(Al, ns, rcond=None)
    sd = np.std(ns - Al @ coef)
    zf = lambda s, L: (s - np.c_[np.ones(len(L)), np.log(L)] @ coef) / sd
    q90, q99 = np.quantile(zf(ns, lens[null]), [0.9, 0.99])
    beta = (q99 - q90) / (np.log(-np.log(0.9)) - np.log(-np.log(0.99)))
    return zf, q90 + beta * (-np.log(-np.log(1 - 1 / 2000)) + np.log(-np.log(0.9)))


def rows_from(Q, seq, targets, tseqs, keep, names_):
    out = []
    if len(keep):
        _, pr = ea.sw_many(Q, [targets[k] for k in keep], BIAS, GAP, keep=True)
        for k, pairs in zip(keep, pr):
            r = ["-"] * len(seq)
            for i, j in pairs:
                r[i] = tseqs[k][j]
            out.append(f">{names_[k]}\n{''.join(r)}\n")
    return "".join(out)


stats = {}
for d in ("bigdb", "bigdb_yours", "bigdb_atlas", "bigdb_both"):
    (D / "a3m" / d).mkdir(parents=True, exist_ok=True)
for q in shallow:
    s = QSEQ[q]
    b640, Q = embed(s)
    # BinPLM (Swiss-Prot)
    sc = window_search(b640)
    cy = np.argsort(-sc)[:200]
    cy = cy[sc[cy] > 0]
    zf, thr = chance(Q, set(cy.tolist()))
    Ty = [sp(p) for p in cy]
    sy, _ = ea.sw_many(Q, Ty, BIAS, GAP)
    ky = np.flatnonzero(zf(sy, lens[cy]) > thr)
    add_y = rows_from(Q, s, Ty, [db_seqs[p] for p in cy], ky, [db_ids[p] for p in cy])
    # atlas (hit sequences embedded with the same 150M model, first 1022 residues, as in earlier Atlas tests)
    seen = set()                                           # same Atlas id can sit in several shards: keep once
    ca = [i for i in cand[q] if ids[i] in cache and not (ids[i] in seen or seen.add(ids[i]))]
    Ta = [embed(cache[ids[i]])[1] for i in ca]
    zf2, thr2 = chance(Q, set())
    sa, _ = ea.sw_many(Q, Ta, BIAS, GAP)
    ka = np.flatnonzero(zf2(sa, np.array([len(cache[ids[i]][:1022]) for i in ca])) > thr2)
    add_a = rows_from(Q, s, Ta, [cache[ids[i]][:1022] for i in ca], ka, [ids[i] for i in ca])
    big = (D / "msa" / f"{q}.a3m").read_text().replace("\x00", "")
    big = big if big.endswith("\n") else big + "\n"
    for d, extra in (("bigdb", ""), ("bigdb_yours", add_y), ("bigdb_atlas", add_a), ("bigdb_both", add_y + add_a)):
        (D / "a3m" / d / f"{q}.a3m").write_text(big + extra)
    stats[q] = {"yours_rows": int(len(ky)), "atlas_rows": int(len(ka))}
    log(f"  {q}: + {len(ky)} Swiss-Prot rows (BinPLM), + {len(ka)} Atlas rows")
(D / os.environ.get("A_ROWS", "rows.json")).write_text(json.dumps(stats, indent=1))
log("ROWS DONE")
