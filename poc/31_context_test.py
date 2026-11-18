"""Is it CONTEXT?  ESM-2 embeddings depend on the surrounding sequence. d1cr5a1 / d2dzqa1 are small domains cut out
of big proteins; their relatives in Swiss-Prot are embedded at full length.  Test, with the known MMseqs2 relatives:

  1. query as an isolated domain (what the pipeline does)                    -> window-search rank, alignment z
  2. query IN CONTEXT: embed the identical full-length parent protein, keep only the domain's residues
                                                                              -> window-search rank, alignment z
  3. relatives CUT to their domain region and embedded alone (isolated vs isolated) -> alignment z
Cutoff = the usual random-protein chance level.  Run in the 'fold' env.  Writes results/context_test.md."""
import json
import time

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

import emb_align as ea
from common import DATA, DB_FASTA, RESULTS, read_fasta

E = DATA / "e2e"
QS = ["d1cr5a1", "d2dzqa1"]
LOG = open(RESULTS / "context_test.log", "a")


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


qids, qseqs = read_fasta(E / "queries.fasta")
QSEQ = dict(zip(qids, qseqs))
db_ids, db_seqs = read_fasta(DB_FASTA)
id2idx = {d: i for i, d in enumerate(db_ids)}
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
def embed(seq, a=None, b=None):
    """Embed seq (first 1022 residues); return 640-bit packed signs and 256-bit codes of residues a..b (default all)."""
    seq = seq[:1022]
    enc = tok([seq], return_tensors="pt")
    r = esm(input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda()).last_hidden_state.float()
    sgn = (r[0, 1:len(seq) + 1] - mu_res >= 0)[a:b]
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


def zfun(Q):
    null = null_pool[:200]
    ns, _ = ea.sw_many(Q, [sp(p) for p in null], BIAS, GAP)
    Al = np.c_[np.ones(len(null)), np.log(lens[null])]
    coef, *_ = np.linalg.lstsq(Al, ns, rcond=None)
    sd = np.std(ns - Al @ coef)
    zf = lambda s, L: (s - np.c_[np.ones(len(L)), np.log(L)] @ coef) / sd
    q90, q99 = np.quantile(zf(ns, lens[null]), [0.9, 0.99])
    beta = (q99 - q90) / (np.log(-np.log(0.9)) - np.log(-np.log(0.99)))
    return zf, q90 + beta * (-np.log(-np.log(1 - 1 / 2000)) + np.log(-np.log(0.9)))


def relatives(q):
    out = []
    for l in (E / "a3m" / "mmseqs2" / f"{q}.a3m").read_text().splitlines():
        if l.startswith(">") and "\t" in l:
            c = l[1:].split("\t")
            if c[0] in id2idx:
                out.append({"acc": c[0], "ident": float(c[2]), "ts": int(c[7]), "te": int(c[8])})
    return out


R, L = {}, []
for q in QS:
    rel = relatives(q)
    ri = np.array([id2idx[r["acc"]] for r in rel])
    parent = max(rel, key=lambda r: r["ident"])
    res = {}
    for mode in ("isolated", "in context"):
        if mode == "isolated":
            b640, Q = embed(QSEQ[q])
        else:
            b640, Q = embed(db_seqs[id2idx[parent["acc"]]], parent["ts"], parent["te"] + 1)
        sc = window_search(b640)
        rank = (sc[None, :] > sc[ri][:, None]).sum(1) + 1
        zf, thr = zfun(Q)
        s, _ = ea.sw_many(Q, [sp(p) for p in ri], BIAS, GAP)          # relatives at full length (as stored)
        zz = zf(s, lens[ri])
        res[mode] = {"rank": rank.tolist(), "z": zz.tolist(), "thr": float(thr)}
        if mode == "isolated":                                          # 3. relatives cut to their domain region
            cut = [embed(db_seqs[p], r["ts"], r["te"] + 1)[1] for p, r in zip(ri, rel)]
            s2, _ = ea.sw_many(Q, cut, BIAS, GAP)
            res["relatives cut"] = {"z": zf(s2, np.array([r["te"] - r["ts"] + 1 for r in rel])).tolist(), "thr": float(thr)}
        log(f"{q} query {mode}: best rank {rank.min()}, in top 200: {(rank <= 200).sum()}/{len(ri)}, "
            f"above cutoff {(zz > thr).sum()}/{len(ri)} (max z {zz.max():.1f}, cutoff {thr:.1f})")
    c = res["relatives cut"]
    log(f"{q} relatives cut to the domain: above cutoff {(np.array(c['z']) > c['thr']).sum()}/{len(ri)} "
        f"(max z {max(c['z']):.1f})")
    R[q] = {"relatives": rel, **res}
(RESULTS / "context_test.json").write_text(json.dumps(R, indent=1))

L = ["# Context test: small domains cut out of big proteins", "",
     "Relatives = MMseqs2's Swiss-Prot hits (all same fold; AlphaFold folds these proteins correctly from them).",
     "Window rank = where BinPLM fast search puts them among 576k proteins (top 200 become candidates).",
     "z = embedding-alignment score; above the cutoff = accepted.", "",
     "| protein | setup | relatives in top 200 | best rank | median rank | above cutoff | max z (cutoff) |",
     "|---|---|---|---|---|---|---|"]
for q in QS:
    for mode, lab in (("isolated", "query = isolated domain (normal)"), ("in context", "query = domain embedded inside its full parent protein")):
        r, zz = np.array(R[q][mode]["rank"]), np.array(R[q][mode]["z"])
        L.append(f"| {q} | {lab} | {(r <= 200).sum()}/{len(r)} | {r.min():,} | {int(np.median(r)):,} | "
                 f"{(zz > R[q][mode]['thr']).sum()}/{len(zz)} | {zz.max():.1f} ({R[q][mode]['thr']:.1f}) |")
    c = np.array(R[q]["relatives cut"]["z"])
    L.append(f"| {q} | query isolated, relatives cut to their domain | - | - | - | {(c > R[q]['relatives cut']['thr']).sum()}/{len(c)} | "
             f"{c.max():.1f} ({R[q]['relatives cut']['thr']:.1f}) |")
(RESULTS / "context_test.md").write_text("\n".join(L) + "\n")
log("\n".join(L))
