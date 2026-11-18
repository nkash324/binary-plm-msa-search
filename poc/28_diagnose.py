"""Why do d1cr5a1 / d2dzqa1 get no Atlas rows (and a wrong fold from BinPLM Swiss-Prot rows), and why does d1xg0a_
fold wrong despite 93 Atlas rows?  Truth = structure (TM-score vs the real SCOPe structure, normalised by the query).

Swiss-Prot part (BinPLM pipeline, combined settings, exactly as in 18_e2e.py):
  * MMseqs2's relatives (known good: AlphaFold folds d1cr5a1/d2dzqa1 correctly from them): where does BinPLM window
    index rank them (top 200 = candidate?) and how does their embedding-alignment score compare with the cutoff?
  * BinPLM accepted rows: are they the same fold?  (AlphaFold DB models, up to 40 per protein)
Atlas part (20-shard index):
  * d1cr5a1, d2dzqa1: top 100 candidates -> ESMFold structures -> any same fold?  their score vs the cutoff
  * d1xg0a_: the 93 accepted rows -> same fold?  how much of the protein do they cover?
Run in the 'fold' env.  Writes results/diagnose.md.
"""
import os
import json
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

import emb_align as ea
from common import DATA, DB_FASTA, RESULTS, read_fasta

E = DATA / "e2e"
A = Path.home() / "folding-data" / "atlas"
APDB, SPDB = A / "pdb", DATA / "afdb"
APDB.mkdir(exist_ok=True)
SPDB.mkdir(exist_ok=True)
USALIGN = os.path.expanduser("~") + "/miniforge3/envs/cf/bin/USalign"
QS = ["d1cr5a1", "d2dzqa1", "d1xg0a_"]
LOG = open(RESULTS / "diagnose.log", "a")
rng = np.random.default_rng(0)


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


def get(url, binary=False):
    for k in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                d = r.read()
            return d if binary else d.decode()
        except Exception:                                   # noqa: BLE001
            time.sleep(2 * (k + 1))
    return None


def atlas_pdb(mid):
    f = APDB / f"{mid}.pdb"
    if not f.exists():
        d = get(f"https://api.esmatlas.com/fetchPredictedStructure/{mid}")
        if not d:
            return None
        f.write_text(d)
    return f


def afdb_pdb(acc):
    f = SPDB / f"{acc}.pdb"
    if not f.exists():
        j = get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}")
        if not j:
            return None
        try:
            d = get(json.loads(j)[0]["pdbUrl"])
        except Exception:                                   # noqa: BLE001
            return None
        if not d:
            return None
        f.write_text(d)
    return f


def tm(f, q):
    if f is None:
        return np.nan
    o = subprocess.run([USALIGN, str(f), str(E / "ref" / f"{q}.ent"), "-outfmt", "2"], capture_output=True, text=True).stdout
    try:
        return float([l for l in o.splitlines() if l and not l.startswith("#")][0].split("\t")[3])
    except Exception:                                       # noqa: BLE001
        return np.nan


def tms(fetch, ids, q):
    with ThreadPoolExecutor(8) as ex:
        files = list(ex.map(fetch, ids))
    return np.array([tm(f, q) for f in files])


def headers(p, q):
    return [l[1:].split("\t")[0].strip() for l in (E / "a3m" / p / f"{q}.a3m").read_text().splitlines()
            if l.startswith(">")][1:]


qids, qseqs = read_fasta(E / "queries.fasta")
QSEQ = dict(zip(qids, qseqs))

# ================================================================ Swiss-Prot side (as 18_e2e.py)
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


def calibrate(Q, avoid):
    null = np.array([p for p in null_pool if p not in avoid])[:200]
    ns, _ = ea.sw_many(Q, [ea.pm1(np.array(R256[off[p]:off[p + 1]])) for p in null], BIAS, GAP)
    Al = np.c_[np.ones(len(null)), np.log(lens[null])]
    coef, *_ = np.linalg.lstsq(Al, ns, rcond=None)
    sd = np.std(ns - Al @ coef)
    zf = lambda s, L: (s - np.c_[np.ones(len(L)), np.log(L)] @ coef) / sd
    q90, q99 = np.quantile(zf(ns, lens[null]), [0.9, 0.99])
    beta = (q99 - q90) / (np.log(-np.log(0.9)) - np.log(-np.log(0.99)))
    return zf, q90 + beta * (-np.log(-np.log(1 - 1 / 2000)) + np.log(-np.log(0.9)))


S = {}
for q in QS:
    b640, Q = embed(QSEQ[q])
    sc = window_search(b640)
    order = np.argsort(-sc)
    rank = np.empty(NP, int)
    rank[order] = np.arange(1, NP + 1)
    cand = order[:200]
    cand = cand[sc[cand] > 0]
    zf, thr = calibrate(Q, set(cand.tolist()))
    mm = [a for a in headers("mmseqs2", q) if a in id2idx]
    ys = [a for a in headers("yours", q) if a in id2idx]
    mi = np.array([id2idx[a] for a in mm])
    msc, _ = ea.sw_many(Q, [ea.pm1(np.array(R256[off[p]:off[p + 1]])) for p in mi], BIAS, GAP)
    mz = zf(msc, lens[mi])
    ysamp = list(rng.choice(ys, min(40, len(ys)), replace=False)) if ys else []
    S[q] = {"mm": mm, "mm_rank": rank[mi].tolist(), "mm_z": mz.tolist(), "thr": float(thr),
            "mm_tm": tms(afdb_pdb, mm, q).tolist(), "yours_n": len(ys), "overlap": len(set(mm) & set(ys)),
            "yours_sample_tm": tms(afdb_pdb, ysamp, q).tolist()}
    log(f"Swiss-Prot {q}: MMseqs2 relatives {len(mm)}, in BinPLM top 200: {(rank[mi] <= 200).sum()}, "
        f"above cutoff: {(mz > thr).sum()}; BinPLM rows {len(ys)} (overlap {S[q]['overlap']}), "
        f"sampled same fold {np.nansum(np.array(S[q]['yours_sample_tm']) > 0.5)}/{len(ysamp)}")

# ================================================================ Atlas side (20-shard index)
names = [u.rsplit("/", 1)[1][:-4] for u in (RESULTS / "atlas20_shards.txt").read_text().split()]
codes = np.ascontiguousarray(np.concatenate([np.load(A / "idx20" / f"{n}.codes.npy") for n in names])).view(np.uint64)
ids = sum(((A / "idx20" / f"{n}.ids.txt").read_text().split() for n in names), [])
apca = np.load(A / "pca.npz")
amu, aP = apca["mu"], apca["P"]
tok3 = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
m3 = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D", add_pooling_layer=False, torch_dtype=torch.float16,
                              device_map="cuda").eval()
cache = json.loads((A / "seqs.json").read_text())
AT = {}
for q in ("d1cr5a1", "d2dzqa1"):
    with torch.no_grad():
        e = m3(**tok3([QSEQ[q]], return_tensors="pt").to("cuda")).last_hidden_state[0, 1:len(QSEQ[q]) + 1].float().mean(0).cpu().numpy()
    qc = np.ascontiguousarray(np.packbits(((e - amu) @ aP > 0)[None], axis=1)).view(np.uint64)
    ham = np.bitwise_count(codes ^ qc).sum(1)
    top = np.argpartition(ham, 100)[:100]
    top = top[np.lexsort((top, ham[top]))]
    tid = [ids[i] for i in top]
    t = tms(atlas_pdb, tid, q)
    _, Q = embed(QSEQ[q])
    zf, thr = calibrate(Q, set())
    have = [k for k, m in enumerate(tid) if m in cache]
    sc, _ = ea.sw_many(Q, [embed(cache[tid[k]])[1] for k in have], BIAS, GAP)
    zz = np.full(len(tid), np.nan)
    zz[have] = zf(sc, np.array([len(cache[tid[k]][:1022]) for k in have]))
    AT[q] = {"tm": t.tolist(), "z": zz.tolist(), "thr": float(thr), "ham": ham[top].tolist()}
    same = np.flatnonzero(t > 0.5)
    log(f"Atlas {q}: same-fold in top 100: {len(same)} (best TM {np.nanmax(t):.2f}); their z "
        f"{np.round(zz[same], 1).tolist()} vs cutoff {thr:.1f}")
del m3
torch.cuda.empty_cache()

q = "d1xg0a_"
rows = [l for l in (E / "a3m" / "atlas20_only" / f"{q}.a3m").read_text().splitlines()][2:]
acc = [rows[k][1:].strip() for k in range(0, len(rows), 2)]
cov = np.array([sum(c != "-" for c in rows[k]) / len(QSEQ[q]) for k in range(1, len(rows), 2)])
t = tms(atlas_pdb, acc, q)
mmt = S[q]["mm_tm"]
AT[q] = {"tm": t.tolist(), "coverage": cov.tolist()}
log(f"Atlas {q}: accepted {len(acc)}, same fold {np.nansum(t > 0.5)}, median TM {np.nanmedian(t):.2f}, "
    f"median coverage {np.median(cov):.2f}")
(RESULTS / "diagnose.json").write_text(json.dumps({"swissprot": S, "atlas": AT}, indent=1))

# ================================================================ report
L = ["# Diagnosis: d1cr5a1, d2dzqa1 (no Atlas rows, wrong with BinPLM Swiss-Prot rows) and d1xg0a_ (93 Atlas rows, still wrong)",
     "", "Truth = structure: TM-score of each relative's predicted model (AlphaFold DB for Swiss-Prot, ESMFold for Atlas) vs",
     "the protein's real structure, normalised by the protein. TM > 0.5 = same fold.", "",
     "## Swiss-Prot: MMseqs2's relatives vs BinPLM pipeline", "",
     "| protein | length | MMseqs2 relatives | same fold (of those with a model) | in BinPLM top 200 | best rank | above BinPLM cutoff | "
     "best z vs cutoff | BinPLM rows | overlap with MMseqs2 | BinPLM rows same fold (sample) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for q in QS:
    s = S[q]
    r, zz, mt, yt = np.array(s["mm_rank"]), np.array(s["mm_z"]), np.array(s["mm_tm"]), np.array(s["yours_sample_tm"])
    L.append(f"| {q} | {len(QSEQ[q])} | {len(s['mm'])} | {np.nansum(mt > 0.5)}/{np.isfinite(mt).sum()} | {(r <= 200).sum()} | "
             f"{r.min() if len(r) else '-'} | {(zz > s['thr']).sum()} | {np.max(zz) if len(zz) else np.nan:.1f} vs {s['thr']:.1f} | "
             f"{s['yours_n']} | {s['overlap']} | {np.nansum(yt > 0.5)}/{np.isfinite(yt).sum()} |")
L += ["", "## Atlas (20 shards, 7.9M proteins)", "", "| protein | top-100 same fold | best TM | same-fold above cutoff |",
      "|---|---|---|---|"]
for q in ("d1cr5a1", "d2dzqa1"):
    t, zz = np.array(AT[q]["tm"]), np.array(AT[q]["z"])
    L.append(f"| {q} | {np.nansum(t > 0.5)} | {np.nanmax(t):.2f} | {np.nansum((t > 0.5) & (zz > AT[q]['thr']))} |")
t, cov = np.array(AT["d1xg0a_"]["tm"]), np.array(AT["d1xg0a_"]["coverage"])
L += ["", f"d1xg0a_: {len(t)} accepted Atlas rows; same fold {np.nansum(t > 0.5)}/{np.isfinite(t).sum()}, "
      f"median TM {np.nanmedian(t):.2f}; median coverage of the protein {np.median(cov):.0%}. "
      f"MMseqs2 relatives same fold: {np.nansum(np.array(mmt) > 0.5)}/{np.isfinite(mmt).sum()}."]
(RESULTS / "diagnose.md").write_text("\n".join(L) + "\n")
log("\n".join(L))
