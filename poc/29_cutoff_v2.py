"""Fairer acceptance cutoff (v2): a relative must beat BOTH
   (a) the old chance level: 200 random Swiss-Prot proteins (Gumbel tail, 1-in-2000), and
   (b) a look-alike chance level: the search's OWN candidates ranked 1000-1099 (embedding look-alikes that are mostly
       not relatives).  Fitted from the 50th/90th percentiles (robust if a few true relatives sneak in), same tail.
Everything else unchanged.  Rebuilds two MSAs for the 19 fresh proteins:
   yours_v2    : BinPLM Swiss-Prot pipeline (window index -> 200 candidates -> embedding alignment)
   atlas20_v2  : Atlas 20-shard pipeline (per-protein index -> 100 candidates -> embedding alignment)
Run in the 'fold' env.  Writes e2e/a3m/{yours_v2, atlas20_v2} and e2e/cutoff_v2.json."""
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
LOG = open(RESULTS / "cutoff_v2.log", "a")
G = lambda p: -np.log(-np.log(p))                           # Gumbel quantile function (standardised)
TAIL = 1 - 1 / 2000


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


qs = sorted({k.split("|")[1] for k in json.loads((RESULTS / "e2e.json").read_text())["accuracy"]})
qids, qseqs = read_fasta(E / "queries.fasta")
QSEQ = dict(zip(qids, qseqs))

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


def thresholds(Q, avoid, la_codes, la_lens):
    """(a) random-protein chance level and z-function, as before; (b) look-alike chance level on the same z scale."""
    null = np.array([p for p in null_pool if p not in avoid])[:200]
    ns, _ = ea.sw_many(Q, [sp(p) for p in null], BIAS, GAP)
    Al = np.c_[np.ones(len(null)), np.log(lens[null])]
    coef, *_ = np.linalg.lstsq(Al, ns, rcond=None)
    sd = np.std(ns - Al @ coef)
    zf = lambda s, L: (s - np.c_[np.ones(len(L)), np.log(L)] @ coef) / sd
    q90, q99 = np.quantile(zf(ns, lens[null]), [0.9, 0.99])
    thr_a = q90 + (q99 - q90) / (G(0.99) - G(0.9)) * (G(TAIL) - G(0.9))
    thr_b = -np.inf
    if len(la_codes) >= 50:
        ls, _ = ea.sw_many(Q, la_codes, BIAS, GAP)
        lz = zf(ls, np.asarray(la_lens))
        q50, q90b = np.quantile(lz, [0.5, 0.9])
        thr_b = q50 + (q90b - q50) / (G(0.9) - G(0.5)) * (G(TAIL) - G(0.5))
    return zf, float(thr_a), float(thr_b)


def rows_for(Q, seq, targets, tseqs, keep):
    out = []
    if len(keep):
        _, pr = ea.sw_many(Q, [targets[k] for k in keep], BIAS, GAP, keep=True)
        for k, pairs in zip(keep, pr):
            r = ["-"] * len(seq)
            for i, j in pairs:
                r[i] = tseqs[k][j]
            out.append(r)
    return ["".join(r) for r in out]


def write(d, q, names, rows):
    (E / "a3m" / d).mkdir(parents=True, exist_ok=True)
    head = "\n".join((E / "a3m" / "mmseqs2" / f"{q}.a3m").read_text().splitlines()[:2]) + "\n"
    (E / "a3m" / d / f"{q}.a3m").write_text(head + "".join(f">{n}\n{r}\n" for n, r in zip(names, rows)))


def old_set(d, q):
    return {l[1:].split("\t")[0].strip() for l in (E / "a3m" / d / f"{q}.a3m").read_text().splitlines()
            if l.startswith(">")} - {q}


OUT = {"yours_v2": {}, "atlas20_v2": {}}
# ================================================================ Swiss-Prot (BinPLM pipeline)
EMB = {}
for q in qs:
    seq = QSEQ[q]
    b640, Q = embed(seq)
    EMB[q] = Q
    sc = window_search(b640)
    order = np.argsort(-sc)
    cand = order[:200]
    cand = cand[sc[cand] > 0]
    la = order[1000:1100]
    la = la[sc[la] > 0]
    zf, ta, tb = thresholds(Q, set(cand.tolist()), [sp(p) for p in la], lens[la])
    Tc = [sp(p) for p in cand]
    cs, _ = ea.sw_many(Q, Tc, BIAS, GAP)
    zc = zf(cs, lens[cand])
    keep = np.flatnonzero(zc > max(ta, tb))
    names = [db_ids[cand[k]] for k in keep]
    write("yours_v2", q, names, rows_for(Q, seq, Tc, [db_seqs[p] for p in cand], keep))
    old = old_set("yours", q)
    OUT["yours_v2"][q] = {"old": len(old), "new": len(keep), "thr_random": ta, "thr_lookalike": tb,
                          "changed": set(names) != old}
    log(f"Swiss-Prot {q}: cutoff random {ta:.1f}, look-alike {tb:.1f} -> rows {len(old)} -> {len(keep)}")

# ================================================================ Atlas 20 shards
names20 = [u.rsplit("/", 1)[1][:-4] for u in (RESULTS / "atlas20_shards.txt").read_text().split()]
codes = np.ascontiguousarray(np.concatenate([np.load(A / "idx20" / f"{n}.codes.npy") for n in names20])).view(np.uint64)
ids = sum(((A / "idx20" / f"{n}.ids.txt").read_text().split() for n in names20), [])
apca = np.load(A / "pca.npz")
amu, aP = apca["mu"], apca["P"]
tok3 = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
m3 = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D", add_pooling_layer=False, torch_dtype=torch.float16,
                              device_map="cuda").eval()
CAND, LA = {}, {}
with torch.no_grad():
    for q in qs:
        s = QSEQ[q]
        e = m3(**tok3([s], return_tensors="pt").to("cuda")).last_hidden_state[0, 1:len(s) + 1].float().mean(0).cpu().numpy()
        qc = np.ascontiguousarray(np.packbits(((e - amu) @ aP > 0)[None], axis=1)).view(np.uint64)
        ham = np.bitwise_count(codes ^ qc).sum(1)
        top = np.argpartition(ham, 1100)[:1100]
        top = top[np.lexsort((top, ham[top]))]
        CAND[q], LA[q] = top[:100], top[1000:1100]
del m3
torch.cuda.empty_cache()

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


need = sorted({int(i) for q in qs for i in list(CAND[q]) + list(LA[q])})
with ThreadPoolExecutor(8) as ex:
    for k, s in ex.map(fetch, need):
        if s:
            cache[k] = s
cache_f.write_text(json.dumps(cache))
log(f"Atlas sequences available: {sum(ids[i] in cache for i in need)}/{len(need)}")

for q in qs:
    seq, Q = QSEQ[q], EMB[q]
    ci = [ids[i] for i in CAND[q] if ids[i] in cache]
    li = [ids[i] for i in LA[q] if ids[i] in cache]
    Tl = [embed(cache[m])[1] for m in li]
    zf, ta, tb = thresholds(Q, set(), Tl, [len(cache[m][:1022]) for m in li])
    Tc = [embed(cache[m])[1] for m in ci]
    cs, _ = ea.sw_many(Q, Tc, BIAS, GAP)
    zc = zf(cs, np.array([len(cache[m][:1022]) for m in ci]))
    keep = np.flatnonzero(zc > max(ta, tb))
    names = [ci[k] for k in keep]
    write("atlas20_v2", q, names, rows_for(Q, seq, Tc, [cache[m][:1022] for m in ci], keep))
    old = old_set("atlas20_only", q)
    OUT["atlas20_v2"][q] = {"old": len(old), "new": len(keep), "thr_random": ta, "thr_lookalike": tb,
                            "changed": set(names) != old}
    log(f"Atlas {q}: cutoff random {ta:.1f}, look-alike {tb:.1f} -> rows {len(old)} -> {len(keep)}")
(E / "cutoff_v2.json").write_text(json.dumps(OUT, indent=1, default=bool))
log("MSAS DONE")
