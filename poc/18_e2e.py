"""End-to-end comparison on 20 FRESH hard proteins: sequence -> MSA, timed per stage, 4 pipelines.

  jackhmmer : AlphaFold-style (AF2 flags, 1 iteration, 6 CPU threads) -> its own Stockholm alignment -> a3m
  mmseqs2   : ColabFold-style (3 iterations, -s 8, 6 threads, pre-built index) -> filterresult -> result2msa (a3m)
  BinPLM     : combined settings (embed query -> window index -> 200 candidates -> 256-bit embedding alignment,
              200-random calibration with fitted tail) -> a3m
  hybrid    : MMseqs2 a3m + BinPLM relatives that MMseqs2 did not find

Fresh proteins: SCOPe 40% domains (classes a-d, length 60-400) that are not among the 100 earlier test proteins
and not in any superfamily used earlier (the 100 test proteins' or the aligner-tuning pairs').  "Hard" = both
JackHMMER and MMseqs2 find <= 20 relatives in Swiss-Prot; 20 picked at random from those.
Everything is run one protein at a time, after a warm-up protein (so each tool's data is in memory).
Writes ~/folding-data/poc/e2e/{a3m/<pipeline>/<q>.a3m, times.json, queries.*}.  Run in the 'fold' env.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import emb_align as ea
from common import DATA, DB_FASTA, PHASE1, RESULTS, read_fasta

E = DATA / "e2e"
E.mkdir(exist_ok=True)
LOG = open(RESULTS / "e2e.log", "a")
PIPES = ["jackhmmer", "mmseqs2", "yours", "hybrid"]
torch.cuda.set_per_process_memory_fraction(0.6)


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


def sh(*a, **k):
    return subprocess.run([str(x) for x in a], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **k)


# ================================================================ 0. pick fresh hard proteins
spec = importlib.util.spec_from_file_location("p1common", "/mnt/c/Folding/phase1/common.py")
p1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p1)
meta = p1.load_meta()
sid, sccs, fam, sf = meta["sid"], meta["sccs"], meta["fam"], meta["sf"]
_, p1seqs = read_fasta(PHASE1 / "seqs.fasta")
sf_of = lambda c: ".".join(c.split(".")[:3])
poc_q = [l.split("\t") for l in (DATA / "queries.tsv").read_text().splitlines()[1:]]
excl_sf = {sf_of(c) for _, c, _ in poc_q}
# rebuild the aligner-tuning pairs exactly as 09_align_check.py drew them
off1 = np.load(PHASE1 / "offsets.npy")
s1 = [s[:off1[i + 1] - off1[i]] for i, s in enumerate(p1seqs)]
ok1 = np.array([sf_of(c) not in excl_sf and len(q) <= 500 for c, q in zip(sccs, s1)])
r1, by_sf, pairs = np.random.default_rng(1), {}, set()
for i in np.flatnonzero(ok1):
    by_sf.setdefault(sf[i], []).append(i)
groups = [g for g in by_sf.values() if len(set(fam[g])) > 1]
while len(pairs) < 300:
    g = groups[r1.integers(len(groups))]
    a, b = r1.choice(g, 2, replace=False)
    if fam[a] != fam[b]:
        pairs.add((min(a, b), max(a, b)))
excl_sf |= {sf_of(sccs[i]) for p in pairs for i in p}
used = {s for s, _, _ in poc_q}
STD = set("ACDEFGHIKLMNPQRSTVWY")
pool = [i for i in range(len(sid)) if sccs[i][0] in "abcd" and 60 <= len(p1seqs[i]) <= 400
        and set(p1seqs[i]) <= STD and sid[i] not in used and sf_of(sccs[i]) not in excl_sf
        and (PHASE1 / "ent" / f"{sid[i]}.ent").exists()]
log(f"fresh pool: {len(pool)} domains outside {len(excl_sf)} excluded superfamilies")
QF = E / "queries.fasta"
if not QF.exists():
    rng = np.random.default_rng(123)
    pool_s = sorted(rng.choice(pool, min(800, len(pool)), replace=False))
    W0 = E / "select"
    shutil.rmtree(W0, ignore_errors=True)
    W0.mkdir()
    (W0 / "pool.fa").write_text("".join(f">{sid[i]}\n{p1seqs[i]}\n" for i in pool_s))
    t0 = time.time()
    sh("mmseqs", "createdb", W0 / "pool.fa", W0 / "q")
    sh("mmseqs", "search", W0 / "q", DATA / "mmseqs_work" / "db", W0 / "res", W0 / "tmp", "--num-iterations", 3,
       "-s", 8, "-e", 0.1, "--max-seqs", 10000, "--threads", 6)
    sh("mmseqs", "convertalis", W0 / "q", DATA / "mmseqs_work" / "db", W0 / "res", W0 / "res.m8")
    mmc = defaultdict(set)
    for l in open(W0 / "res.m8"):
        c = l.split("\t")
        mmc[c[0]].add(c[1])
    log(f"  MMseqs2 on {len(pool_s)} pool proteins: {time.time() - t0:.0f}s")

    def jk_count(i):
        fa, tb = W0 / f"{sid[i]}.fa", W0 / f"{sid[i]}.tbl"
        fa.write_text(f">{sid[i]}\n{p1seqs[i]}\n")
        sh("jackhmmer", "-N", 1, "--F1", 0.0005, "--F2", 0.00005, "--F3", 0.0000005, "--incE", 0.0001, "-E", 0.0001,
           "--cpu", 1, "-o", "/dev/null", "--tblout", tb, fa, DB_FASTA)
        return len({l.split()[0] for l in open(tb) if not l.startswith("#")})
    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        jkc = dict(zip(pool_s, ex.map(jk_count, pool_s)))
    log(f"  JackHMMER on {len(pool_s)} pool proteins: {time.time() - t0:.0f}s")
    hard = [i for i in pool_s if max(len(mmc[sid[i]]), jkc[i]) <= 20]
    if len(hard) < 21:                                            # fallback: the 21 with fewest relatives
        hard = sorted(pool_s, key=lambda i: max(len(mmc[sid[i]]), jkc[i]))[:21]
    pick = sorted(rng.choice(hard, 21, replace=False))           # 1 warm-up + 20 test
    log(f"  {len(hard)} of {len(pool_s)} are hard (<=20 relatives for both); picked 21")
    QF.write_text("".join(f">{sid[i]}\n{p1seqs[i]}\n" for i in pick))
    (E / "queries.tsv").write_text("sid\tsccs\tlength\tmmseqs_hits\tjackhmmer_hits\n" + "".join(
        f"{sid[i]}\t{sccs[i]}\t{len(p1seqs[i])}\t{len(mmc[sid[i]])}\t{jkc[i]}\n" for i in pick))
    shutil.rmtree(W0, ignore_errors=True)
qids, qseqs = read_fasta(QF)
REF = E / "ref"
REF.mkdir(exist_ok=True)
for q in qids:
    shutil.copy(PHASE1 / "ent" / f"{q}.ent", REF / f"{q}.ent")

# ================================================================ shared data for BinPLM
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
from transformers import AutoTokenizer, EsmModel                                   # noqa: E402
t0 = time.time()
tok = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
esm = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D", add_pooling_layer=False,
                               attn_implementation="sdpa").eval().cuda().half()
SETUP = {"esm_model_load_s": time.time() - t0}
SH8 = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], device="cuda", dtype=torch.uint8)


@torch.no_grad()
def embed(seq):
    enc = tok([seq], return_tensors="pt")
    r = esm(input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda()).last_hidden_state.float()
    r = r[0, 1:len(seq) + 1]
    sgn = r - mu_res >= 0                                                           # 640-bit residue signs
    b640 = (sgn.view(-1, 80, 8).to(torch.uint8) * SH8).sum(-1, dtype=torch.uint8).cpu().numpy()
    Q256 = ((sgn.half() * 2 - 1).float() - gmu) @ gP > 0                           # 256-bit PCA residue code
    return b640, (Q256.half() * 2 - 1)


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


def yours(seq, exclude_ids, T, tag):
    """Combined settings. Returns MSA rows (query length, no insertions) and accepted ids."""
    t = time.perf_counter()
    b640, Q = embed(seq)
    torch.cuda.synchronize()
    T[f"{tag}: embed query"] = time.perf_counter() - t
    t = time.perf_counter()
    sc = window_search(b640)
    ex = [id2idx[x] for x in exclude_ids if x in id2idx]
    sc[ex] = -1
    cand = np.argsort(-sc)[:200]
    cand = cand[sc[cand] > 0]
    T[f"{tag}: window index search"] = time.perf_counter() - t
    t = time.perf_counter()
    cs_, exs = set(cand.tolist()), set(ex)
    null = np.array([p for p in null_pool if p not in exs and p not in cs_])[:200]
    ns, _ = ea.sw_many(Q, [ea.pm1(np.array(R256[off[p]:off[p + 1]])) for p in null], BIAS, GAP)
    A = np.c_[np.ones(len(null)), np.log(lens[null])]
    coef, *_ = np.linalg.lstsq(A, ns, rcond=None)
    sd = np.std(ns - A @ coef)
    zf = lambda s, ps: (s - np.c_[np.ones(len(ps)), np.log(lens[ps])] @ coef) / sd
    q90, q99 = np.quantile(zf(ns, null), [0.9, 0.99])
    beta = (q99 - q90) / (np.log(-np.log(0.9)) - np.log(-np.log(0.99)))
    thr = q90 + beta * (-np.log(-np.log(1 - 1 / 2000)) + np.log(-np.log(0.9)))
    torch.cuda.synchronize()
    T[f"{tag}: calibration (200 random)"] = time.perf_counter() - t
    t = time.perf_counter()
    Tc = [ea.pm1(np.array(R256[off[p]:off[p + 1]])) for p in cand]
    cs, _ = ea.sw_many(Q, Tc, BIAS, GAP)
    keep = np.flatnonzero(zf(cs, cand) > thr)
    torch.cuda.synchronize()
    T[f"{tag}: align 200 candidates"] = time.perf_counter() - t
    t = time.perf_counter()
    rows = []
    if len(keep):
        _, pr = ea.sw_many(Q, [Tc[k] for k in keep], BIAS, GAP, keep=True)
        for k, pairs in zip(keep, pr):
            r = ["-"] * len(seq)
            for i, j in pairs:
                r[i] = db_seqs[cand[k]][j]
            rows.append("".join(r))
    T[f"{tag}: build MSA rows"] = time.perf_counter() - t
    fd = os.open(R256.filename, os.O_RDONLY)          # forget the candidates' cached codes, so every candidate
    for p in cand:                                    # read is a fresh read (as for a new protein in real use)
        os.posix_fadvise(fd, R256.offset + int(off[p]) * 32, int(lens[p]) * 32, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    return rows, [db_ids[cand[k]] for k in keep]


# ================================================================ native MSA builders
def sto_to_a3m(sto, q, seq):
    """AlphaFold-style: model (RF 'x') columns = query positions; insert-column residues kept lowercase."""
    seqs, rf = defaultdict(str), ""
    for line in open(sto):
        if line.startswith("#=GC RF"):
            rf += line.split()[-1]
        elif line.strip() and not line.startswith(("#", "//")):
            n, s = line.split()
            seqs[n] += s
    out = [f">{q}\n{seq}\n"]
    for n, s in seqs.items():
        r = "".join(c.upper() if m == "x" else (c.lower() if c not in "-." else "")
                    for c, m in zip(s, rf)).replace(".", "-")
        if sum(ch.isupper() or ch == "-" for ch in r) == len(seq) and n.split("/")[0] != q:
            out.append(f">{n}\n{r}\n")
    return "".join(out), len(out) - 1


def jackhmmer(q, seq, T, W):
    fa, sto = W / f"{q}.fa", W / f"{q}.jk.sto"
    fa.write_text(f">{q}\n{seq}\n")
    t = time.perf_counter()
    sh("jackhmmer", "-N", 1, "--F1", 0.0005, "--F2", 0.00005, "--F3", 0.0000005, "--incE", 0.0001, "-E", 0.0001,
       "--cpu", 6, "-o", "/dev/null", "-A", sto, fa, DB_FASTA)
    T["jackhmmer: search + align"] = time.perf_counter() - t
    t = time.perf_counter()
    a3m, n = (sto_to_a3m(sto, q, seq) if sto.exists() and sto.stat().st_size else (f">{q}\n{seq}\n", 0))
    T["jackhmmer: build MSA"] = time.perf_counter() - t
    return a3m, n


def mmseqs(q, seq, T, W):
    d = W / f"mm_{q}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    (d / "q.fa").write_text(f">{q}\n{seq}\n")
    db = DATA / "mmseqs_work" / "db"
    t = time.perf_counter()
    sh("mmseqs", "createdb", d / "q.fa", d / "q")
    sh("mmseqs", "search", d / "q", db, d / "res", d / "tmp", "--num-iterations", 3, "-s", 8, "-e", 0.1,
       "--max-seqs", 10000, "-a", "--threads", 6)
    T["mmseqs2: search + align (3 rounds)"] = time.perf_counter() - t
    t = time.perf_counter()
    sh("mmseqs", "filterresult", d / "q", db, d / "res", d / "res_f", "--qid", 0, "--qsc", 0.8, "--diff", 0,
       "--max-seq-id", 1.0, "--filter-min-enable", 100, "--threads", 6)
    sh("mmseqs", "result2msa", d / "q", db, d / "res_f", d / "msa", "--msa-format-mode", 6, "--filter-msa", 1,
       "--filter-min-enable", 1000, "--diff", 3000, "--qid", "0.0,0.2,0.4,0.6,0.8,1.0", "--qsc", 0,
       "--max-seq-id", 0.95, "--threads", 6)
    raw = (d / "msa").read_bytes().replace(b"\x00", b"").decode()
    T["mmseqs2: build MSA (ColabFold filters)"] = time.perf_counter() - t
    lines = raw.strip().splitlines()
    heads = [l[1:].split()[0] for l in lines if l.startswith(">")]
    a3m = f">{q}\n{seq}\n" + "\n".join(lines[2:]) + ("\n" if len(lines) > 2 else "")
    return a3m, heads[1:]


# ================================================================ run, one protein at a time
WK = E / "work"
shutil.rmtree(WK, ignore_errors=True)
WK.mkdir()
for p in PIPES:
    (E / "a3m" / p).mkdir(parents=True, exist_ok=True)
times, depth = {}, {}
import os                                                                          # noqa: E402
N_RUN = int(os.environ.get("N_RUN", 21))
for n, (q, seq) in enumerate(list(zip(qids, qseqs))[:N_RUN]):
    T = {}
    a_jk, n_jk = jackhmmer(q, seq, T, WK)
    a_mm, mm_ids = mmseqs(q, seq, T, WK)
    rows_y, acc_y = yours(seq, [], T, "yours")
    Th = {}
    rows_h, acc_h = yours(seq, mm_ids, Th, "hybrid-part")
    a_y = f">{q}\n{seq}\n" + "".join(f">{a}\n{r}\n" for a, r in zip(acc_y, rows_y))
    a_h = a_mm + "".join(f">{a}\n{r}\n" for a, r in zip(acc_h, rows_h))
    if n == 0:
        log(f"warm-up protein {q} done (not counted)")
        continue
    T.update(Th)
    times[q] = T
    depth[q] = {"jackhmmer": n_jk, "mmseqs2": len(mm_ids), "yours": len(rows_y), "hybrid": len(mm_ids) + len(rows_h)}
    for p, a in zip(PIPES, (a_jk, a_mm, a_y, a_h)):
        (E / "a3m" / p / f"{q}.a3m").write_text(a)
    log(f"  [{n}/20] {q} L={len(seq)}  rows: " + ", ".join(f"{p} {depth[q][p]}" for p in PIPES))
shutil.rmtree(WK, ignore_errors=True)
SETUP.update({"mmseqs2_index_s": 21.1, "yours_db_embedding_h": 1.7, "note": "index/embedding times from earlier runs"})
(E / "times.json").write_text(json.dumps({"per_protein": times, "depth": depth, "setup": SETUP}, indent=1))
log("e2e search + MSA done")
