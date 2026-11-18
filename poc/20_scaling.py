"""How do the three methods scale with database size?  Nested random subsets of Swiss-Prot
(50k, 100k, 200k, 400k, all 576k proteins), the same 50 test proteins at every size.

  BinPLM     : window index restricted to the subset (no new embedding) -> time per query, and recall of the
              true relatives in the top 200 / 500 / 2000, and how many candidates are needed for 90% recall
  jackhmmer : AlphaFold settings, 6 threads, one query at a time
  mmseqs2   : ColabFold search settings, 6 threads, index built per subset, 50 queries as one batch
True relatives = the exhaustive phmmer --max hits over all of Swiss-Prot that are inside the subset
(sequence-detectable relatives). Writes results/scaling.{md,json}.  Run in the 'fold' env.
"""
import json
import shutil
import subprocess
import time

import numpy as np

from common import DATA, DB_FASTA, Q_FASTA, RESULTS, read_fasta

SIZES = [50_000, 100_000, 200_000, 400_000, 576_073]
KS = [200, 500, 2000]
S = DATA / "scaling"
LOG = open(RESULTS / "scaling.log", "a")


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


def sh(*a):
    subprocess.run([str(x) for x in a], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


db_ids, db_seqs = read_fasta(DB_FASTA)
NP = len(db_ids)
id2idx = {d: i for i, d in enumerate(db_ids)}
q_all, qs_all = read_fasta(Q_FASTA)
QSEL = list(range(0, 100, 2))
qids = [q_all[i] for i in QSEL]
q_off = np.load(DATA / "q_offsets.npy")
q_res = np.load(DATA / "q_res_bits.npy")
truth_all = json.loads((DATA / "hits" / "phmmer_max.json").read_text())["hits"]
truth_all = {q: {id2idx[t] for t in truth_all[q] if t in id2idx} for q in qids}
order = np.random.default_rng(0).permutation(NP)             # nested subsets: first N of one shuffle

z = np.load(DATA / "win64_256.npz")
codes_all = np.ascontiguousarray(z["codes"]).view(np.uint64)
owner_all, MU, PM = z["owner"], z["mu"], z["P"]
W, B, TOPW = 64, 256, 2000


def qcodes(qi):
    X = np.unpackbits(q_res[q_off[qi]:q_off[qi + 1]], axis=1).astype(np.float32) * 2 - 1
    L = len(X)
    starts = sorted(set(list(range(0, max(1, L - W + 1), W // 4)) + [max(0, L - W)]))
    return np.ascontiguousarray(np.packbits((np.stack([X[a:a + W].mean(0) for a in starts]) - MU) @ PM > 0,
                                            axis=1)).view(np.uint64)


def window_search(qc, codes, owner):
    score = np.zeros(NP, np.float32)
    for lo in range(0, len(codes), 500_000):
        c = codes[lo:lo + 500_000]
        h = np.zeros((len(qc), len(c)), np.uint16)
        for w in range(c.shape[1]):
            h += np.bitwise_count(c[None, :, w] ^ qc[:, None, w])
        k = min(TOPW, len(c) - 1)
        i = np.argpartition(h, k, axis=1)[:, :k]
        zz = (B / 2 - np.take_along_axis(h, i, 1).astype(np.float32)) / (np.sqrt(B) / 2)
        for j in range(len(qc)):
            o = np.argsort(-zz[j])
            pp, first = np.unique(owner[lo + i[j][o]], return_index=True)
            score[pp] += np.maximum(zz[j][o][first], 0)
    return score


QC = {q: qcodes(qi) for q, qi in zip(qids, QSEL)}
rows = []
S.mkdir(exist_ok=True)
for N in SIZES:
    sub = np.sort(order[:N])
    insub = np.zeros(NP, bool)
    insub[sub] = True
    truth = {q: truth_all[q] & set(sub.tolist()) for q in qids}
    tq = [q for q in qids if truth[q]]
    r = {"proteins": N, "residues": int(sum(len(db_seqs[i]) for i in sub)),
         "true_relatives_per_query": float(np.mean([len(truth[q]) for q in qids]))}

    # ---- BinPLM: window index restricted to the subset
    m = insub[owner_all]
    codes, owner = codes_all[m], owner_all[m]
    r["yours_index_MB"] = codes.nbytes / 1e6
    ts, rec, k90 = [], {k: [] for k in KS}, []
    for q in qids:
        t = time.perf_counter()
        sc = window_search(QC[q], codes, owner)
        nz = np.flatnonzero(sc > 0)
        rank = nz[np.argsort(-sc[nz])]
        ts.append(time.perf_counter() - t)
        if truth[q]:
            pos = np.flatnonzero(np.isin(rank, list(truth[q])))          # ranks of true relatives found at all
            for k in KS:
                rec[k].append((pos < k).sum() / len(truth[q]))
            need = int(np.ceil(0.9 * len(truth[q])))
            k90.append(pos[need - 1] + 1 if len(pos) >= need else np.nan)
    r["yours_search_s"] = float(np.mean(ts))
    r.update({f"yours_recall@{k}": float(np.mean(rec[k])) for k in KS})
    r["yours_K_for_90pct_median"] = float(np.nanmedian(k90)) if np.isfinite(k90).any() else None
    r["yours_K90_reached_share"] = float(np.mean(np.isfinite(k90)))

    # ---- sequences of the subset (for JackHMMER + MMseqs2)
    d = S / f"n{N}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    fa = d / "db.fa"
    with open(fa, "w") as f:
        for i in sub:
            f.write(f">{db_ids[i]}\n{db_seqs[i]}\n")

    # ---- JackHMMER, one query at a time
    ts, rec = [], []
    for q, qi in zip(qids, QSEL):
        qf, tb = d / f"{q}.fa", d / f"{q}.tbl"
        qf.write_text(f">{q}\n{qs_all[qi]}\n")
        t = time.perf_counter()
        sh("jackhmmer", "-N", 1, "--F1", 0.0005, "--F2", 0.00005, "--F3", 0.0000005, "--incE", 0.0001, "-E", 0.0001,
           "--cpu", 6, "-o", "/dev/null", "--tblout", tb, qf, fa)
        ts.append(time.perf_counter() - t)
        if truth[q]:
            hits = {id2idx[l.split()[0]] for l in open(tb) if not l.startswith("#")}
            rec.append(len(hits & truth[q]) / len(truth[q]))
    r["jackhmmer_s"], r["jackhmmer_recall"] = float(np.mean(ts)), float(np.mean(rec))

    # ---- MMseqs2: index per subset, 50 queries in one batch
    (d / "q.fa").write_text("".join(f">{q}\n{qs_all[qi]}\n" for q, qi in zip(qids, QSEL)))
    t = time.perf_counter()
    sh("mmseqs", "createdb", fa, d / "db")
    sh("mmseqs", "createindex", d / "db", d / "tmpi", "-s", 8, "--threads", 6)
    r["mmseqs_index_build_s"] = time.perf_counter() - t
    r["mmseqs_index_MB"] = sum(f.stat().st_size for f in d.glob("db.idx*")) / 1e6
    sh("mmseqs", "createdb", d / "q.fa", d / "q")
    t = time.perf_counter()
    sh("mmseqs", "search", d / "q", d / "db", d / "res", d / "tmp", "--num-iterations", 3, "-s", 8, "-e", 0.1,
       "--max-seqs", 10000, "--threads", 6)
    r["mmseqs_s_per_query_batch50"] = (time.perf_counter() - t) / len(qids)
    sh("mmseqs", "convertalis", d / "q", d / "db", d / "res", d / "res.m8")
    hits = {q: set() for q in qids}
    for l in open(d / "res.m8"):
        c = l.split("\t")
        hits[c[0]].add(id2idx[c[1]])
    r["mmseqs_recall"] = float(np.mean([len(hits[q] & truth[q]) / len(truth[q]) for q in tq]))
    shutil.rmtree(d, ignore_errors=True)
    rows.append(r)
    log(f"N={N:>7,}: BinPLM {r['yours_search_s']:.2f}s R@200={r['yours_recall@200']:.3f} K90={r['yours_K_for_90pct_median']} | "
        f"jackhmmer {r['jackhmmer_s']:.2f}s R={r['jackhmmer_recall']:.3f} | mmseqs {r['mmseqs_s_per_query_batch50']:.2f}s "
        f"R={r['mmseqs_recall']:.3f} idx {r['mmseqs_index_MB']:.0f}MB")

# ---------------------------------------------------------------- growth rates + rough projection to 100x
lnN = np.log([r["proteins"] for r in rows])
slope = lambda key: float(np.polyfit(lnN, np.log([r[key] for r in rows]), 1)[0])
proj = lambda key: float(np.exp(np.polyval(np.polyfit(lnN, np.log([r[key] for r in rows]), 1), np.log(57_607_300))))
keys = ["yours_search_s", "jackhmmer_s", "mmseqs_s_per_query_batch50", "yours_index_MB", "mmseqs_index_MB"]
grow = {k: {"exponent": slope(k), "at_100x": proj(k)} for k in keys}
k90 = [r["yours_K_for_90pct_median"] for r in rows]
if all(v for v in k90):
    grow["yours_K_for_90pct_median"] = {"exponent": slope("yours_K_for_90pct_median"), "at_100x": proj("yours_K_for_90pct_median")}
(RESULTS / "scaling.json").write_text(json.dumps({"rows": rows, "growth": grow}, indent=1))

L = ["# Scaling with database size (nested random subsets of Swiss-Prot, same 50 test proteins)", "",
     "True relatives = exhaustive phmmer --max hits inside each subset (sequence-detectable relatives).", "",
     "| proteins | true rel./query | **BinPLM**: search s | recall@200 | @500 | @2000 | candidates for 90% (median) | index MB "
     "| **JackHMMER** s | recall | **MMseqs2** s/query (batch) | recall | index MB | index build s |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['proteins']:,} | {r['true_relatives_per_query']:.0f} | {r['yours_search_s']:.2f} | {r['yours_recall@200']:.3f} | "
             f"{r['yours_recall@500']:.3f} | {r['yours_recall@2000']:.3f} | {r['yours_K_for_90pct_median']} | {r['yours_index_MB']:.0f} | "
             f"{r['jackhmmer_s']:.2f} | {r['jackhmmer_recall']:.3f} | {r['mmseqs_s_per_query_batch50']:.2f} | {r['mmseqs_recall']:.3f} | "
             f"{r['mmseqs_index_MB']:.0f} | {r['mmseqs_index_build_s']:.0f} |")
L += ["", "Growth exponent: 1.0 = doubles when the database doubles; 0.5 = grows with the square root. "
      "Projection to 100x Swiss-Prot (~58M proteins) is a straight-line extrapolation on a log-log plot over an 11x range: rough.", "",
      "| quantity | exponent | projected at 100x |", "|---|---|---|"]
for k, v in grow.items():
    L.append(f"| {k} | {v['exponent']:.2f} | {v['at_100x']:.3g} |")
(RESULTS / "scaling.md").write_text("\n".join(L) + "\n")
log("\n".join(L))
