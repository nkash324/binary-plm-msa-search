"""Quick full-length test: MMseqs2 vs Atlas (20 shards), 8 proteins.
For each SCOPe domain: find its full-length parent protein (PDB -> UniProt via EBI SIFTS), search with the
FULL-LENGTH sequence (max 1022 residues, a window centred on the domain if longer), then crop the MSA back to the
domain (column mapping by local alignment of domain vs parent) and fold the domain as before.
Same MMseqs2 settings (ColabFold-style) and same Atlas pipeline (top 100 -> 150M embedding alignment -> chance cutoff).
Run in the 'fold' env.  Writes e2e/a3m/{mm_full, atlas_full} and e2e/full_quick.json."""
import json
import shutil
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from Bio import Align
from Bio.Align import substitution_matrices
from transformers import AutoTokenizer, EsmModel

import emb_align as ea
from common import DATA, DB_FASTA, RESULTS, read_fasta

E, A = DATA / "e2e", Path.home() / "folding-data" / "atlas"
QS = ["d1cr5a1", "d2dzqa1", "d1xg0a_", "d1odha_", "d1pbya4", "d2b8ma1", "d5muaa1", "d3reta_"]
LOG = open(RESULTS / "full_quick.log", "a")
W = E / "work_full"
shutil.rmtree(W, ignore_errors=True)
W.mkdir()


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


def get(url):
    for k in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode()
        except Exception:                                   # noqa: BLE001
            time.sleep(2 * (k + 1))
    return None


qids, qseqs = read_fasta(E / "queries.fasta")
QSEQ = dict(zip(qids, qseqs))

# ---------------------------------------------------------------- 1. full-length parents + domain -> parent mapping
al = Align.PairwiseAligner(mode="local", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                           open_gap_score=-10, extend_gap_score=-0.5)
P = {}
for q in QS:
    pdb, chain = q[1:5], q[5].upper()
    m = json.loads(get(f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb}"))[pdb]["UniProt"]
    best = None
    for acc, v in m.items():
        for x in v["mappings"]:
            if x["chain_id"] == chain:
                n = x["unp_end"] - x["unp_start"]
                if best is None or n > best[1]:
                    best = (acc, n)
    acc = best[0]
    fa = get(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta")
    parent = "".join(fa.splitlines()[1:])
    dom = QSEQ[q]
    a = al.align(dom, parent)[0]
    mp = {}
    for (d0, d1), (p0, p1) in zip(*a.aligned):
        for k in range(d1 - d0):
            mp[d0 + k] = p0 + k
    lo, hi = min(mp.values()), max(mp.values())
    s = 0 if len(parent) <= 1022 else int(np.clip((lo + hi) // 2 - 511, 0, len(parent) - 1022))
    win = parent[s:s + 1022]
    P[q] = {"acc": acc, "parent_len": len(parent), "win_start": int(s), "win": win,
            "map": {i: j - s for i, j in mp.items() if 0 <= j - s < len(win)},
            "ident": float(np.mean([dom[i] == parent[j] for i, j in mp.items()])), "dom_span": [int(lo), int(hi)]}
    log(f"{q}: parent {acc} ({len(parent)} aa), domain at {lo + 1}-{hi + 1}, identity {P[q]['ident']:.2f}, "
        f"query window {s + 1}-{s + len(win)}")


def crop(rows, q):
    """rows: aligned strings in query-window coordinates (uppercase/'-' = match columns, lowercase = insertions).
    Returns rows in domain coordinates; drops rows with no residue in the domain."""
    mp, L = P[q]["map"], len(QSEQ[q])
    out = []
    for name, r in rows:
        cols = [c for c in r if not c.islower()]
        d = "".join(cols[mp[i]] if i in mp and mp[i] < len(cols) else "-" for i in range(L))
        if d.strip("-"):
            out.append((name, d))
    return out


def write(dirname, q, rows):
    (E / "a3m" / dirname).mkdir(parents=True, exist_ok=True)
    (E / "a3m" / dirname / f"{q}.a3m").write_text(f">{q}\n{QSEQ[q]}\n" + "".join(f">{n}\n{r}\n" for n, r in rows))


def sh(*a):
    subprocess.run([str(x) for x in a], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------- 2. MMseqs2 with the full-length query (same settings)
INFO = {}
for q in QS:
    d = W / f"mm_{q}"
    d.mkdir()
    (d / "q.fa").write_text(f">{q}\n{P[q]['win']}\n")
    db = DATA / "mmseqs_work" / "db"
    t = time.perf_counter()
    sh("mmseqs", "createdb", d / "q.fa", d / "q")
    sh("mmseqs", "search", d / "q", db, d / "res", d / "tmp", "--num-iterations", 3, "-s", 8, "-e", 0.1,
       "--max-seqs", 10000, "-a", "--threads", 6)
    sh("mmseqs", "filterresult", d / "q", db, d / "res", d / "res_f", "--qid", 0, "--qsc", 0.8, "--diff", 0,
       "--max-seq-id", 1.0, "--filter-min-enable", 100, "--threads", 6)
    sh("mmseqs", "result2msa", d / "q", db, d / "res_f", d / "msa", "--msa-format-mode", 6, "--filter-msa", 1,
       "--filter-min-enable", 1000, "--diff", 3000, "--qid", "0.0,0.2,0.4,0.6,0.8,1.0", "--qsc", 0,
       "--max-seq-id", 0.95, "--threads", 6)
    lines = (d / "msa").read_bytes().replace(b"\x00", b"").decode().strip().splitlines()
    rows = [(lines[k][1:].split()[0], lines[k + 1]) for k in range(2, len(lines) - 1, 2)]
    c = crop(rows, q)
    write("mm_full", q, c)
    INFO[q] = {"mm_rows_full": len(rows), "mm_rows_domain": len(c), "mm_s": time.perf_counter() - t}
    log(f"  MMseqs2 {q}: {len(rows)} rows for the full-length query -> {len(c)} cover the domain "
        f"({INFO[q]['mm_s']:.1f}s)")

# ---------------------------------------------------------------- 3. Atlas 20 shards with the full-length query
names = [u.rsplit("/", 1)[1][:-4] for u in (RESULTS / "atlas20_shards.txt").read_text().split()]
codes = np.ascontiguousarray(np.concatenate([np.load(A / "idx20" / f"{n}.codes.npy") for n in names])).view(np.uint64)
ids = sum(((A / "idx20" / f"{n}.ids.txt").read_text().split() for n in names), [])
apca = np.load(A / "pca.npz")
amu, aP = apca["mu"], apca["P"]
tok3 = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
m3 = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D", add_pooling_layer=False, torch_dtype=torch.float16,
                              device_map="cuda").eval()
cand = {}
with torch.no_grad():
    for q in QS:
        s = P[q]["win"]
        e = m3(**tok3([s], return_tensors="pt").to("cuda")).last_hidden_state[0, 1:len(s) + 1].float().mean(0).cpu().numpy()
        qc = np.ascontiguousarray(np.packbits(((e - amu) @ aP > 0)[None], axis=1)).view(np.uint64)
        ham = np.bitwise_count(codes ^ qc).sum(1)
        top = np.argpartition(ham, 100)[:100]
        cand[q] = top[np.lexsort((top, ham[top]))]
del m3, codes
torch.cuda.empty_cache()

cache_f = A / "seqs.json"
cache = json.loads(cache_f.read_text())


def fetch(i):
    if ids[i] in cache:
        return ids[i], cache[ids[i]]
    s = get(f"https://api.esmatlas.com/fetchSequence/{ids[i]}")
    return ids[i], json.loads(s)["sequence"] if s else None


with ThreadPoolExecutor(8) as ex:
    for k, s in ex.map(fetch, sorted({int(i) for c in cand.values() for i in c})):
        if s:
            cache[k] = s
cache_f.write_text(json.dumps(cache))

tok = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
m1 = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D", add_pooling_layer=False,
                              attn_implementation="sdpa").eval().cuda().half()
mu_res = torch.from_numpy(np.load(DATA / "mu.npz")["res"]).cuda()
gp = np.load(DATA / "pca_res.npz")
gmu, gP = torch.from_numpy(gp["mu"]).cuda(), torch.from_numpy(gp["P"][:, :256].copy()).cuda()
PRM = json.loads((DATA / "emb_align_params_256.json").read_text())
BIAS, GAP = PRM["bias"], PRM["gap"]


@torch.no_grad()
def code(s):
    s = s[:1022]
    r = m1(**tok([s], return_tensors="pt").to("cuda")).last_hidden_state[0, 1:len(s) + 1].float()
    return ((((r - mu_res >= 0).half() * 2 - 1).float() - gmu) @ gP > 0).half() * 2 - 1


db_ids, _ = read_fasta(DB_FASTA)
off = np.load(DATA / "offsets.npy")
lens = np.diff(off)
R256 = np.load(DATA / "res_bits256.npy", mmap_mode="r")
null = np.random.default_rng(42).choice(len(db_ids), 2000, replace=False)[:200]
Tnull = [ea.pm1(np.array(R256[off[p]:off[p + 1]])) for p in null]
for q in QS:
    s = P[q]["win"]
    Q = code(s)
    cs = [ids[i] for i in cand[q] if ids[i] in cache]
    Tc = [code(cache[c]) for c in cs]
    ns, _ = ea.sw_many(Q, Tnull, BIAS, GAP)
    Al = np.c_[np.ones(len(null)), np.log(lens[null])]
    coef, *_ = np.linalg.lstsq(Al, ns, rcond=None)
    sd = np.std(ns - Al @ coef)
    zf = lambda sc, L: (sc - np.c_[np.ones(len(L)), np.log(L)] @ coef) / sd
    q90, q99 = np.quantile(zf(ns, lens[null]), [0.9, 0.99])
    beta = (q99 - q90) / (np.log(-np.log(0.9)) - np.log(-np.log(0.99)))
    thr = q90 + beta * (-np.log(-np.log(1 - 1 / 2000)) + np.log(-np.log(0.9)))
    sc, _ = ea.sw_many(Q, Tc, BIAS, GAP)
    keep = np.flatnonzero(zf(sc, np.array([len(cache[c][:1022]) for c in cs])) > thr)
    rows = []
    if len(keep):
        _, pr = ea.sw_many(Q, [Tc[k] for k in keep], BIAS, GAP, keep=True)
        for k, pairs in zip(keep, pr):
            r = ["-"] * len(s)
            for i, j in pairs:
                r[i] = cache[cs[k]][j]
            rows.append((cs[k], "".join(r)))
    c = crop(rows, q)
    write("atlas_full", q, c)
    INFO[q].update({"atlas_rows_full": len(rows), "atlas_rows_domain": len(c)})
    log(f"  Atlas {q}: {len(rows)} accepted for the full-length query -> {len(c)} cover the domain")
(E / "full_quick.json").write_text(json.dumps({"parents": {q: {k: v for k, v in P[q].items() if k not in ("win", "map")}
                                                               for q in QS}, "rows": INFO}, indent=1))
log("MSAS DONE")
