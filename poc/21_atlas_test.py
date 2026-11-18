"""Small ESM Atlas test: does a binary per-protein index built from Meta's public ESM-2 3B embeddings find
structural relatives of our hard proteins among ~430k metagenomic proteins?

 1. check we reproduce the Atlas embedding (mean of ESM-2 3B last layer over residues) for a few shard proteins
 2. shard vectors -> PCA -> 256 bits (32 B/protein); also keep full float vectors as an upper-bound comparison
 3. 19 fresh hard proteins (end-to-end test set): embed with ESM-2 3B -> top 10 by Hamming (and by float cosine)
 4. truth = STRUCTURE: fetch each hit's ESMFold structure from the Atlas API, TM-score vs the protein's real
    (SCOPe) structure; TM > 0.5 = same fold.  Control: 5 random shard proteins per query.
 5. could sequence search have found them?  phmmer --max (most sensitive) of each query vs its hits, E <= 1e-3
    at the shard's size.
Writes results/atlas_test.md.  Run in the 'fold' env.
"""
import os
import base64
import json
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

from common import DATA, RESULTS, read_fasta

A = Path.home() / "folding-data" / "atlas"
PDB = A / "pdb"
PDB.mkdir(parents=True, exist_ok=True)
E = DATA / "e2e"
USALIGN = os.path.expanduser("~") + "/miniforge3/envs/cf/bin/USalign"
LOG = open(RESULTS / "atlas_test.log", "a")
rng = np.random.default_rng(0)


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


def api(path, binary=False, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(f"https://api.esmatlas.com/{path}", timeout=60) as r:
                d = r.read()
            time.sleep(0.2)                                   # be polite to the public API
            return d if binary else d.decode()
        except Exception as e:                                # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


# ---------------------------------------------------------------- 1. shard -> one array
if not (A / "shard_vecs.npy").exists():
    t0 = time.time()
    z = zipfile.ZipFile(A / "shard.npz")
    names = z.namelist()
    V = np.zeros((len(names), 2560), np.float16)
    ids = []
    for i, n in enumerate(names):
        with z.open(n) as f:
            V[i] = np.lib.format.read_array(f)
        ids.append(Path(n).stem)
    np.save(A / "shard_vecs.npy", V)
    (A / "shard_ids.txt").write_text("\n".join(ids))
    log(f"shard unpacked: {len(ids):,} proteins in {time.time() - t0:.0f}s")
V = np.load(A / "shard_vecs.npy")
ids = (A / "shard_ids.txt").read_text().split()
N = len(ids)

# ---------------------------------------------------------------- ESM-2 3B on the GPU
t0 = time.time()
tok = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D", add_pooling_layer=False, torch_dtype=torch.float16,
                                 device_map="cuda").eval()
log(f"ESM-2 3B loaded in {time.time() - t0:.0f}s, GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB")


@torch.no_grad()
def embed(seq):
    enc = tok([seq], return_tensors="pt").to("cuda")
    h = model(**enc).last_hidden_state[0, 1:len(seq) + 1].float()
    return h.mean(0).cpu().numpy()


# ---------------------------------------------------------------- 1b. do we reproduce the Atlas embedding?
chk = []
for i in rng.choice(N, 3, replace=False):
    seq = json.loads(api(f"fetchSequence/{ids[i]}"))["sequence"]
    if len(seq) > 1000:
        continue
    mine = embed(seq)
    ref = V[i].astype(np.float32)
    chk.append(float(mine @ ref / np.linalg.norm(mine) / np.linalg.norm(ref)))
log(f"reproduce Atlas embedding: cosine {['%.4f' % c for c in chk]} (1.0 = identical recipe)")

# ---------------------------------------------------------------- 2. binary index
samp = V[np.sort(rng.choice(N, 50_000, replace=False))].astype(np.float32)
mu = samp.mean(0)
w, v = np.linalg.eigh(np.cov((samp - mu).T))
P = v[:, np.argsort(-w)[:256]].astype(np.float32)
del samp
CH = range(0, N, 50_000)                                   # chunked: never hold a float32 copy of all vectors
codes = np.ascontiguousarray(np.concatenate(
    [np.packbits((V[a:a + 50_000].astype(np.float32) - mu) @ P > 0, axis=1) for a in CH])).view(np.uint64)
norms = np.concatenate([np.linalg.norm(V[a:a + 50_000].astype(np.float32), axis=1) for a in CH])


def cosine_all(e):
    return np.concatenate([V[a:a + 50_000].astype(np.float32) @ e for a in CH]) / norms / np.linalg.norm(e)
log(f"binary index: {codes.nbytes / 1e6:.1f} MB for {N:,} proteins (float vectors: {V.nbytes / 1e9:.1f} GB)")

# ---------------------------------------------------------------- 3. queries
qids, qseqs = read_fasta(E / "queries.fasta")
qids, qseqs = qids[1:], qseqs[1:]                    # first one was the warm-up protein
TOP, NRAND = 10, 5
hits, t_emb, t_srch = {}, [], []
for q, s in zip(qids, qseqs):
    t = time.perf_counter()
    e = embed(s)
    torch.cuda.synchronize()
    t_emb.append(time.perf_counter() - t)
    t = time.perf_counter()
    qc = np.ascontiguousarray(np.packbits(((e - mu) @ P > 0)[None], axis=1)).view(np.uint64)
    ham = np.bitwise_count(codes ^ qc).sum(1)
    hb = np.argsort(ham, kind="stable")[:TOP]
    t_srch.append(time.perf_counter() - t)
    hf = np.argsort(-cosine_all(e))[:TOP]
    hits[q] = {"binary": hb.tolist(), "float": hf.tolist(), "random": rng.choice(N, NRAND, replace=False).tolist()}
log(f"query embedding (3B) {np.median(t_emb):.2f} s, binary search over {N:,} proteins {np.median(t_srch) * 1e3:.0f} ms (median)")

# ---------------------------------------------------------------- 4. structure truth + 5. sequence check
three = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
         "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
         "TYR": "Y", "VAL": "V"}


def structure(i):
    f = PDB / f"{ids[i]}.pdb"
    if not f.exists():
        f.write_text(api(f"fetchPredictedStructure/{ids[i]}"))
    return f


def seq_of(f):
    return "".join(three.get(l[17:20], "X") for l in open(f) if l.startswith("ATOM") and l[12:16].strip() == "CA")


def tm(f, ref):
    o = subprocess.run([USALIGN, str(f), str(ref), "-outfmt", "2"], capture_output=True, text=True).stdout
    return float([l for l in o.splitlines() if l and not l.startswith("#")][0].split("\t")[3])  # normalised by query


res = {}
for n, (q, s) in enumerate(zip(qids, qseqs)):
    ref = E / "ref" / f"{q}.ent"
    allh = sorted(set(hits[q]["binary"] + hits[q]["float"] + hits[q]["random"]))
    tms = {i: tm(structure(i), ref) for i in allh}
    fa, qf, tb = A / "cand.fa", A / "q.fa", A / "q.tbl"
    fa.write_text("".join(f">{i}\n{seq_of(structure(i))}\n" for i in allh))
    qf.write_text(f">{q}\n{s}\n")
    subprocess.run(["phmmer", "--max", "-E", "1e-3", "-Z", str(N), "--cpu", "4", "-o", "/dev/null", "--tblout", str(tb),
                    str(qf), str(fa)], check=True)
    seqhit = {int(l.split()[0]) for l in open(tb) if not l.startswith("#")}
    res[q] = {k: [{"id": ids[i], "tm": tms[i], "seq_detectable": i in seqhit} for i in hits[q][k]]
              for k in ("binary", "float", "random")}
    b = res[q]["binary"]
    log(f"  [{n + 1}/{len(qids)}] {q}: binary top-{TOP} same-fold (TM>0.5): {sum(h['tm'] > 0.5 for h in b)}, "
        f"best TM {max(h['tm'] for h in b):.2f}; of those, sequence-detectable: "
        f"{sum(h['tm'] > 0.5 and h['seq_detectable'] for h in b)}")
(RESULTS / "atlas_test.json").write_text(json.dumps({"hits": res, "reproduce_cosine": chk,
                                                     "embed_s": t_emb, "search_s": t_srch}, indent=1))


# ---------------------------------------------------------------- report
def summ(kind):
    tms = np.array([[h["tm"] for h in res[q][kind]] for q in qids])
    same = tms > 0.5
    sd = np.array([[h["tm"] > 0.5 and h["seq_detectable"] for h in res[q][kind]] for q in qids])
    return {"top1_same_fold": f"{same[:, 0].sum()}/{len(qids)}", "proteins_with_any": f"{same.any(1).sum()}/{len(qids)}",
            "per_query": same.sum(1).mean(), "precision": same.mean(), "mean_tm": tms.mean(),
            "seq_det": sd.sum(), "same_total": same.sum()}


L = [f"# ESM Atlas small test: {len(qids)} fresh hard proteins vs {N:,} metagenomic proteins (one shard)", "",
     "Index: Meta's public ESM-2 3B per-protein embeddings. Truth = structure: each hit's ESMFold model vs the protein's",
     "real SCOPe structure (US-align, TM normalised by the query). TM > 0.5 = same fold.", "",
     f"Reproducing the Atlas embedding with our own ESM-2 3B run: cosine {', '.join('%.4f' % c for c in chk)}.",
     f"Binary index {codes.nbytes / 1e6:.1f} MB (32 B/protein). Query embedding {np.median(t_emb):.2f} s, "
     f"search {np.median(t_srch) * 1e3:.0f} ms per protein (median).", "",
     f"| hits | top-1 is same fold | proteins with >=1 same-fold hit in top {TOP} | same-fold hits per protein | "
     f"precision | mean TM | same-fold hits a sequence search (phmmer --max) could detect |", "|---|---|---|---|---|---|---|"]
for k, lab in (("binary", f"binary 256-bit, top {TOP}"), ("float", f"full float vectors, top {TOP}"),
               ("random", f"random shard proteins ({NRAND}/protein)")):
    r = summ(k)
    L.append(f"| {lab} | {r['top1_same_fold'] if k != 'random' else '-'} | {r['proteins_with_any']} | {r['per_query']:.1f} | "
             f"{100 * r['precision']:.0f}% | {r['mean_tm']:.2f} | {r['seq_det']} of {r['same_total']} |")
L += ["", "## Per protein (binary index, top 10)", "", "| protein | same-fold hits | best TM | sequence-detectable same-fold |",
      "|---|---|---|---|"]
for q in qids:
    b = res[q]["binary"]
    L.append(f"| {q} | {sum(h['tm'] > 0.5 for h in b)} | {max(h['tm'] for h in b):.2f} | "
             f"{sum(h['tm'] > 0.5 and h['seq_detectable'] for h in b)} |")
(RESULTS / "atlas_test.md").write_text("\n".join(L) + "\n")
log("\n".join(L))
