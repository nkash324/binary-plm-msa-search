"""Step 4: Swiss-Prot MSAs for the 20 e2e targets with three ways of finding candidates, one aligner, one cutoff.

Retrieval:
  exh640   : every Swiss-Prot protein (576k), 640-bit embedding SW (fused Triton kernel)
  twostage : codebook first pass over everything (TWOSTAGE_VARIANT, default from step 3), top 1% -> 640-bit SW
  window   : the existing BinPLM window index, top 200 (as 18_e2e.py) -> 640-bit SW
Aligner: 640-bit embedding SW, leak-free bias/gap (emb_align_params_sprot.json). Rows = matched query columns.
Acceptance, identical for all three: database-scale E-value E = 576,073 x p < E_CUT (default 0.1, i.e. the same
0.1 expected chance rows per target as the 2x2 cutoff), p from a Gumbel fitted to the tail (99th / 99.9th
percentile) of the query's z-scores against 20,000 reversed-sequence Swiss-Prot decoys (z = length-corrected).
Calibration check: half of the decoys are held out and counted at E < 1.
Also writes a depth-matched control: the 2x2 binplm_emb rows subsampled to each target's exh640 depth.
Outputs ~/folding-data/poc/msa_exh/<method>/<target>.a3m and results/exh_sprot.{md,json}.  Run in the 'fold' env."""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import emb_align as ea
import esw_triton as et
from common import DATA, DB_FASTA, RESULTS, read_fasta

torch.cuda.set_per_process_memory_fraction(0.6)
log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)
E2E, OUT = DATA / "e2e", DATA / "msa_exh"
METHODS = ["exh640", "twostage", "window"]
for m in METHODS + ["depth_ctrl"]:
    (OUT / m).mkdir(parents=True, exist_ok=True)
E_CUT = float(os.environ.get("E_CUT", 0.1))
VAR = os.environ.get("TWOSTAGE_VARIANT", "cb1024")
FRAC = float(os.environ.get("TWOSTAGE_FRAC", 0.05))
P = json.loads((DATA / "emb_align_params_sprot.json").read_text())
BIAS, GAP = P["bias"], P["gap"]
PAUSE = DATA / "PAUSE"
CBP = json.loads(Path("/mnt/c/Folding/phase1/results/codebook_twostage.json").read_text())[f"params_{VAR}"]
VB, VG = CBP["bias"], CBP["gap"]                                               # rescaled first-pass params (step 3)

db_ids, db_seqs = read_fasta(DB_FASTA)
off = np.load(DATA / "offsets.npy")
NP, lens = len(off) - 1, np.diff(off)
bits = np.load(DATA / "res_bits.npy", mmap_mode="r")
mu_res = torch.from_numpy(np.load(DATA / "mu.npz")["res"]).cuda()
from transformers import AutoTokenizer, EsmModel                                   # noqa: E402
tok = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
esm = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D", add_pooling_layer=False,
                               attn_implementation="sdpa").eval().cuda().half()
SH8 = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], device="cuda", dtype=torch.uint8)


@torch.no_grad()
def embed_bits(seqs):
    out = []
    for s in seqs:
        enc = tok([s], return_tensors="pt")
        r = esm(input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda()).last_hidden_state
        sg = r[0, 1:len(s) + 1].float() - mu_res >= 0
        out.append((sg.view(-1, 80, 8).to(torch.uint8) * SH8).sum(-1, dtype=torch.uint8).cpu().numpy())
    return out


# ---------------------------------------------------------------- 20k reversed-sequence decoys (built once)
DEC = DATA / "sprot_decoys.npz"
if not DEC.exists():
    t = time.time()
    rng = np.random.default_rng(21)
    pick = np.sort(rng.choice(np.flatnonzero((lens >= 30) & (lens <= 1022)), 20_000, replace=False))
    db = embed_bits([db_seqs[p][::-1] for p in pick])
    np.savez(DEC, bits=np.concatenate(db), off=np.r_[0, np.cumsum([len(x) for x in db])], src=pick)
    log(f"20,000 Swiss-Prot decoys embedded ({time.time() - t:.0f}s)")
z = np.load(DEC)
dbits, doff = z["bits"], z["off"]
dlen = np.diff(doff)
half = np.random.default_rng(0).permutation(len(dlen)) < len(dlen) // 2               # fit half / check half

# ---------------------------------------------------------------- retrieval helpers
CBK = np.load(DATA / "codebooks.npz")
TAB = {"cb20": CBK["T20"][None], "cb256": CBK["T256"][None], "cb1024": CBK["T1024"][None],
       "pq4": np.stack([CBK[f"PQT{m}"] for m in range(4)])}[VAR]
CC = {"cb20": CBK.get("C20"), "cb256": CBK.get("C256"), "cb1024": CBK.get("C1024")}
sym = np.load(DATA / f"res_{VAR}.npy", mmap_mode="r")


def query_sym(b640):
    Y = ea.pm1(b640)
    if VAR == "pq4":
        return np.stack([(Y[:, m * 160:(m + 1) * 160] @ torch.from_numpy(CBK[f"PQ{m}"]).cuda().half().T).argmax(1)
                         .cpu().numpy() for m in range(4)], 1)
    return (Y @ torch.from_numpy(CC[VAR]).cuda().half().T).argmax(1).cpu().numpy()


wz = np.load(DATA / "win64_256.npz")
wcodes = np.ascontiguousarray(wz["codes"]).view(np.uint64)
owner, WMU, WPM = wz["owner"], wz["mu"], wz["P"]


def window_top(b640, top=200):
    X = np.unpackbits(b640, axis=1).astype(np.float32) * 2 - 1
    L, Wn, B, TOPW = len(X), 64, 256, 2000
    starts = sorted(set(list(range(0, max(1, L - Wn + 1), Wn // 4)) + [max(0, L - Wn)]))
    qc = np.ascontiguousarray(np.packbits((np.stack([X[a:a + Wn].mean(0) for a in starts]) - WMU) @ WPM > 0,
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
    c = np.argsort(-score)[:top]
    return np.sort(c[score[c] > 0])


def neff(rows):
    if len(rows) < 2:
        return float(len(rows))
    A = np.array([list(r) for r in rows])
    X = np.stack([(A == c) for c in "ACDEFGHIKLMNPQRSTVWY"], -1).reshape(len(rows), -1).astype(np.float32)
    nong = (A != "-").sum(1).astype(np.float32)
    S = X @ X.T / np.maximum(np.minimum.outer(nong, nong), 1)
    return float((1 / (S >= 0.8).sum(1)).sum())


# ---------------------------------------------------------------- run
qids, qseqs = read_fasta(E2E / "queries.fasta")
QS = [(q, s) for q, s in zip(qids, qseqs) if (E2E / "a3m" / "mmseqs2" / f"{q}.a3m").exists()]
mm_hits = {q: [l[1:].split()[0] for l in (E2E / "a3m" / "mmseqs2" / f"{q}.a3m").read_text().splitlines()
               if l.startswith(">")][1:] for q, _ in QS}                         # MMseqs2's MSA rows (reporting only)
id2i = {d: i for i, d in enumerate(db_ids)}
Qb = embed_bits([s for _, s in QS])
et.search(Qb[:1], bits, off, BIAS, GAP, targets=np.arange(300))                # compile / warm-up
et.search_lut([query_sym(Qb[0])], sym, off, TAB, VB, VG, targets=np.arange(300))
A_db = np.c_[np.ones(NP), np.log(lens)]
R, T = {}, {m: [] for m in METHODS}
# exhaustive: all 20 queries in one streaming pass
torch.cuda.synchronize()
t = time.time()
EX = et.search(Qb, bits, off, BIAS, GAP, pause=PAUSE)
torch.cuda.synchronize()
t_exh = (time.time() - t) / len(QS)
log(f"exhaustive 640-bit SW over Swiss-Prot: {t_exh:.1f} s/query")
DS = et.search(Qb, dbits, doff, BIAS, GAP)                                     # decoys (calibration)
for n, ((q, seq), qb) in enumerate(zip(QS, Qb)):
    while PAUSE.exists():
        time.sleep(10)
    # calibration from the fit half of the decoys
    Ad = np.c_[np.ones(len(dlen)), np.log(dlen)]
    coef, *_ = np.linalg.lstsq(Ad[half], DS[n][half], rcond=None)
    sd = np.std(DS[n][half] - Ad[half] @ coef)
    zd = (DS[n] - Ad @ coef) / sd
    q99, q999 = np.quantile(zd[half], [0.99, 0.999])
    beta = (q999 - q99) / (np.log(-np.log(0.99)) - np.log(-np.log(0.999)))
    mu = q99 + beta * np.log(-np.log(0.99))
    Ef = lambda zz: NP * -np.expm1(-np.exp(-(zz - mu) / beta))
    zall = (EX[n] - A_db @ coef) / sd
    cands = {}
    t = time.time()
    F = et.search_lut([query_sym(qb)], sym, off, TAB, VB, VG)[0]
    zf = F - A_db @ np.linalg.lstsq(A_db, F, rcond=None)[0]
    cands["twostage"] = np.sort(np.argsort(-zf)[:int(FRAC * NP)])
    et.search([qb], bits, off, BIAS, GAP, targets=cands["twostage"])           # timed rescoring (same values)
    torch.cuda.synchronize()
    T["twostage"].append(time.time() - t)
    t = time.time()
    cands["window"] = window_top(qb)
    et.search([qb], bits, off, BIAS, GAP, targets=cands["window"])
    torch.cuda.synchronize()
    T["window"].append(time.time() - t)
    cands["exh640"] = np.arange(NP)
    T["exh640"].append(t_exh)
    R[q] = {"L": len(seq), "decoys_heldout_E<1": int((Ef(zd[~half]) < 1).sum()),
            "decoys_heldout_expected": float(NP and (~half).sum() / NP)}
    acc = {}
    for m in METHODS:
        c = cands[m]
        keep = c[Ef(zall[c]) < E_CUT]
        keep = keep[np.argsort(Ef(zall[keep]))]
        rows = []
        if len(keep):
            Qp = ea.pm1(qb)
            _, prs = ea.sw_many(Qp, [ea.pm1(np.asarray(bits[off[p]:off[p + 1]])) for p in keep], BIAS, GAP, keep=True)
            for p, pr in zip(keep, prs):
                r = ["-"] * len(seq)
                for i, j in pr:
                    r[i] = db_seqs[p][j]
                rows.append("".join(r))
        (OUT / m / f"{q}.a3m").write_text(f">{q}\n{seq}\n" + "".join(f">{db_ids[p]}\n{r}\n" for p, r in zip(keep, rows)))
        acc[m] = set(keep.tolist())
        R[q][m] = {"rows": len(rows), "neff80": neff([seq] + rows), "candidates": int(len(c))}
    win_c = set(cands["window"].tolist())
    R[q]["exh_not_in_window_candidates"] = len(acc["exh640"] - win_c)
    mmh = {id2i[a] for a in mm_hits[q] if a in id2i}
    R[q]["mmseqs_hits"] = len(mmh)
    R[q]["mmseqs_hits_accepted"] = {m: len(mmh & acc[m]) for m in METHODS}
    # depth-matched control: 2x2 binplm_emb rows subsampled to the exh640 depth
    ids_, rows_ = read_fasta(DATA / "msa_2x2" / "binplm_emb" / f"{q}.a3m")
    d = min(R[q]["exh640"]["rows"], len(rows_) - 1)
    pick = sorted(np.random.default_rng(0).choice(np.arange(1, len(rows_)), d, replace=False)) if d else []
    (OUT / "depth_ctrl" / f"{q}.a3m").write_text(f">{q}\n{seq}\n" + "".join(f">{ids_[i]}\n{rows_[i]}\n" for i in pick))
    R[q]["depth_ctrl"] = {"rows": d, "neff80": neff([seq] + [rows_[i] for i in pick])}
    log(f"[{n + 1}/{len(QS)}] {q}: rows " + ", ".join(f"{m} {R[q][m]['rows']}" for m in METHODS)
        + f"; exh rows outside window top-200: {R[q]['exh_not_in_window_candidates']}; MMseqs2 hits accepted "
        + "/".join(str(R[q]['mmseqs_hits_accepted'][m]) for m in METHODS) + f" of {len(mmh)}")

qs = [q for q, _ in QS]
med = lambda m, k: float(np.median([R[q][m][k] for q in qs]))
L = [f"# Swiss-Prot MSAs, {len(qs)} e2e targets: exhaustive vs two-stage vs window index", "",
     f"Same aligner for all (640-bit embedding SW, bias {BIAS}, gap {GAP}) and the same cutoff: database-scale "
     f"E < {E_CUT} (E = 576,073 x p; p from a Gumbel fit to the tail of 10,000 reversed-sequence decoys). "
     f"Two-stage = {VAR} first pass, top {FRAC:.0%} rescored. Files: ~/folding-data/poc/msa_exh/<method>/<target>.a3m",
     "", f"Calibration check: held-out decoys at E < 1, per target, mean "
     f"{np.mean([R[q]['decoys_heldout_E<1'] for q in qs]):.2f} (expected {R[qs[0]]['decoys_heldout_expected']:.3f}).",
     "", "| method | candidates / target | s / target | rows median | Neff80 median | targets with 0 rows | "
     "MMseqs2 hits accepted (sum) | database bytes scanned |", "|---|---|---|---|---|---|---|---|"]
bts = {"exh640": "16.7 GB (80 B/residue)", "twostage": f"{sym.nbytes / 1e9:.2f} GB first pass + 1% at 80 B/res",
       "window": "143 MB window index + 200 proteins"}
for m in METHODS:
    L.append(f"| {m} | {np.median([R[q][m]['candidates'] for q in qs]):,.0f} | {np.median(T[m]):.2f} | {med(m, 'rows'):.0f} | "
             f"{med(m, 'neff80'):.1f} | {sum(R[q][m]['rows'] == 0 for q in qs)} | "
             f"{sum(R[q]['mmseqs_hits_accepted'][m] for q in qs)} / {sum(R[q]['mmseqs_hits'] for q in qs)} | {bts[m]} |")
L.append(f"| depth_ctrl (2x2 binplm_emb rows, subsampled to exh640 depth) | - | - | {med('depth_ctrl', 'rows'):.0f} | "
         f"{med('depth_ctrl', 'neff80'):.1f} | - | - | - |")
L += ["", "## Per target (rows)", "", "| target | L | exh640 | twostage | window | exh rows not in window top-200 | "
      "MMseqs2 hits accepted exh/2s/win of n |", "|---|---|---|---|---|---|---|"]
for q in qs:
    d = R[q]
    L.append(f"| {q} | {d['L']} | {d['exh640']['rows']} | {d['twostage']['rows']} | {d['window']['rows']} | "
             f"{d['exh_not_in_window_candidates']} | " + "/".join(str(d['mmseqs_hits_accepted'][m]) for m in METHODS)
             + f" of {d['mmseqs_hits']} |")
(RESULTS / "exh_sprot.md").write_text("\n".join(L) + "\n")
(RESULTS / "exh_sprot.json").write_text(json.dumps({"per_target": R, "times": T, "E_cut": E_CUT, "variant": VAR,
                                                    "frac": FRAC}, indent=1))
log("\n".join(L))
