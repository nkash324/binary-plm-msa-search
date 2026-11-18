"""Quick full-length test for BinPLM Swiss-Prot method (combined settings, unchanged), same 8 proteins as
32_fulllength_quick.py: search with the full-length parent (max 1022 aa around the domain), crop the MSA rows back to
the domain, fold the domain.  Compare with the domain-query result and with MMseqs2 (domain / full-length).
Run in the 'fold' env (MSAs), then 'cf' env with --fold.  Writes e2e/a3m/yours_full and results/yours_full.md."""
import glob
import json
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np

from common import DATA, RESULTS, read_fasta

E = DATA / "e2e"
QS = ["d1cr5a1", "d2dzqa1", "d1xg0a_", "d1odha_", "d1pbya4", "d2b8ma1", "d5muaa1", "d3reta_"]
LOG = open(RESULTS / "yours_full.log", "a")


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


qids, qseqs = read_fasta(E / "queries.fasta")
QSEQ = dict(zip(qids, qseqs))

if "--fold" not in sys.argv:
    import torch
    from Bio import Align
    from Bio.Align import substitution_matrices
    from transformers import AutoTokenizer, EsmModel

    import emb_align as ea
    from common import DB_FASTA

    def get(url):
        for k in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    return r.read().decode()
            except Exception:                               # noqa: BLE001
                time.sleep(2 * (k + 1))
        return None

    al = Align.PairwiseAligner(mode="local", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                               open_gap_score=-10, extend_gap_score=-0.5)
    PAR = json.loads((E / "full_quick.json").read_text())["parents"]

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

    stats = {}
    for q in QS:
        pa = PAR[q]
        parent = "".join(get(f"https://rest.uniprot.org/uniprotkb/{pa['acc']}.fasta").splitlines()[1:])
        s0 = pa["win_start"]
        win = parent[s0:s0 + 1022]
        a = al.align(QSEQ[q], parent)[0]
        mp = {}
        for (d0, d1), (p0, p1) in zip(*a.aligned):
            for k in range(d1 - d0):
                if 0 <= p0 + k - s0 < len(win):
                    mp[d0 + k] = p0 + k - s0
        # ---- BinPLM pipeline, combined settings (as 18_e2e.py), query = full-length window
        b640, Q = embed(win)
        sc = window_search(b640)
        cand = np.argsort(-sc)[:200]
        cand = cand[sc[cand] > 0]
        cs_ = set(cand.tolist())
        null = np.array([p for p in null_pool if p not in cs_])[:200]
        ns, _ = ea.sw_many(Q, [sp(p) for p in null], BIAS, GAP)
        Al = np.c_[np.ones(len(null)), np.log(lens[null])]
        coef, *_ = np.linalg.lstsq(Al, ns, rcond=None)
        sd = np.std(ns - Al @ coef)
        zf = lambda s, ps: (s - np.c_[np.ones(len(ps)), np.log(lens[ps])] @ coef) / sd
        q90, q99 = np.quantile(zf(ns, null), [0.9, 0.99])
        beta = (q99 - q90) / (np.log(-np.log(0.9)) - np.log(-np.log(0.99)))
        thr = q90 + beta * (-np.log(-np.log(1 - 1 / 2000)) + np.log(-np.log(0.9)))
        Tc = [sp(p) for p in cand]
        csc, _ = ea.sw_many(Q, Tc, BIAS, GAP)
        keep = np.flatnonzero(zf(csc, cand) > thr)
        rows = []
        if len(keep):
            _, pr = ea.sw_many(Q, [Tc[k] for k in keep], BIAS, GAP, keep=True)
            for k, pairs in zip(keep, pr):
                r = ["-"] * len(win)
                for i, j in pairs:
                    r[i] = db_seqs[cand[k]][j]
                rows.append((db_ids[cand[k]], "".join(r)))
        # ---- crop to the domain
        crop = []
        for n, r in rows:
            d = "".join(r[mp[i]] if i in mp else "-" for i in range(len(QSEQ[q])))
            if d.strip("-"):
                crop.append((n, d))
        (E / "a3m" / "yours_full").mkdir(parents=True, exist_ok=True)
        (E / "a3m" / "yours_full" / f"{q}.a3m").write_text(f">{q}\n{QSEQ[q]}\n" + "".join(f">{n}\n{r}\n" for n, r in crop))
        stats[q] = {"rows_full": len(rows), "rows_domain": len(crop)}
        log(f"  BinPLM {q}: {len(rows)} accepted for the full-length query -> {len(crop)} cover the domain")
    (E / "yours_full.json").write_text(json.dumps(stats, indent=1))
    log("MSAS DONE")
    sys.exit()

# ================================================================ fold + report ('cf' env)
CF = ["--num-models", "1", "--num-recycle", "3", "--model-type", "alphafold2_ptm"]
p = "yours_full"


def pdb(p, q):
    f = sorted(glob.glob(str(E / "out" / p / f"{q}_unrelaxed_rank_001*.pdb")))
    return f[0] if f else None


t0 = time.time()
shutil.rmtree(E / "out" / p, ignore_errors=True)
subprocess.run(["colabfold_batch", str(E / "a3m" / p), str(E / "out" / p), *CF], capture_output=True, text=True)
for q in QS:
    if not pdb(p, q):
        d = E / "retry_full" / p / q
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(E / "a3m" / p / f"{q}.a3m", d)
        subprocess.run(["colabfold_batch", str(d), str(E / "out" / p), *CF], capture_output=True, text=True)
log(f"folded {p} in {time.time() - t0:.0f}s")


def tm(f, q):
    if not f:
        return np.nan
    o = subprocess.run(["USalign", str(f), str(E / "ref" / f"{q}.ent"), "-outfmt", "2"], capture_output=True, text=True).stdout
    return float([l for l in o.splitlines() if l and not l.startswith("#")][0].split("\t")[3])


T = {k: {q: tm(pdb(k, q), q) for q in QS} for k in ("mmseqs2", "mm_full", "yours", "yours_full")}
st = json.loads((E / "yours_full.json").read_text())
olds = {q: sum(1 for l in (E / "a3m" / "yours" / f"{q}.a3m").read_text().splitlines() if l.startswith(">")) - 1 for q in QS}
NM = {"mmseqs2": "MMseqs2, domain query", "mm_full": "MMseqs2, full-length query",
      "yours": "BinPLM (Swiss-Prot), domain query", "yours_full": "BinPLM (Swiss-Prot), full-length query"}
L = [f"# Quick full-length test: BinPLM Swiss-Prot method vs MMseqs2, {len(QS)} proteins", "",
     "Search with the full-length parent protein (max 1022 aa around the domain), MSA cropped to the domain, domain folded "
     "(ColabFold, 1 model, 3 recycles, no templates). BinPLM unchanged (combined settings).", "",
     "| MSA | mean TM | correct folds (TM>0.5) |", "|---|---|---|"]
for k in T:
    t = np.array([T[k][q] for q in QS])
    L.append(f"| {NM[k]} | {np.nanmean(t):.3f} | {(t > 0.5).sum()}/{len(QS)} |")
L += ["", "| protein | BinPLM rows domain / full | BinPLM TM domain / full | MMseqs2 TM domain / full |", "|---|---|---|---|"]
for q in QS:
    L.append(f"| {q} | {olds[q]} / {st[q]['rows_domain']} | {T['yours'][q]:.2f} / {T['yours_full'][q]:.2f} | "
             f"{T['mmseqs2'][q]:.2f} / {T['mm_full'][q]:.2f} |")
(RESULTS / "yours_full.md").write_text("\n".join(L) + "\n")
log("\n".join(L))
