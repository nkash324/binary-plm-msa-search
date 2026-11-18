"""DHR (Dense Homolog Retriever, Hong et al., Nat Biotech 2024) on the SAME database as everything else (Swiss-Prot),
for the 19 fresh hard proteins.  Faithful to the official code (github.com/ml4bio/Dense-Homolog-Retrieval):
  * two fine-tuned ESM-2 35M encoders (official weights): database proteins -> context encoder, queries -> query encoder,
    vector = last-layer <cls> token (480 numbers), exact L2 search (their IndexFlatL2)
  * top K retrieved -> JackHMMER over the retrieved set with DHR's flags (-N 1, incE/E 1e-3, F1/F2/F3 as in their script)
    (standard HMMER jackhmmer instead of their bundled pre-compiled 'qjackhmmer' binary)
K = 2000 (~0.35% of Swiss-Prot; their default 400,000 is ~0.3% of UniRef90), fixed before seeing any results.
Run in the 'fold' env.  Writes e2e/a3m/dhr, e2e/dhr.json."""
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from common import DATA, DB_FASTA, RESULTS, read_fasta

E = DATA / "e2e"
WD = Path.home() / "tools" / "dhr_weights"
K = 2000
LOG = open(RESULTS / "dhr.log", "a")


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


import esm.multihead_attention as _mha                    # noqa: E402
import torch.nn.functional as F                           # noqa: E402

_orig_forward = _mha.MultiheadAttention.forward


def _fast_forward(self, query, key=None, value=None, key_padding_mask=None, incremental_state=None, need_weights=True,
                  static_kv=False, attn_mask=None, before_softmax=False, need_head_weights=False):
    """Same maths as fair-esm self-attention (q/k/v projections, rotary embedding, softmax(qk/sqrt(d))v, out_proj),
    computed with PyTorch's fused scaled_dot_product_attention instead of materialising the full attention matrix."""
    T, B, C = query.size()
    H, d = self.num_heads, self.head_dim
    q = self.q_proj(query).contiguous().view(T, B * H, d).transpose(0, 1)
    k = self.k_proj(query).contiguous().view(T, B * H, d).transpose(0, 1)
    v = self.v_proj(query).contiguous().view(T, B * H, d).transpose(0, 1)
    if self.rot_emb:
        q, k = self.rot_emb(q, k)
    q, k, v = (x.view(B, H, T, d) for x in (q, k, v))
    m = None if key_padding_mask is None else ~key_padding_mask.bool()[:, None, None, :]
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
    o = o.permute(2, 0, 1, 3).reshape(T, B, C)
    return self.out_proj(o), None


qenc, alphabet = torch.load(WD / "dhr_qencoder.pt", weights_only=False)   # checked: only esm/torch classes inside
cenc, _ = torch.load(WD / "dhr_cencoder.pt", weights_only=False)
qenc, cenc = qenc.eval().cuda().half(), cenc.eval().cuda().half()
bc = alphabet.get_batch_converter()
NL = cenc.num_layers


@torch.no_grad()
def cls_vectors(model, seqs, max_tokens=24_000):
    order = np.argsort([len(s) for s in seqs])
    out = np.zeros((len(seqs), 480), np.float32)
    i = 0
    while i < len(order):
        L = min(len(seqs[order[i]]), 1022) + 2
        n = max(1, max_tokens // max(L, 1))
        j = i
        while j < len(order) and j - i < n and (min(len(seqs[order[j]]), 1022) + 2) * (j - i + 1) <= max_tokens:
            j += 1
        j = max(j, i + 1)
        idx = order[i:j]
        _, _, toks = bc([(str(k), seqs[k][:1022]) for k in idx])
        rep = model(toks.cuda(), repr_layers=[NL])["representations"][NL][:, 0]
        out[idx] = rep.float().cpu().numpy()
        i = j
    return out


# ---------------------------------------------------------------- database embedding (one-time)
db_ids, db_seqs = read_fasta(DB_FASTA)

# check the fast attention gives the same vectors as the original code (64 random proteins, batched with padding)
chk = list(np.random.default_rng(1).choice(len(db_seqs), 64, replace=False))
_mha.MultiheadAttention.forward = _orig_forward
ref = cls_vectors(cenc, [db_seqs[i] for i in chk], max_tokens=8000)
_mha.MultiheadAttention.forward = _fast_forward
new = cls_vectors(cenc, [db_seqs[i] for i in chk], max_tokens=8000)
cos = (ref * new).sum(1) / np.linalg.norm(ref, axis=1) / np.linalg.norm(new, axis=1)
log(f"DHR fast-attention check: cosine to original min {cos.min():.5f}, mean {cos.mean():.5f}")
assert cos.min() > 0.999, "fast attention does not reproduce the original vectors"
t_chk = time.time()
cls_vectors(cenc, [db_seqs[i] for i in range(50_000, 52_000)])
log(f"DHR speed: 2000 proteins in {time.time() - t_chk:.1f}s with fast attention")
emb_f = DATA / "dhr_sprot_emb.npy"
if not emb_f.exists():
    t = time.time()
    part = DATA / "dhr_parts"
    part.mkdir(exist_ok=True)
    for a in range(0, len(db_seqs), 50_000):                # saved per chunk: a GPU hiccup only loses one chunk
        f = part / f"{a}.npy"
        if not f.exists():
            np.save(f, cls_vectors(cenc, db_seqs[a:a + 50_000]).astype(np.float16))
            log(f"DHR: embedded proteins {a:,}-{min(a + 50_000, len(db_seqs)):,}")
    V = np.concatenate([np.load(part / f"{a}.npy") for a in range(0, len(db_seqs), 50_000)])
    np.save(emb_f, V)
    log(f"DHR: embedded Swiss-Prot ({len(db_seqs):,} proteins) in {(time.time() - t) / 60:.1f} min (one-time)")
V = np.load(emb_f).astype(np.float32)
vn = (V * V).sum(1)

# ---------------------------------------------------------------- queries
qs = sorted({k.split("|")[1] for k in json.loads((RESULTS / "e2e.json").read_text())["accuracy"]})
qids, qseqs = read_fasta(E / "queries.fasta")
QSEQ = dict(zip(qids, qseqs))
W = E / "work_dhr"
W.mkdir(exist_ok=True)
(E / "a3m" / "dhr").mkdir(parents=True, exist_ok=True)


def sto_to_a3m(sto, q, seq):
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


stats = {}
for q in qs:
    s = QSEQ[q]
    T = {}
    t = time.perf_counter()
    qv = cls_vectors(qenc, [s])[0]
    torch.cuda.synchronize()
    T["embed query"] = time.perf_counter() - t
    t = time.perf_counter()
    d = vn - 2 * V @ qv                                      # squared L2 up to a constant (exact flat search)
    top = np.argpartition(d, K)[:K]
    top = top[np.argsort(d[top])]
    T["vector search"] = time.perf_counter() - t
    t = time.perf_counter()
    sub, qf, sto = W / f"{q}_db.fa", W / f"{q}.fa", W / f"{q}.sto"
    sub.write_text("".join(f">{db_ids[i]}\n{db_seqs[i]}\n" for i in top))
    qf.write_text(f">{q}\n{s}\n")
    subprocess.run(["jackhmmer", "-N", "1", "--incE", "1e-3", "-E", "1e-3", "--F1", "0.0005", "--F2", "5e-05",
                    "--F3", "5e-07", "--cpu", "6", "--noali", "-A", str(sto), "-o", "/dev/null", str(qf), str(sub)],
                   check=True)
    a3m, n = (sto_to_a3m(sto, q, s) if sto.exists() and sto.stat().st_size else (f">{q}\n{s}\n", 0))
    (E / "a3m" / "dhr" / f"{q}.a3m").write_text(a3m)
    T["jackhmmer on retrieved"] = time.perf_counter() - t
    stats[q] = {"rows": n, "seconds": T}
    log(f"  DHR {q}: top {K} retrieved -> {n} MSA rows ({sum(T.values()):.1f}s: " +
        ", ".join(f"{k} {v:.2f}" for k, v in T.items()) + ")")
(E / "dhr.json").write_text(json.dumps(stats, indent=1))
log("DHR MSAS DONE")
