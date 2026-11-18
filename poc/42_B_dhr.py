"""Test B: DHR on the SCOPe benchmark (all 15,177 domains vs all), same scoring as every other Phase 1 method.
Official DHR weights; database side = context encoder, query side = query encoder; exact L2 ranking (their IndexFlatL2);
fused attention verified equal to the original (36_dhr.py).  Run in the 'fold' env."""
import importlib.util
import sys
import time

import numpy as np
import torch

sys.argv = sys.argv[:1]
spec = importlib.util.spec_from_file_location("p1common", "/mnt/c/Folding/phase1/common.py")
p1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p1)
import esm.multihead_attention as _mha                    # noqa: E402
from pathlib import Path                                  # noqa: E402

import torch.nn.functional as F                           # noqa: E402


def _fast_forward(self, query, key=None, value=None, key_padding_mask=None, incremental_state=None, need_weights=True,
                  static_kv=False, attn_mask=None, before_softmax=False, need_head_weights=False):
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
    return self.out_proj(o.permute(2, 0, 1, 3).reshape(T, B, C)), None


_mha.MultiheadAttention.forward = _fast_forward
WD = Path.home() / "tools" / "dhr_weights"
qenc, alphabet = torch.load(WD / "dhr_qencoder.pt", weights_only=False)
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
        j = i
        while j < len(order) and (min(len(seqs[order[j]]), 1022) + 2) * (j - i + 1) <= max_tokens:
            j += 1
        j = max(j, i + 1)
        idx = order[i:j]
        _, _, toks = bc([(str(k), seqs[k][:1022]) for k in idx])
        out[idx] = model(toks.cuda(), repr_layers=[NL])["representations"][NL][:, 0].float().cpu().numpy()
        i = j
    return out


_, seqs = p1.read_fasta(p1.DATA / "seqs.fasta")
t0 = time.time()
Dv = cls_vectors(cenc, seqs)
Qv = cls_vectors(qenc, seqs)
dn = (Dv * Dv).sum(1)
ranked = []
for a in range(0, len(seqs), 2000):
    d = dn[None, :] - 2 * Qv[a:a + 2000] @ Dv.T
    top = np.argpartition(d, p1.TOP_N, axis=1)[:, :p1.TOP_N]
    for r, t in zip(d, top):
        ranked.append(t[np.argsort(r[t])].astype(np.int32))
p1.save_ranking("dhr", ranked, time.time() - t0, 480 * 2, note="DHR official weights, L2, fp16 vectors")
print("B DHR DONE")
