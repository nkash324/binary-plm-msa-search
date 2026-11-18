"""1-minute check: does PyTorch SDPA attention speed up ESM-2 without changing the outputs?"""
import time

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

from common import DB_FASTA, MAX_LEN, read_fasta

HF = "facebook/esm2_t30_150M_UR50D"
tok = AutoTokenizer.from_pretrained(HF)
ids, seqs = read_fasta(DB_FASTA)
rng = np.random.default_rng(1)
sample = sorted((seqs[i][:MAX_LEN] for i in rng.choice(len(seqs), 600, replace=False)), key=len, reverse=True)
batches, cur = [], []
for s in sample:                                   # same token-budget batching as 02_embed.py
    if cur and (len(cur[0]) + 2) * (len(cur) + 1) > 12000:
        batches.append(cur)
        cur = []
    cur.append(s)
batches.append(cur)
n_res = sum(map(len, sample))


def run(impl):
    model = EsmModel.from_pretrained(HF, add_pooling_layer=False, attn_implementation=impl).eval().cuda().half()
    outs = []
    with torch.inference_mode():
        for b in batches[:2]:                      # warm-up
            enc = tok(b, padding=True, return_tensors="pt").to("cuda")
            model(**enc)
        torch.cuda.synchronize()
        t = time.time()
        for b in batches:
            enc = tok(b, padding=True, return_tensors="pt").to("cuda")
            rep = model(**enc).last_hidden_state.float()
            outs += [rep[k, 1:len(s) + 1].cpu() for k, s in enumerate(b)]
        torch.cuda.synchronize()
        secs = time.time() - t
    print(f"{impl:6s}: {secs:.1f} s for {n_res:,} residues -> {n_res / secs:,.0f} residues/s "
          f"(full Swiss-Prot ~{210e6 / (n_res / secs) / 3600:.1f} h unthrottled)", flush=True)
    del model
    torch.cuda.empty_cache()
    return torch.cat(outs), secs


a, ta = run("eager")
b, tb = run("sdpa")
mu = a.mean(0)
cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
bits = ((a - mu) >= 0) == ((b - mu) >= 0)
print(f"speedup {ta / tb:.2f}x | per-residue cosine eager vs sdpa: min {cos.min():.5f}, mean {cos.mean():.6f} "
      f"| sign bits identical: {bits.float().mean() * 100:.3f}%")
