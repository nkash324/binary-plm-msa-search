"""Embed every domain with ESM-2 150M: per-residue (fp16, on disk) + mean-pooled per-protein."""
import time

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

from common import DATA, EMB_DIM, ESM_MODEL, MAX_LEN, read_fasta

# Same ESM-2 weights as fair-esm, loaded from Hugging Face (Meta's original server cert has expired).
# last_hidden_state == fair-esm's final-layer representation (includes the final layer norm).
HF_NAME = f"facebook/{ESM_MODEL}"

TOKENS_PER_BATCH = 12000

ids, seqs = read_fasta(DATA / "seqs.fasta")
N = len(seqs)
lens = np.array([min(len(s), MAX_LEN) for s in seqs])
offsets = np.zeros(N + 1, dtype=np.int64)
offsets[1:] = np.cumsum(lens)
print(f"{N} sequences, {offsets[-1]:,} residues -> "
      f"{offsets[-1] * EMB_DIM * 2 / 1e9:.2f} GB of fp16 residue embeddings")

tokenizer = AutoTokenizer.from_pretrained(HF_NAME)
model = EsmModel.from_pretrained(HF_NAME, add_pooling_layer=False).eval().cuda().half()


def forward(toks, mask):
    return model(input_ids=toks, attention_mask=mask).last_hidden_state

res = np.lib.format.open_memmap(DATA / "res_emb.npy", mode="w+", dtype=np.float16,
                                shape=(int(offsets[-1]), EMB_DIM))
prot = np.zeros((N, EMB_DIM), dtype=np.float32)

# Length-sorted batches under a token budget.
order = np.argsort(-lens)
batches, cur, cur_max = [], [], 0
for i in order:
    L = lens[i] + 2
    if cur and max(cur_max, L) * (len(cur) + 1) > TOKENS_PER_BATCH:
        batches.append(cur)
        cur, cur_max = [], 0
    cur.append(i)
    cur_max = max(cur_max, L)
batches.append(cur)

t0 = time.time()
n_fallback = 0
with torch.inference_mode():
    for b, idxs in enumerate(batches):
        enc = tokenizer([seqs[i][:MAX_LEN] for i in idxs], padding=True, return_tensors="pt")
        toks, mask = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
        rep = forward(toks, mask)                  # token 0 = <cls>, residues start at 1
        if not torch.isfinite(rep).all():          # rare fp16 overflow -> redo in fp32
            n_fallback += 1
            model.float()
            rep = forward(toks, mask)
            model.half()
        for k, i in enumerate(idxs):
            r = rep[k, 1:lens[i] + 1].float()
            res[offsets[i]:offsets[i + 1]] = r.half().cpu().numpy()
            prot[i] = r.mean(0).cpu().numpy()
        if b % 50 == 0:
            print(f"  batch {b}/{len(batches)}  {time.time() - t0:.0f}s", flush=True)

res.flush()
np.save(DATA / "offsets.npy", offsets)
np.save(DATA / "prot_emb.npy", prot)
print(f"Done in {time.time() - t0:.0f}s (fp32 fallbacks: {n_fallback})")
