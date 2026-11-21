"""Step 1a: leak-free SCOPe codes + decoys.

1. Re-encode the SCOPe40 residues (phase1/res_emb.npy) as 640 sign bits around the SWISS-PROT residue mean
   (poc/mu.npz), not the SCOPe mean -> phase1/res_bits_sp.npy. Same code space as the Swiss-Prot database.
2. Decoys: embed every SCOPe domain's sequence REVERSED (same truncation, ESM-2 150M, same recipe as
   phase1/02_embed.py) -> phase1/decoy_res_bits_sp.npy (same offsets as the real set). Bits only, no floats kept.
Run in the 'fold' env (~15-20 min GPU)."""
import time

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

from common import DATA, PHASE1, read_fasta

torch.cuda.set_per_process_memory_fraction(0.6)
PAUSE = DATA / "PAUSE"
off = np.load(PHASE1 / "offsets.npy")
lens = np.diff(off)
mu = np.load(DATA / "mu.npz")["res"].astype(np.float32)
t0 = time.time()

# 1. re-encode the real residues
out = PHASE1 / "res_bits_sp.npy"
if not out.exists():
    res = np.load(PHASE1 / "res_emb.npy", mmap_mode="r")
    old = np.load(PHASE1 / "res_bits.npy", mmap_mode="r")
    o = np.lib.format.open_memmap(out.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint8, shape=(len(res), 80))
    agree = []
    for s in range(0, len(res), 200_000):
        o[s:s + 200_000] = b = np.packbits(np.asarray(res[s:s + 200_000], np.float32) - mu >= 0, axis=1)
        agree.append(1 - np.unpackbits(b ^ np.asarray(old[s:s + 200_000]), axis=1).mean())
    o.flush()
    del o
    out.with_suffix(".tmp.npy").rename(out)
    print(f"re-encoded {len(res):,} residues with the Swiss-Prot mean ({time.time() - t0:.0f}s); "
          f"bits equal to the old SCOPe-mean codes: {np.mean(agree):.1%}", flush=True)

# 2. reversed-sequence decoys
dout = PHASE1 / "decoy_res_bits_sp.npy"
if not dout.exists():
    _, seqs = read_fasta(PHASE1 / "seqs.fasta")
    rev = [s[:L][::-1] for s, L in zip(seqs, lens)]
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D", add_pooling_layer=False).eval().cuda().half()
    gmu = torch.from_numpy(mu).cuda()
    SH8 = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], device="cuda", dtype=torch.uint8)
    o = np.lib.format.open_memmap(dout.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint8, shape=(int(off[-1]), 80))
    order, batches, cur, cm = np.argsort(-lens), [], [], 0
    for i in order:                                   # length-sorted batches under a 12k-token budget
        L = lens[i] + 2
        if cur and max(cm, L) * (len(cur) + 1) > 12000:
            batches.append(cur)
            cur, cm = [], 0
        cur.append(i)
        cm = max(cm, L)
    batches.append(cur)
    t1 = time.time()
    with torch.inference_mode():
        for b, idxs in enumerate(batches):
            while PAUSE.exists():
                time.sleep(10)
            enc = tok([rev[i] for i in idxs], padding=True, return_tensors="pt")
            ids, m = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
            rep = model(input_ids=ids, attention_mask=m).last_hidden_state
            if not torch.isfinite(rep).all():         # rare fp16 overflow -> redo in fp32
                model.float()
                rep = model(input_ids=ids, attention_mask=m).last_hidden_state
                model.half()
            for k, i in enumerate(idxs):
                sg = rep[k, 1:lens[i] + 1].float() - gmu >= 0
                o[off[i]:off[i + 1]] = (sg.view(-1, 80, 8).to(torch.uint8) * SH8).sum(-1, dtype=torch.uint8).cpu().numpy()
            if b % 100 == 0:
                print(f"  decoy batch {b}/{len(batches)} {time.time() - t1:.0f}s", flush=True)
    o.flush()
    del o
    dout.with_suffix(".tmp.npy").rename(dout)
    print(f"decoys embedded ({time.time() - t1:.0f}s)", flush=True)
print(f"step 1a done ({time.time() - t0:.0f}s)")
