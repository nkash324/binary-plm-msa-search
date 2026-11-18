"""Test B: PLMSearch (Liu et al., Nat Commun 2024) on the SCOPe benchmark, all vs all.
Exactly their scoring (plmsearch/main_similarity.py, 'ss_predictor' mode, i.e. without the optional Pfam-clan prefilter):
  z = ESM-1b layer-33 mean over residues (1280 numbers, first 1022 residues)
  score(q, t) = cos(q, t)                                      if cos > 0.995
              = cos(q, t) * sigmoid((W z_q + b) . z_t)         otherwise      (W, b = their trained plmsearch.sav)
ESM-1b from Meta's official download (fair-esm).  Run in the 'fold' env."""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("p1common", "/mnt/c/Folding/phase1/common.py")
p1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p1)

SAV = Path(sys.argv[1])                                    # path to plmsearch.sav (state dict: one Linear 1280x1280)
_, seqs = p1.read_fasta(p1.DATA / "seqs.fasta")
emb_f = p1.DATA / "esm1b_mean.npy"
t0 = time.time()
if not emb_f.exists():
    # Meta's ESM-1b weights via Hugging Face, fp16 (the fair-esm .pt is 7.8 GB in the legacy format and would
    # have to be read fully into RAM; same network, last_hidden_state = layer 33 after the final layer norm)
    from transformers import AutoTokenizer, EsmModel
    tok = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
    model = EsmModel.from_pretrained("facebook/esm1b_t33_650M_UR50S", add_pooling_layer=False,
                                     torch_dtype=torch.float16, low_cpu_mem_usage=True).eval().cuda()
    order = np.argsort([len(s) for s in seqs])
    Z = np.zeros((len(seqs), 1280), np.float32)
    i = 0
    with torch.no_grad():
        while i < len(order):
            j = i
            while j < len(order) and (min(len(seqs[order[j]]), 1022) + 2) * (j - i + 1) <= 16384:
                j += 1
            j = max(j, i + 1)
            idx = order[i:j]
            enc = tok([seqs[k][:1022] for k in idx], return_tensors="pt", padding=True).to("cuda")
            out = model(**enc).last_hidden_state
            for b, k in enumerate(idx):
                Z[k] = out[b, 1:min(len(seqs[k]), 1022) + 1].float().mean(0).cpu().numpy()
            i = j
    np.save(emb_f, Z)
    del model
    torch.cuda.empty_cache()
emb_s = time.time() - t0
Z = torch.from_numpy(np.load(emb_f)).cuda()
sd = torch.load(SAV, map_location="cpu", weights_only=True)
W, b = sd["projection_linear.weight"].cuda(), sd["projection_linear.bias"].cuda()
Zn = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-8)
t1 = time.time()
ranked = []
with torch.no_grad():
    for a in range(0, len(seqs), 1000):
        q = Z[a:a + 1000]
        cos = Zn[a:a + 1000] @ Zn.T
        sim = torch.sigmoid((q @ W.T + b) @ Z.T)
        s = torch.where(cos > 0.995, cos, cos * sim)
        top = torch.topk(s, p1.TOP_N, dim=1).indices.cpu().numpy().astype(np.int32)
        ranked += list(top)
p1.save_ranking("plmsearch_nopfam", ranked, emb_s + time.time() - t1, 1280 * 4,
                note="PLMSearch ss_predictor scoring (no Pfam prefilter), ESM-1b mean")
print("B PLMSEARCH DONE")
