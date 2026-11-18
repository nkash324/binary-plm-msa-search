"""Test A, stage 3a: ESM-2 3B query vectors -> 256-bit Atlas codes for the shallow proteins, then the model is freed.
Weights from the safetensors copy of facebook/esm2_t36_3B_UR50D (Hugging Face conversion revision), streamed tensor by
tensor straight to the GPU in fp16, so WSL never holds the 10 GB .bin in RAM.  Writes e2e/dark/atlas_qcode.npz.
Run in the 'fold' env."""
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

from common import DATA, read_fasta

D, A = DATA / "e2e" / "dark", Path.home() / "folding-data" / "atlas"
REV = "6c3c8884b1ac09c09c27990edfe8d3680216e4f2"           # safetensors revision of the same weights
shallow = [l.strip() for l in (D / os.environ.get("A_SHALLOW", "shallow.txt")).read_text().split() if l.strip()]
cids, cseqs = read_fasta(D / "candidates.fasta")
QSEQ = dict(zip(cids, cseqs))
apca = np.load(A / "pca.npz")
amu, aP = apca["mu"], apca["P"]
tok = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
m = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D", revision=REV, use_safetensors=True, add_pooling_layer=False,
                             torch_dtype=torch.float16, device_map="cuda", low_cpu_mem_usage=True).eval()
codes = {}
with torch.no_grad():
    for q in shallow:
        s = QSEQ[q]
        e = m(**tok([s], return_tensors="pt").to("cuda")).last_hidden_state[0, 1:len(s) + 1].float().mean(0).cpu().numpy()
        codes[q] = np.packbits(((e - amu) @ aP > 0)[None], axis=1)
np.savez(D / "atlas_qcode.npz", **codes)
print(f"ATLAS QVEC DONE: {len(codes)} proteins")
