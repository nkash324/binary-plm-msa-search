"""Step 0: check the exhaustive engine against emb_align, then time it (10 SCOPe queries x all 15,177 domains).
Writes poc/results/esw_smoke.md.  Run in the 'fold' env."""
import json
import time
from pathlib import Path

import numpy as np
import torch

import emb_align as ea
import esw
from common import DATA, PHASE1, RESULTS

torch.cuda.set_per_process_memory_fraction(0.6)
rng = np.random.default_rng(0)
bits = np.load(PHASE1 / "res_bits.npy", mmap_mode="r")
off = np.load(PHASE1 / "offsets.npy")
N, lens = len(off) - 1, np.diff(off)
P = json.loads((DATA / "emb_align_params.json").read_text())
BIAS, GAP = P["bias"], P["gap"]
code = lambda i: esw.unpack(bits[off[i]:off[i + 1]])

# 1. correctness: 5 queries x 10 random targets each, engine vs emb_align.sw_many
err = []
for q in rng.choice(N, 5, replace=False):
    ts = np.sort(rng.choice(N, 10, replace=False))
    a = esw.search([code(q)], bits, off, BIAS, GAP, targets=ts)[0]
    b, _ = ea.sw_many(ea.pm1(bits[off[q]:off[q + 1]]), [ea.pm1(bits[off[t]:off[t + 1]]) for t in ts], BIAS, GAP)
    err.append(np.abs(a - b).max())
print(f"max |engine - emb_align| = {max(err):.2e}", flush=True)

# 2. timing: 10 queries against everything (one pass), after a 1-query warm-up
qs = rng.choice(N, 10, replace=False)
esw.search([code(qs[0])], bits, off, BIAS, GAP, targets=np.arange(2000))
torch.cuda.synchronize()
mem0 = torch.cuda.max_memory_allocated()
t0 = time.time()
S = esw.search([code(q) for q in qs], bits, off, BIAS, GAP)
torch.cuda.synchronize()
dt = time.time() - t0
cells = float(sum(lens[q] for q in qs) * off[-1])
self_top = float(np.mean([np.argmax(S[k]) == q for k, q in enumerate(qs)]))
res = {"max_abs_diff": float(max(err)), "seconds_per_query": dt / len(qs), "gcups": cells / dt / 1e9,
       "gpu_peak_GB": torch.cuda.max_memory_allocated() / 1e9, "self_is_top_hit": self_top,
       "mean_query_len": float(np.mean(lens[qs]))}
# projections from the measured speed
cell_rate = cells / dt
sp_res = 209_202_719
res["proj_scope_1000q_with_decoys_min"] = 2 * 1000 * 185 * off[-1] / cell_rate / 60
res["proj_sprot_20q_min"] = 20 * 150 * sp_res / cell_rate / 60
L = ["# Step 0: exhaustive embedding-SW engine smoke test", "",
     f"- correctness vs emb_align.sw_many (50 pairs): max abs diff {res['max_abs_diff']:.1e}",
     f"- 10 SCOPe queries (mean length {res['mean_query_len']:.0f}) x 15,177 domains: "
     f"**{res['seconds_per_query']:.2f} s/query**, {res['gcups']:.2f} G cells/s, GPU peak {res['gpu_peak_GB']:.2f} GB",
     f"- query itself is the top hit: {self_top:.0%}",
     f"- projected: SCOPe 1000 queries + decoys {res['proj_scope_1000q_with_decoys_min']:.0f} min; "
     f"Swiss-Prot 20 queries {res['proj_sprot_20q_min']:.0f} min"]
(RESULTS / "esw_smoke.md").write_text("\n".join(L) + "\n")
(RESULTS / "esw_smoke.json").write_text(json.dumps(res, indent=1))
print("\n".join(L))
