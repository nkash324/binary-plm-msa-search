"""GPU speed test for ESM-2 per-residue embedding on the Illinois Campus Cluster.

Measures residues/second for the same work 02_embed.py does in production:
tokenize -> ESM-2 forward (fp16, SDPA) -> sign bits (640 -> 80 bytes/residue) -> copy to CPU.
No GPU throttling (02_embed.py's rest/pause logic is for the home desktop only).

Tries several token-batch sizes per model, stops a model's sweep on out-of-memory, and
projects full-database embedding time and disk use from the best setting.

Example:
  python bench_embed.py --fasta /scratch/$USER/binplm/sprot.fasta --n 20000 \
      --models esm2_t30_150M_UR50D esm2_t33_650M_UR50D --out bench.json
"""
import argparse
import json
import os
import platform
import subprocess
import time

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

MAX_LEN = 1022
# Approximate database sizes (residues). Swiss-Prot is from this repo's own run; UniRef sizes are rough
# 2025-era figures; replace with real counts (`grep -v '>' x.fasta | tr -d '\n' | wc -c`) when downloaded.
DATABASES = {"Swiss-Prot": 0.21e9, "UniRef50 (approx)": 17e9, "UniRef90 (approx)": 50e9}
# Bytes per residue = hidden size / 8 (one sign bit per dimension): 150M -> 80 B, 650M -> 160 B.


def read_fasta(path):
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                cur = []
            elif line:
                cur.append(line)
    if cur:
        seqs.append("".join(cur))
    return seqs


def chunks(seqs):
    """Split proteins longer than MAX_LEN into equal pieces, as 02_embed.py does."""
    out = []
    for s in seqs:
        n = -(-len(s) // MAX_LEN)
        cuts = np.linspace(0, len(s), n + 1).astype(int)
        out += [s[cuts[c]:cuts[c + 1]] for c in range(n)]
    return sorted(out, key=len, reverse=True)


def make_batches(pieces, tokens):
    batches, cur = [], []
    for p in pieces:                         # sorted longest first, so cur[0] is the longest
        if cur and (len(cur[0]) + 2) * (len(cur) + 1) > tokens:
            batches.append(cur)
            cur = []
        cur.append(p)
    batches.append(cur)
    return batches


def gpu_info():
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "nvidia-smi not found"


class FastTokenizer:
    """ESM tokenization by table lookup (one char -> one token), much faster than the Python HF tokenizer."""

    def __init__(self, tok):
        v = tok.get_vocab()
        self.cls, self.eos, self.pad = tok.cls_token_id, tok.eos_token_id, tok.pad_token_id
        self.lut = np.full(256, tok.unk_token_id, dtype=np.int64)
        for t, i in v.items():
            if len(t) == 1:
                self.lut[ord(t)] = i

    def __call__(self, seqs):
        L = max(map(len, seqs)) + 2
        ids = np.full((len(seqs), L), self.pad, dtype=np.int64)
        for k, s in enumerate(seqs):
            ids[k, 0] = self.cls
            ids[k, 1:len(s) + 1] = self.lut[np.frombuffer(s.encode(), dtype=np.uint8)]
            ids[k, len(s) + 1] = self.eos
        ids = torch.from_numpy(ids).pin_memory().cuda(non_blocking=True)
        return ids, ids != self.pad


def run(model, ftok, pieces, tokens, max_seconds, slow_copy=False):
    W = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], device="cuda", dtype=torch.uint8)
    D = model.config.hidden_size
    batches = make_batches(pieces, tokens)
    # Shuffle batch order so a time-limited run sees a mix of long and short proteins, not only the longest.
    order = np.random.default_rng(0).permutation(len(batches))
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for b in [batches[i] for i in order[:2]]:          # warm-up (kernel selection, allocator)
            ids, mask = ftok(b)
            model(input_ids=ids, attention_mask=mask)
        torch.cuda.synchronize()
        n_res, t0 = 0, time.time()
        for i in order:
            b = batches[i]
            ids, mask = ftok(b)
            rep = model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 1:-1]   # drop CLS / last slot
            if slow_copy:                                  # old way: one small GPU->CPU copy per protein
                for k, s in enumerate(b):
                    r = rep[k, :len(s)]
                    ((r >= 0).view(-1, D // 8, 8).to(torch.uint8) * W).sum(-1, dtype=torch.uint8).cpu()
            else:                                          # pack the whole batch on GPU, one copy, slice on CPU
                bits = ((rep >= 0).view(*rep.shape[:2], D // 8, 8).to(torch.uint8) * W).sum(-1, dtype=torch.uint8)
                bits = bits.cpu().numpy()
                _ = [bits[k, :len(s)] for k, s in enumerate(b)]
            n_res += sum(map(len, b))
            if time.time() - t0 > max_seconds:
                break
        torch.cuda.synchronize()
        secs = time.time() - t0
    return {"tokens_per_batch": tokens, "copy": "per-protein" if slow_copy else "per-batch", "residues": n_res,
            "seconds": round(secs, 2), "residues_per_s": round(n_res / secs),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--n", type=int, default=20000, help="random proteins to sample")
    ap.add_argument("--models", nargs="+", default=["esm2_t30_150M_UR50D"])
    ap.add_argument("--tokens", nargs="+", type=int, default=[12000, 24000, 48000, 96000])
    ap.add_argument("--max-seconds", type=float, default=120, help="time cap per (model, batch size)")
    ap.add_argument("--compare-old", action="store_true", help="also time the old per-protein copy once")
    ap.add_argument("--out", default="bench.json")
    a = ap.parse_args()

    seqs = read_fasta(a.fasta)
    rng = np.random.default_rng(1)
    sample = [seqs[i] for i in rng.choice(len(seqs), min(a.n, len(seqs)), replace=False)]
    pieces = chunks(sample)
    print(f"host {platform.node()} | job {os.environ.get('SLURM_JOB_ID', '-')} | "
          f"partition {os.environ.get('SLURM_JOB_PARTITION', '-')}")
    print(gpu_info())
    print(f"{len(sample):,} proteins, {sum(map(len, pieces)):,} residues sampled\n", flush=True)

    report = {"host": platform.node(), "gpu": gpu_info(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
              "partition": os.environ.get("SLURM_JOB_PARTITION"), "n_proteins": len(sample), "models": {}}
    for name in a.models:
        hf = f"facebook/{name}"
        ftok = FastTokenizer(AutoTokenizer.from_pretrained(hf))
        model = EsmModel.from_pretrained(hf, add_pooling_layer=False, attn_implementation="sdpa").eval().cuda().half()
        bytes_per_res = model.config.hidden_size // 8
        if a.compare_old:                                  # one run the old way, for a before/after number
            r = run(model, ftok, pieces, a.tokens[0], a.max_seconds, slow_copy=True)
            print(f"{name:24s} tokens {a.tokens[0]:>6}: {r['residues_per_s']:>9,} residues/s  (old per-protein copy)",
                  flush=True)
        results = []
        for t in a.tokens:
            try:
                r = run(model, ftok, pieces, t, a.max_seconds)
            except torch.cuda.OutOfMemoryError:
                print(f"{name:24s} tokens {t:>6}: out of GPU memory, stopping sweep", flush=True)
                torch.cuda.empty_cache()
                break
            results.append(r)
            print(f"{name:24s} tokens {t:>6}: {r['residues_per_s']:>9,} residues/s  "
                  f"peak mem {r['peak_gpu_mem_gb']:.1f} GB", flush=True)
        del model
        torch.cuda.empty_cache()
        if not results:
            continue
        best = max(results, key=lambda r: r["residues_per_s"])
        proj = {db: {"gpu_hours": round(n / best["residues_per_s"] / 3600, 2),
                     "disk_gb": round(n * bytes_per_res / 1e9)} for db, n in DATABASES.items()}
        report["models"][name] = {"sweep": results, "best": best, "projection_1gpu": proj}
        print(f"  best: {best['residues_per_s']:,} residues/s at {best['tokens_per_batch']} tokens/batch")
        for db, p in proj.items():
            print(f"  {db:20s} ~{p['gpu_hours']:>8.1f} GPU-hours on this GPU, ~{p['disk_gb']:,} GB of residue bits")
        print(flush=True)

    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
