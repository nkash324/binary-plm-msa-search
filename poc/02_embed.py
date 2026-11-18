"""Embed Swiss-Prot + queries with ESM-2 150M and store ONLY binary codes (as a real index would).

Per residue : 640 sign bits of (embedding - mean)          -> res_bits.npy   (80 B / residue)
Per protein : mean embedding (fp16, for reference) + bits   -> prot_mean.npy, prot_bits.npy
Proteins longer than 1022 are embedded in equal chunks and concatenated.
Resumable: re-running skips finished batches.
"""
import json
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

from common import GPU_REST, DATA, DB_FASTA, EMB_DIM, ESM_MODEL, MAX_LEN, Q_FASTA, read_fasta

D = EMB_DIM
TOKENS_PER_BATCH = 12000      # defines the (resumable) batch list; don't change mid-run
SUB_TOKENS = int(os.environ.get("POC_SUB_TOKENS", 3000))   # actual GPU burst size
PAUSE_FILE = DATA / "PAUSE"   # touch to pause embedding, delete to resume
REST_FILE = DATA / "gpu_rest" # e.g. "1.0" = GPU idle as long as it works (~50% busy)
HF_NAME = f"facebook/{ESM_MODEL}"
tok = AutoTokenizer.from_pretrained(HF_NAME)
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("POC_GPU_MEM", 0.35)))   # 0.35: leave ~5 GB for other programs
model = EsmModel.from_pretrained(HF_NAME, add_pooling_layer=False, attn_implementation="sdpa").eval().cuda().half()
W = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], device="cuda", dtype=torch.uint8)


def forward(seqs):
    enc = tok(seqs, padding=True, return_tensors="pt")
    ids, mask = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
    rep = model(input_ids=ids, attention_mask=mask).last_hidden_state
    if not torch.isfinite(rep).all():              # rare fp16 overflow -> fp32
        model.float()
        rep = model(input_ids=ids, attention_mask=mask).last_hidden_state
        model.half()
    return rep.float()


def pack(x):                                        # (L, 640) bool -> (L, 80) uint8, np.packbits order
    return (x.view(-1, D // 8, 8).to(torch.uint8) * W).sum(-1, dtype=torch.uint8)


def make_batches(lens):
    items = []                                      # (seq index, start, end)
    for i, L in enumerate(lens):
        n = -(-L // MAX_LEN)
        cuts = np.linspace(0, L, n + 1).astype(int)
        items += [(i, cuts[c], cuts[c + 1]) for c in range(n)]
    items.sort(key=lambda t: -(t[2] - t[1]))
    batches, cur, cur_max = [], [], 0
    for it in items:
        L = it[2] - it[1] + 2
        if cur and max(cur_max, L) * (len(cur) + 1) > TOKENS_PER_BATCH:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(it)
        cur_max = max(cur_max, L)
    batches.append(cur)
    return batches


# ---------------------------------------------------------------- centering means (from a sample)
db_ids, db_seqs = read_fasta(DB_FASTA)
mu_path = DATA / "mu.npz"
if not mu_path.exists():
    rng = np.random.default_rng(0)
    sample = [db_seqs[i][:MAX_LEN] for i in rng.choice(len(db_seqs), 3000, replace=False)]
    rs, ps = torch.zeros(D, device="cuda"), torch.zeros(D, device="cuda")
    n_res = 0
    with torch.inference_mode():
        for s in range(0, len(sample), 16):
            chunk = sample[s:s + 16]
            rep = forward(chunk)
            for k, q in enumerate(chunk):
                r = rep[k, 1:len(q) + 1]
                rs += r.sum(0)
                ps += r.mean(0)
                n_res += len(q)
    np.savez(mu_path, res=(rs / n_res).cpu().numpy(), prot=(ps / len(sample)).cpu().numpy())
    print("centering means computed from 3000 random proteins")
mu = np.load(mu_path)
mu_res = torch.from_numpy(mu["res"]).cuda()
mu_prot = mu["prot"]


def embed_set(prefix, seqs):
    lens = np.array([len(s) for s in seqs])
    N = len(seqs)
    offsets = np.zeros(N + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lens)
    np.save(DATA / f"{prefix}offsets.npy", offsets)
    batches = make_batches(lens)
    multi = {i: j for j, i in enumerate(np.flatnonzero(lens > MAX_LEN))}
    n_multi_chunks = int(sum(-(-lens[i] // MAX_LEN) for i in multi))
    prog_path = DATA / f"{prefix}progress.json"
    done = json.loads(prog_path.read_text())["batches"] if prog_path.exists() else 0
    mode = "r+" if done else "w+"
    res = np.lib.format.open_memmap(DATA / f"{prefix}res_bits.npy", mode=mode, dtype=np.uint8,
                                    shape=(int(offsets[-1]), D // 8))
    pmean = np.lib.format.open_memmap(DATA / f"{prefix}prot_mean.npy", mode=mode, dtype=np.float16,
                                      shape=(N, D))
    csum = np.lib.format.open_memmap(DATA / f"{prefix}chunk_sums.npy", mode=mode, dtype=np.float32,
                                     shape=(max(n_multi_chunks, 1), D))
    chunk_row, r_ = {}, 0                           # (seq, start) -> row in csum, for multi-chunk proteins
    for i in multi:
        cuts = np.linspace(0, lens[i], -(-lens[i] // MAX_LEN) + 1).astype(int)
        for c in cuts[:-1]:
            chunk_row[(i, c)] = r_
            r_ += 1
    print(f"{prefix or 'db '}: {N:,} proteins, {offsets[-1]:,} residues, {len(batches)} batches "
          f"({done} already done), {len(multi)} proteins >1022 chunked", flush=True)

    t0 = time.time()
    rest = GPU_REST
    with torch.inference_mode():
        for b in range(done, len(batches)):
            if b % 20 == 0:                         # live controls: edit/create these files while running
                while PAUSE_FILE.exists():
                    time.sleep(5)
                try:
                    rest = float(REST_FILE.read_text())
                except (OSError, ValueError):
                    rest = GPU_REST
            # Short GPU bursts (sub-batches of <= SUB_TOKENS) with rests in between keep the desktop responsive.
            subs, cur = [], []
            for it in batches[b]:
                if cur and (cur[0][2] - cur[0][1] + 2) * (len(cur) + 1) > SUB_TOKENS:
                    subs.append(cur)
                    cur = []
                cur.append(it)
            subs.append(cur)
            for items in subs:
                tb = time.time()
                rep = forward([seqs[i][s:e] for i, s, e in items])
                for k, (i, s, e) in enumerate(items):
                    r = rep[k, 1:e - s + 1]
                    res[offsets[i] + s:offsets[i] + e] = pack(r - mu_res >= 0).cpu().numpy()
                    if i in multi:
                        csum[chunk_row[(i, s)]] = r.sum(0).cpu().numpy()
                    else:
                        pmean[i] = r.mean(0).cpu().numpy()
                torch.cuda.synchronize()
                time.sleep(rest * (time.time() - tb))
            if (b + 1) % 500 == 0 or b + 1 == len(batches):
                res.flush(); pmean.flush(); csum.flush()
                prog_path.write_text(json.dumps({"batches": b + 1}))
                el = time.time() - t0
                print(f"  batch {b + 1}/{len(batches)}  {el / 60:.1f} min  "
                      f"(~{el / (b + 1 - done) * (len(batches) - b - 1) / 60:.0f} min left)", flush=True)
    for i in multi:                                 # finish means of chunked proteins
        cuts = np.linspace(0, lens[i], -(-lens[i] // MAX_LEN) + 1).astype(int)
        pmean[i] = sum(csum[chunk_row[(i, c)]] for c in cuts[:-1]) / lens[i]
    pmean.flush()
    bits = np.concatenate([np.packbits(np.asarray(pmean[s:s + 50_000], np.float32) - mu_prot >= 0, axis=1)
                           for s in range(0, N, 50_000)])
    np.save(DATA / f"{prefix}prot_bits.npy", bits)
    print(f"  done in {(time.time() - t0) / 60:.1f} min", flush=True)


q_ids, q_seqs = read_fasta(Q_FASTA)
embed_set("q_", q_seqs)
embed_set("", db_seqs)
