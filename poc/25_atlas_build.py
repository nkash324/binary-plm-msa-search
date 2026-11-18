"""Build the 20-shard Atlas binary index (32 bytes per protein).

The bit recipe (mean + 256 PCA directions) is fitted ONCE on shard 00 exactly as in 22_atlas_msa.py and reused for
every shard, so all codes are comparable and shard 00's codes are identical to the 1-shard test.
Each shard: wait for the Windows download (F:\\atlas_shards\\<name>.ok), copy it into WSL, project 50k proteins at a
time -> bits, save codes + ids, delete the copy (the raw file stays on F:).
Run in the 'fold' env.  Output: ~/folding-data/atlas/idx20/
"""
import os
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np

from common import RESULTS

A = Path.home() / "folding-data" / "atlas"
OUT = A / "idx20"
OUT.mkdir(exist_ok=True)
F = Path("/mnt/f/atlas_shards")
LOG = open(RESULTS / "atlas20.log", "a")


def log(m):
    m = f"{time.strftime('%H:%M:%S')} {m}"
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


# ---------------------------------------------------------------- bit recipe (same as 22_atlas_msa.py)
V = np.load(A / "shard_vecs.npy", mmap_mode="r")
if not (A / "pca.npz").exists():
    rng = np.random.default_rng(0)
    samp = np.asarray(V[np.sort(rng.choice(len(V), 50_000, replace=False))], np.float32)
    mu = samp.mean(0)
    w, v = np.linalg.eigh(np.cov((samp - mu).T))
    np.savez(A / "pca.npz", mu=mu, P=v[:, np.argsort(-w)[:256]].astype(np.float32))
pca = np.load(A / "pca.npz")
mu, P = pca["mu"], pca["P"]


def bits(X):
    return np.packbits((np.asarray(X, np.float32) - mu) @ P > 0, axis=1)


names = [u.rsplit("/", 1)[1][:-4] for u in (RESULTS / os.environ.get("ATLAS_LIST", "atlas20_shards.txt")).read_text().split()]
def forget(path):
    """Ask Linux to drop this file from its page cache (keeps WSL's memory footprint small)."""
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except OSError:
        pass


for n in names:
    if (OUT / f"{n}.codes.npy").exists() and (OUT / f"{n}.ids.txt").exists():
        continue
    t0 = time.time()
    if n.endswith("plddt_.80_.90_00"):                              # the shard we already unpacked
        codes = np.concatenate([bits(V[a:a + 50_000]) for a in range(0, len(V), 50_000)])
        ids = (A / "shard_ids.txt").read_text().split()
    else:
        while not (F / f"{n}.npz.ok").exists():
            time.sleep(15)
        tmp = A / "tmp_shard.npz"
        shutil.copyfile(F / f"{n}.npz", tmp)                       # one sequential read from the HDD
        forget(F / f"{n}.npz")                                     # don't keep the raw file in Linux's cache
        z = zipfile.ZipFile(tmp)
        members = z.namelist()
        codes, ids, bad = [], [], []
        for a in range(0, len(members), 50_000):
            X, keep = [], []
            for m in members[a:a + 50_000]:
                v = np.lib.format.read_array(z.open(m))
                if v.shape == (2560,):
                    X.append(v)
                    keep.append(Path(m).stem)
                else:
                    bad.append((Path(m).stem, v.shape))
            codes.append(bits(np.stack(X)))
            ids += keep
        codes = np.concatenate(codes)
        if bad:
            log(f"  {n}: skipped {len(bad)} malformed entries, e.g. {bad[:3]}")
        z.close()
        forget(tmp)
        tmp.unlink()
    np.save(OUT / f"{n}.codes.npy", codes)
    (OUT / f"{n}.ids.txt").write_text("\n".join(ids))
    log(f"index: {n} {len(ids):,} proteins in {time.time() - t0:.0f}s")
tot = sum(len((OUT / f"{n}.ids.txt").read_text().split()) for n in names)
log(f"INDEX BUILT: {len(names)} shards, {tot:,} proteins, {tot * 32 / 1e6:.0f} MB")
