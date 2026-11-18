"""Ask Linux to drop its cached copies of the big code files (keeps WSL from holding GBs of RAM that
Windows needs). Only drops clean cache pages; never touches the data. No admin rights needed."""
import os
from pathlib import Path

home = Path.home() / "folding-data"
for f in list((home / "poc").glob("*.npy")) + list((home / "phase1").glob("*.npy")):
    if f.stat().st_size > 50_000_000:
        fd = os.open(f, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
