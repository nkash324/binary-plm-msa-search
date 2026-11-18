#!/usr/bin/env bash
# Small ESM Atlas test: one embedding shard + ESM-2 3B weights.
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate fold
A=$HOME/folding-data/atlas; mkdir -p "$A"
pip install -q accelerate > /dev/null 2>&1 && echo "accelerate installed"
( curl -sS -C - --retry 5 -o "$A/shard.npz" \
   "https://dl.fbaipublicfiles.com/esmatlas/v2023_02/full/lm_reps/tm_.80_.90_plddt_.80_.90_00.npz" \
   && echo "shard downloaded: $(du -h "$A/shard.npz" | cut -f1)" ) &
python - <<'PY'
from huggingface_hub import snapshot_download, list_repo_files
files = list_repo_files("facebook/esm2_t36_3B_UR50D")
pat = ["*.json", "*.txt"] + (["*.safetensors"] if any(f.endswith(".safetensors") for f in files) else ["*.bin"])
p = snapshot_download("facebook/esm2_t36_3B_UR50D", allow_patterns=pat)
print("3B weights at", p)
PY
wait
echo "DOWNLOADS DONE"
