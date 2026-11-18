#!/usr/bin/env bash
# Installs the Phase 1 toolchain into the user's home directory (no sudo needed).
set -euo pipefail

MF="$HOME/miniforge3"
if [ ! -x "$MF/bin/conda" ]; then
  echo "== Downloading Miniforge"
  curl -fsSL -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge.sh -b -p "$MF"
  rm /tmp/miniforge.sh
fi
source "$MF/etc/profile.d/conda.sh"

if ! conda env list | grep -q '^fold '; then
  echo "== Creating conda env 'fold' (python, mmseqs2, faiss, numpy...)"
  conda create -y -n fold -c conda-forge -c bioconda \
    python=3.12 mmseqs2 faiss-cpu numpy pandas scikit-learn biopython matplotlib tqdm
fi
conda activate fold

echo "== Installing PyTorch (CUDA) and ESM"
# Ubuntu's /tmp is a 2 GB RAM disk; the CUDA wheels don't fit. Use the real disk.
export TMPDIR="$HOME/.piptmp"
mkdir -p "$TMPDIR"
PIP_OPTS="--quiet --timeout 120 --retries 10 --resume-retries 30"
pip install $PIP_OPTS torch
pip install $PIP_OPTS fair-esm
# Meta's fair-esm download server has an expired TLS cert; load ESM-2 weights from Hugging Face instead.
pip install $PIP_OPTS transformers safetensors
rm -rf "$TMPDIR"

echo "== Verifying"
python - <<'PY'
import torch, esm, faiss, numpy
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
import transformers
print("faiss", faiss.__version__, "| numpy", numpy.__version__, "| transformers", transformers.__version__)
PY
mmseqs version
echo "== DONE"
