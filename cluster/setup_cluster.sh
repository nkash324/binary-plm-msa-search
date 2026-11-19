#!/usr/bin/env bash
# One-time setup on the Illinois Campus Cluster. Run on a LOGIN node (needs internet; light work only).
#   bash cluster/setup_cluster.sh
# Installs Miniforge + a 'binplm' env, downloads Swiss-Prot and the ESM-2 weights into scratch.
# Compute nodes then run fully offline.
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/envs/binplm}"          # ~6 GB (PyTorch+CUDA). Set ENV_DIR=/scratch/... if home is tight.
WORK="${WORK:-/scratch/$USER/binplm}"           # data + model cache (scratch: big, but not backed up)
MODELS="${MODELS:-esm2_t30_150M_UR50D esm2_t33_650M_UR50D}"
export HF_HOME="$WORK/hf"
export PIP_CACHE_DIR="$WORK/pip-cache"           # keep pip's cache out of the home quota
mkdir -p "$WORK" "$HF_HOME"

MF="$HOME/miniforge3"
if [ ! -x "$MF/bin/conda" ]; then
  echo "== Installing Miniforge"
  curl -fsSL -o "$WORK/miniforge.sh" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "$WORK/miniforge.sh" -b -p "$MF"
  rm "$WORK/miniforge.sh"
fi
source "$MF/etc/profile.d/conda.sh"

if [ ! -d "$ENV_DIR" ]; then
  echo "== Creating env at $ENV_DIR"
  conda create -y -q -p "$ENV_DIR" -c conda-forge python=3.12 numpy
fi
conda activate "$ENV_DIR"
pip install --quiet torch transformers safetensors

if [ ! -s "$WORK/sprot.fasta" ]; then
  echo "== Downloading Swiss-Prot (~90 MB gz)"
  curl -fsSL "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz" \
    | gunzip > "$WORK/sprot.fasta"
fi

echo "== Caching model weights"
for m in $MODELS; do
  python -c "from transformers import AutoTokenizer, EsmModel as M; AutoTokenizer.from_pretrained('facebook/$m'); M.from_pretrained('facebook/$m')"
done

python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"
echo "proteins in Swiss-Prot: $(grep -c '>' "$WORK/sprot.fasta")"
du -sh "$ENV_DIR" "$WORK"
echo "== SETUP DONE. Next: sbatch cluster/bench.sbatch"
