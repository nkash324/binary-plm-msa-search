#!/bin/bash
# ESM-2 embedding speed test. Submit from the repo folder:
#   sbatch cluster/bench.sbatch                                   # ic-express (H100 slice, short wait)
#   sbatch -p IllinoisComputes-GPU cluster/bench.sbatch           # full A100/H200 GPU (may queue)
#SBATCH --job-name=binplm-bench
#SBATCH --account=noguera2-ic
#SBATCH --partition=ic-express
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

set -euo pipefail
ENV_DIR="${ENV_DIR:-$HOME/envs/binplm}"
WORK="${WORK:-/scratch/$USER/binplm}"
export HF_HOME="$WORK/hf" HF_HUB_OFFLINE=1      # weights were cached by setup_cluster.sh
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

mkdir -p "$WORK/bench"
nvidia-smi
python cluster/bench_embed.py \
  --fasta "$WORK/sprot.fasta" --n 20000 \
  --models esm2_t30_150M_UR50D esm2_t33_650M_UR50D \
  --max-seconds 120 --compare-old \
  --out "$WORK/bench/bench_${SLURM_JOB_PARTITION}_${SLURM_JOB_ID}.json"
