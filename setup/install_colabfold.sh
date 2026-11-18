#!/usr/bin/env bash
# ColabFold (AlphaFold2) in its own conda env 'cf', plus US-align for TM-scores. No sudo.
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
export TMPDIR="$HOME/.piptmp"; mkdir -p "$TMPDIR"
PIP="pip install --quiet --timeout 120 --retries 10"

if ! conda env list | grep -q '^cf '; then
  conda create -y -q -n cf -c conda-forge -c bioconda python=3.11 openmm=8 pdbfixer usalign kalign2 hhsuite
fi
conda activate cf
$PIP "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"
$PIP "jax[cuda12]==0.5.3" "dm-haiku==0.0.14"
$PIP tensorflow-cpu 2>/dev/null || true      # only used for a few data utilities
rm -rf "$TMPDIR"

echo "== weights"
python -m colabfold.download
echo "== check"
python - <<'PY'
import jax
print("jax", jax.__version__, jax.devices())
import colabfold, alphafold
print("colabfold ok")
PY
which colabfold_batch USalign
echo "== CF DONE"
