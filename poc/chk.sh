source ~/miniforge3/etc/profile.d/conda.sh; conda activate plmblast
pgrep -fa "pip install" | head -2
python -c "import torch,numba,transformers;print('env', torch.__version__, torch.cuda.is_available(), numba.__version__, transformers.__version__)" 2>&1 | tail -2
cd ~/tools/plmblast && python scripts/plmblast.py -h 2>&1 | head -60
