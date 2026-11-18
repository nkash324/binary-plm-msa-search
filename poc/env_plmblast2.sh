source ~/miniforge3/etc/profile.d/conda.sh; conda activate plmblast
cd ~/tools/plmblast && pip install -q -r requirements.txt > /tmp/plmblast_pip.log 2>&1; echo "pip exit $?"
python -c "import torch,numba,transformers;print('env ok', torch.__version__, torch.cuda.is_available(), numba.__version__, transformers.__version__)" 2>&1 | tail -1
python scripts/plmblast.py -h 2>&1 | tail -60
