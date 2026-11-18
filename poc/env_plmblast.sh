source ~/miniforge3/etc/profile.d/conda.sh
conda create -y -q -n plmblast python=3.10 > /tmp/plmblast_env.log 2>&1
conda activate plmblast
cd ~/tools/plmblast && pip install -q -r requirements.txt >> /tmp/plmblast_env.log 2>&1
python -c "import torch, numba, transformers; print('plmblast env ok', torch.__version__, torch.cuda.is_available(), numba.__version__, transformers.__version__)" 2>&1 | tail -1
