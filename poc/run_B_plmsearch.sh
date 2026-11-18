#!/usr/bin/env bash
# Test B, PLMSearch: wait for the 13 GB archive, extract ONLY the trained similarity layer, verify it is plain weights, score.
source "$HOME/miniforge3/etc/profile.d/conda.sh"; export PYTHONWARNINGS=ignore
cd /mnt/c/Folding/poc; L=results/overnight.log; T=/mnt/f/tools/plmsearch
until [ -f $T/plmsearch_data.tar.gz ] && [ "$(stat -c %s $T/plmsearch_data.tar.gz)" -ge 12995086677 ]; do sleep 60; done
echo "$(date +%H:%M) B: PLMSearch archive downloaded" >> $L
mkdir -p ~/tools/plmsearch_model && cd ~/tools/plmsearch_model
tar -xzf $T/plmsearch_data.tar.gz --wildcards '*/model/plmsearch.sav' 2>/tmp/tar.err || tar -tzf $T/plmsearch_data.tar.gz | grep -i "\.sav$" > /tmp/sav_list
SAV=$(find ~/tools/plmsearch_model -name "plmsearch.sav" | head -1)
cd /mnt/c/Folding/poc; conda activate fold
python - "$SAV" >> $L 2>&1 <<'PY'
import sys, zipfile, pickletools, io
z = zipfile.ZipFile(sys.argv[1]); pk = [n for n in z.namelist() if n.endswith("data.pkl")][0]
refs = {a for op, a, _ in pickletools.genops(io.BytesIO(z.read(pk))) if op.name == "GLOBAL"}
print("B: plmsearch.sav references:", sorted(refs))
PY
flock /tmp/gpu.lock python 43_B_plmsearch.py "$SAV" 2>&1 | grep -E "saved|DONE|Error" | sed "s/^/$(date +%H:%M) B: /" >> $L
