"""Hybrid: what a practical tool would likely do — fast MMseqs2 search PLUS the embedding search,
candidates merged (MMseqs2 hits first), then the same alignment step as everyone else."""
import json

from common import DATA

HITS = DATA / "hits"
mm = json.loads((HITS / "mmseqs_cf.json").read_text())
em = json.loads((HITS / "emb_twostage.json").read_text())
hits = {}
for q in mm["hits"]:
    seen = dict.fromkeys(mm["hits"][q])
    seen.update(dict.fromkeys(em["hits"].get(q, [])))
    hits[q] = list(seen)
spq = mm["seconds_per_query"] + em["seconds_per_query"]
(HITS / "hybrid_mmseqs_emb.json").write_text(json.dumps(
    {"hits": hits, "seconds_per_query": spq, "note": f"MMseqs2 + embedding candidates merged ({spq:.1f} s/query)"}))
print(f"hybrid: mean {sum(map(len, hits.values())) / len(hits):.0f} candidates/query")
