#!/usr/bin/env bash
# Real experimental structures of the query domains, from SCOPe's official bulk download.
set -euo pipefail
D="$HOME/folding-data/poc"
mkdir -p "$D/download" "$D/fold/ref"
cd "$D/download"
TGZ=pdbstyle-sel-gs-bib-40-2.08.tgz
[ -s "$TGZ" ] || curl -fsSL -C - --retry 5 -o "$TGZ" "https://scop.berkeley.edu/downloads/pdbstyle/$TGZ"
cut -f1 "$D/queries.tsv" | tail -n +2 | sed 's/$/.ent/' > want.txt
tar -tzf "$TGZ" | grep -F -f want.txt > paths.txt || true
tar -xzf "$TGZ" -C "$D/fold/ref" --strip-components=3 -T paths.txt
echo "$(ls "$D/fold/ref"/*.ent | wc -l) of $(wc -l < want.txt) reference structures extracted"
