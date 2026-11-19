#!/bin/bash
# v2 per-article semantic graphs (2026-10-11), full regeneration from the PDFs.
# Usage: scripts/run_article_graphs_v2.sh <run-dir>
#   <run-dir>/pdfs/*.pdf              input articles
#   <run-dir>/titles.json             optional {"<pdf stem>": "<display title>"}
#   <run-dir>/models/all-mpnet-base-v2  sentence-transformers model (copied from the HF cache)
# Writes <run-dir>/html/semantic_graph_<stem>.{html,json}. Compute steps run in langembed-ml
# on CPU via docker_run_watchdog.sh; text extraction, LLM labels and rendering run on the host.
set -euo pipefail
cd /home/s939/langembed_deploy/langembed
D="$1"
MODEL="${MODEL:-$D/models/all-mpnet-base-v2}"
LEXICON="$D/english_words.txt"

rm -rf "$D/text" "$D/out" "$D/html" "$D"/article_*.jsonl "$D/article_manifest.json"
mkdir -p "$D/text" "$D/html"
cp /usr/share/dict/american-english "$LEXICON"
for f in "$D"/pdfs/*.pdf; do
  pdftotext -enc UTF-8 "$f" "$D/text/$(basename "$f" .pdf).txt"
done

bash scripts/docker_run_watchdog.sh graphs-v2-prep 30 false scripts/prepare_article_units_v2.py \
  --text-dir "$D/text" --out-dir "$D" --lexicon "$LEXICON"
bash scripts/docker_run_watchdog.sh graphs-v2-embed 60 false scripts/embed_article_units_cpu.py \
  --in "$D/article_units_v2.jsonl" --out "$D/article_unit_embeddings_v2.jsonl" --model-dir "$MODEL"
bash scripts/docker_run_watchdog.sh graphs-v2-build 30 false scripts/build_article_graphs_v2.py \
  --in "$D/article_unit_embeddings_v2.jsonl" --manifest "$D/article_manifest.json" --out-dir "$D/out" \
  --lexicon "$LEXICON" --embedding-model "$(basename "$MODEL")"
TITLES=()
[ -f "$D/titles.json" ] && TITLES=(--titles "$D/titles.json")
python3 scripts/label_clusters_llm.py --out-dir "$D/out" "${TITLES[@]}"

for json in "$D"/out/*.json; do
  stem="$(basename "$json" .json)"
  [ "$stem" = "article_graphs_v2" ] && continue
  safe="$(echo "$stem" | tr ' ' '_' | tr -cd 'A-Za-z0-9_.()-')"
  python3 scripts/render_article_graph_viewer.py --template templates/article_graph_viewer.html \
    --data "$json" --out "$D/html/semantic_graph_${safe}.html"
  cp "$json" "$D/html/semantic_graph_${safe}.json"
done
ls -la "$D/html"
