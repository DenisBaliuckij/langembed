"""Stage 3: build a per-article k-NN similarity graph over phrase-level
semantic units (not whole sentences) -- nodes are units, edges connect each
unit to its nearest neighbors by embedding cosine similarity, clustered via
greedy modularity communities. See langembed.data.semantic_graph_lib for the
shared graph/clustering logic (also used by the whole-corpus per-document
graph, unaffected by this change).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from langembed.data.semantic_graph_lib import build_graph, make_label, pick_title


def build_article_graphs(in_path: Path, out_path: Path, k: int, min_cluster_size: int) -> dict:
    records = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    by_article = defaultdict(list)
    for r in records:
        by_article[r["article_id"]].append(r)

    articles_out = []
    for article_id, recs in by_article.items():
        recs = sorted(recs, key=lambda r: (r["sent_idx"], r["unit_idx"]))
        texts = [r["text"] for r in recs]
        edges, degree, cluster_of, cluster_labels, n_raw_communities = build_graph(
            recs, texts, k, min_cluster_size
        )

        nodes_out = []
        for i, r in enumerate(recs):
            text = r["text"]
            nodes_out.append(
                {
                    "id": i,
                    "label": make_label(text),
                    "text": text,
                    "parent_sentence": r["parent_sentence"],
                    "cluster": cluster_of.get(i, -1),
                    "degree": degree.get(i, 0),
                    "meta": f"sentence {r['sent_idx'] + 1}, unit {r['unit_idx'] + 1}",
                }
            )

        edges_out = [
            {"source": i, "target": j, "weight": round(w, 4)} for (i, j), w in edges.items()
        ]

        # best-effort title for the dropdown label: skip PDF->LaTeX running-header
        # lines (journal masthead, "PREPRINT VERSION", volume/page) that sometimes
        # precede the real title in the extracted text -- built from parent
        # sentences (title/author lines are one or two full sentences), not units
        parent_sentences = []
        seen_sent = set()
        for r in recs:
            if r["sent_idx"] not in seen_sent:
                seen_sent.add(r["sent_idx"])
                parent_sentences.append(r["parent_sentence"])
            if len(parent_sentences) >= 2:
                break
        title = pick_title(parent_sentences, n=90)

        articles_out.append(
            {
                "article_id": article_id,
                "title": title,
                "nodes": nodes_out,
                "edges": edges_out,
                "clusters": [
                    {
                        "id": cid,
                        "label": cluster_labels[cid],
                        "size": sum(1 for n in nodes_out if n["cluster"] == cid),
                    }
                    for cid in sorted(cluster_labels)
                ],
                "meta": {
                    "granularity": "semantic-unit (phrase-level, single article)",
                    "branch": "A (contrastive finetune, artifacts/simcse_en)",
                    "embedding_dim": len(recs[0]["embedding"]),
                    "k": k,
                    "num_nodes": len(recs),
                    "num_edges": len(edges_out),
                    "num_clusters": len(cluster_labels),
                    "num_raw_communities": n_raw_communities,
                },
            }
        )

    data = {"articles": articles_out}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    for a in articles_out:
        print(
            a["article_id"],
            "-",
            a["meta"]["num_nodes"],
            "nodes,",
            a["meta"]["num_clusters"],
            "clusters",
        )

    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="output/en/article_unit_embeddings.jsonl")
    ap.add_argument("--out", default="output/en/article_graphs_data.json")
    ap.add_argument("--k", type=int, default=5, help="Neighbors per node.")
    ap.add_argument("--min-cluster-size", type=int, default=8)
    args = ap.parse_args()

    build_article_graphs(Path(args.in_path), Path(args.out), args.k, args.min_cluster_size)


if __name__ == "__main__":
    main()
