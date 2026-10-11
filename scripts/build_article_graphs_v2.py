"""Stage 3 of the v2 per-article semantic graph (2026-10-11): embedded merged units ->
one graph per article (see langembed.data.article_graph_v2). Writes the combined
{"articles": [...]} file plus one <article_id>.json per article for the viewer."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from langembed.data.article_graph_v2 import build_article_graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="embed_article_units_cpu.py output")
    ap.add_argument("--manifest", required=True, help="article_manifest.json (titles / sources)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=1.0, help="embedding share of the edge weight")
    ap.add_argument(
        "--core-min-sim",
        type=float,
        default=-1.0,
        help="min cosine to the cluster centroid; looser nodes go to 'Other (weakly related)'",
    )
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--max-nodes", type=int, default=400)
    ap.add_argument("--min-cluster-size", type=int, default=6)
    ap.add_argument("--max-clusters", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lexicon", default="", help="English word list (drops rare unknown words)")
    ap.add_argument("--embedding-model", default="", help="name recorded in the graph meta")
    args = ap.parse_args()

    by_article: dict[str, list[dict]] = defaultdict(list)
    with open(args.in_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_article[r["article_id"]].append(r)

    lexicon: frozenset[str] = frozenset()
    if args.lexicon:
        lexicon = frozenset(
            w.strip().lower()
            for w in Path(args.lexicon).read_text(encoding="utf-8").splitlines()
            if w.strip() and "'" not in w
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    articles = []
    for article_id, units in by_article.items():
        graph = build_article_graph(
            units,
            lexicon=lexicon,
            embedding_model=args.embedding_model,
            k=args.k,
            alpha=args.alpha,
            core_min_sim=args.core_min_sim,
            min_count=args.min_count,
            max_nodes=args.max_nodes,
            min_cluster_size=args.min_cluster_size,
            max_clusters=args.max_clusters,
            seed=args.seed,
        )
        record = {"article_id": article_id, "title": article_id, **graph}
        articles.append(record)
        (out_dir / f"{article_id}.json").write_text(
            json.dumps({"articles": [record]}, ensure_ascii=False), encoding="utf-8"
        )
        m = graph["meta"]
        print(
            f"{article_id}: {m['num_units_before_selection']} units -> {m['num_nodes']} nodes, "
            f"{m['num_edges']} edges, {m['num_clusters']} clusters "
            f"(resolution {m['louvain_resolution']}), "
            f"{m['num_other']} weakly related"
        )
        for c in graph["clusters"]:
            print(f"   [{c['size']:3d}] {c['label']}  — {', '.join(c['keywords'])}")
    (out_dir / "article_graphs_v2.json").write_text(
        json.dumps({"articles": articles}, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
