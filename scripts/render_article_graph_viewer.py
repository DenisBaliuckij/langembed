"""Stage 4 of the v2 per-article semantic graph (host, no ML deps): put one article's graph
(build_article_graphs_v2.py output, {"articles": [one]}) into the single-article viewer
template (templates/article_graph_viewer.html: search box, frequency-sized nodes, topic
legend with top phrases, term detail with spellings and contexts)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MARKER = "/*ARTICLE_DATA*/null"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="templates/article_graph_viewer.html")
    ap.add_argument("--data", required=True, help='{"articles": [one article]} JSON')
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="", help="display title (default: the article id)")
    args = ap.parse_args()

    html = Path(args.template).read_text(encoding="utf-8")
    if html.count(MARKER) != 1:
        raise SystemExit(f"template must contain {MARKER!r} exactly once")
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.title:
        data["articles"][0]["title"] = args.title
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace(MARKER, blob), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
