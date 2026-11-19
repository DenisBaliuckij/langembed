"""Stage 4 (host, no ML deps): splice a built article_graphs_data.json into
the existing semantic_graph.html D3 viewer template, replacing only the
ARTICLE_DATA JS blob. CORPUS_DATA (the whole-corpus per-document graph,
built separately) is left untouched.

Kept as a plain script rather than a DAG task since it's pure text assembly,
not a "calculation" over the pipeline's data -- but runs on the same host as
the rest of the pipeline so no intermediate artifact needs to leave the
remote machine except the final HTML.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def splice_article_graph_html(template_path: Path, data_path: Path, out_path: Path) -> None:
    html = template_path.read_text(encoding="utf-8")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    new_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    marker = "const ARTICLE_DATA = "
    start = html.index(marker) + len(marker)
    end = html.index(";\n\n(function", start)

    spliced = html[:start] + new_json + html[end:]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(spliced, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--data", default="output/en/article_graphs_data.json")
    ap.add_argument("--out", default="output/en/semantic_graph.html")
    args = ap.parse_args()

    out_path = Path(args.out)
    splice_article_graph_html(Path(args.template), Path(args.data), out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
