"""Test for scripts/splice_article_graph_html.py's core function: verifies
only the ARTICLE_DATA blob is replaced, CORPUS_DATA is left untouched."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from splice_article_graph_html import splice_article_graph_html  # noqa: E402

_TEMPLATE = (
    "<html><script>\n"
    'const CORPUS_DATA = {"nodes": ["keep me"]};\n'
    'const ARTICLE_DATA = {"articles": ["old"]};\n\n'
    "(function () { console.log('viewer'); })();\n"
    "</script></html>"
)


def test_splice_replaces_only_article_data(tmp_path: Path) -> None:
    template_path = tmp_path / "template.html"
    data_path = tmp_path / "new_data.json"
    out_path = tmp_path / "out.html"

    template_path.write_text(_TEMPLATE, encoding="utf-8")
    new_data = {"articles": [{"article_id": "a1", "nodes": []}]}
    data_path.write_text(json.dumps(new_data), encoding="utf-8")

    splice_article_graph_html(template_path, data_path, out_path)
    result = out_path.read_text(encoding="utf-8")

    assert '"nodes": ["keep me"]' in result
    assert '"old"' not in result
    assert '"article_id":"a1"' in result
    assert "(function () { console.log('viewer'); })();" in result
