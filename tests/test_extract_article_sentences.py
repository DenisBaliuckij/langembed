"""Integration test for scripts/extract_article_sentences.py's core function:
sampling, sentence extraction, and the manifest it writes for downstream
copy-back steps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from extract_article_sentences import extract_article_sentences  # noqa: E402

_FAKE_TEX = r"""
\documentclass{article}
\begin{document}
This is the first real sentence of the fake paper, long enough to pass the length filter.
Here is a second sentence that also clears the minimum character threshold easily.
\end{document}
"""


def test_extract_article_sentences_writes_records_and_manifest(tmp_path: Path) -> None:
    tex_dir = tmp_path / "tex"
    tex_dir.mkdir()
    for i in range(3):
        (tex_dir / f"article-{i}.tex").write_text(_FAKE_TEX, encoding="utf-8")

    out_path = tmp_path / "sentences.jsonl"
    manifest_path = tmp_path / "manifest.json"

    n = extract_article_sentences(
        tex_dir=tex_dir,
        out_path=out_path,
        manifest_path=manifest_path,
        n_articles=2,
        seed=42,
        max_sentences_per_article=120,
        min_sentence_chars=25,
    )

    assert n > 0
    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    article_ids = {r["article_id"] for r in records}
    assert len(article_ids) == 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["seed"] == 42
    assert manifest["n_articles"] == 2
    assert set(manifest["article_ids"]) == article_ids
    assert len(manifest["tex_paths"]) == 2
