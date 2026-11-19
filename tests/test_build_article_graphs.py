"""Integration test for scripts/build_article_graphs.py's core function:
verifies unit-level records (with unit_idx/parent_sentence) assemble into a
per-article graph with the expected node/meta shape."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_article_graphs import build_article_graphs  # noqa: E402


def _fake_embedding(seed: int, dim: int = 8) -> list[float]:
    import random

    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def test_build_article_graphs_produces_unit_level_nodes(tmp_path: Path) -> None:
    in_path = tmp_path / "units.jsonl"
    out_path = tmp_path / "graphs.json"

    records = []
    for sent_idx in range(4):
        for unit_idx in range(3):
            records.append(
                {
                    "article_id": "a1",
                    "sent_idx": sent_idx,
                    "unit_idx": unit_idx,
                    "text": f"unit {sent_idx}-{unit_idx}",
                    "parent_sentence": f"Sentence number {sent_idx} of the article.",
                    "embedding": _fake_embedding(sent_idx * 3 + unit_idx),
                }
            )
    in_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )

    data = build_article_graphs(in_path, out_path, k=3, min_cluster_size=2)

    assert out_path.exists()
    assert len(data["articles"]) == 1
    article = data["articles"][0]
    assert article["article_id"] == "a1"
    assert article["meta"]["granularity"] == "semantic-unit (phrase-level, single article)"
    assert article["meta"]["num_nodes"] == len(records)
    assert len(article["nodes"]) == len(records)
    for node in article["nodes"]:
        assert "parent_sentence" in node
        assert node["meta"].startswith("sentence ")
        assert "unit " in node["meta"]
