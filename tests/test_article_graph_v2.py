"""Tests for the v2 per-article graph builder, the LLM-label casing and the viewer renderer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from langembed.data.article_graph_v2 import build_article_graph, select_nodes

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
from label_clusters_llm import sentence_case  # noqa: E402


def _units(n_per_topic: int = 8, dim: int = 16) -> list[dict]:
    """Two well-separated topics (+ noise) so clustering has an unambiguous answer."""
    rng = np.random.default_rng(0)
    centers = [np.eye(dim)[0] * 5, np.eye(dim)[1] * 5]
    units = []
    for t, c in enumerate(centers):
        for i in range(n_per_topic):
            text = f"topic{t} term{i}" if i else f"topic{t}"
            units.append(
                {
                    "article_id": "a",
                    "key": text,
                    "text": text,
                    "count": 10 - i if i < 8 else 2,
                    "n_sentences": 2,
                    "sent_ids": [i, i + 1],
                    "first_sent_idx": i,
                    "variants": [text],
                    "contexts": [{"sent_idx": i, "text": f"... {text} ..."}],
                    "embedding": (c + rng.normal(0, 0.3, dim)).tolist(),
                }
            )
    return units


def test_select_nodes_frequency_cap_and_rare_unknown_words() -> None:
    units = _units()
    units.append({**units[0], "key": "emakume", "text": "emakume", "count": 3})
    units.append({**units[0], "key": "rare", "text": "rare phrase", "count": 1})
    kept = select_nodes(units, min_count=2, max_nodes=10, lexicon={"topic"})
    texts = {u["text"] for u in kept}
    assert "rare phrase" not in texts  # below min_count
    assert "emakume" not in texts  # rare word missing from the lexicon
    assert len(kept) == 10


def test_build_article_graph_finds_the_two_topics() -> None:
    graph = build_article_graph(_units(), k=4, min_cluster_size=3, max_clusters=5)
    by_cluster: dict[int, set[str]] = {}
    for n in graph["nodes"]:
        by_cluster.setdefault(n["cluster"], set()).add(n["text"].split()[0])
    assert sorted(len(v) for v in by_cluster.values()) == [1, 1]  # each cluster is one topic
    labels = sorted(c["label"] for c in graph["clusters"])
    assert [lb.split()[0] for lb in labels] == ["Topic0", "Topic1"]  # named after its own topic
    assert graph["meta"]["num_nodes"] == 16
    node = graph["nodes"][0]
    assert {"count", "variants", "contexts", "key"} <= node.keys()


def test_sentence_case_keeps_acronyms_and_proper_names() -> None:
    assert sentence_case("Performance Evaluation Metrics") == "Performance evaluation metrics"
    assert sentence_case("Multilingual NER Datasets") == "Multilingual NER datasets"
    assert (
        sentence_case("Multilingual Wikipedia Resources", {"Wikipedia"})
        == "Multilingual Wikipedia resources"
    )


def test_render_viewer_embeds_data(tmp_path: Path) -> None:
    graph = build_article_graph(_units(), k=4, min_cluster_size=3, max_clusters=5)
    data = tmp_path / "a.json"
    data.write_text(
        json.dumps({"articles": [{"article_id": "a", "title": "a", **graph}]}), encoding="utf-8"
    )
    out = tmp_path / "a.html"
    template = Path(__file__).resolve().parent.parent / "templates" / "article_graph_viewer.html"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "render_article_graph_viewer.py"),
            "--template",
            str(template),
            "--data",
            str(data),
            "--out",
            str(out),
            "--title",
            "Test article",
        ],
        check=True,
    )
    html = out.read_text(encoding="utf-8")
    assert "/*ARTICLE_DATA*/null" not in html
    assert '"title":"Test article"' in html
    assert 'id="search"' in html
