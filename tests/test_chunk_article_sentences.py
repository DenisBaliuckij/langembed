"""Integration test for scripts/chunk_article_sentences.py's core function:
verifies sentence -> unit explosion wiring (unit_idx, parent_sentence)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from chunk_article_sentences import chunk_article_sentences  # noqa: E402


def test_chunk_article_sentences_explodes_sentences_into_units(tmp_path: Path) -> None:
    in_path = tmp_path / "sentences.jsonl"
    out_path = tmp_path / "units.jsonl"
    in_path.write_text(
        json.dumps({"article_id": "a1", "sent_idx": 0, "text": "The quick brown fox jumps."})
        + "\n"
        + json.dumps({"article_id": "a1", "sent_idx": 1, "text": "We need to figure out why."})
        + "\n",
        encoding="utf-8",
    )

    n_units, n_sentences = chunk_article_sentences(
        in_path, out_path, method="regex", spacy_model="unused"
    )

    assert n_sentences == 2
    assert n_units > 0

    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert all(r["article_id"] == "a1" for r in records)
    sent0 = [r for r in records if r["sent_idx"] == 0]
    assert [r["unit_idx"] for r in sent0] == list(range(len(sent0)))
    assert all(r["parent_sentence"] == "The quick brown fox jumps." for r in sent0)
