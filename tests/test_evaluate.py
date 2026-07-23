"""Unit tests for evaluate helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest


def test_retrieval_at_k_perfect() -> None:
    from langembed.eval.evaluate import _retrieval_at_k

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            n = len(texts)
            # Identity matrix: each query perfectly matches its own doc
            return np.eye(n, n)

    result = _retrieval_at_k(FakeModel(), ["a", "b", "c"], ["a", "b", "c"], k=3)
    assert result["recall@3"] == 1.0
    assert result["mrr@3"] == 1.0


def test_retrieval_at_k_worst() -> None:
    from langembed.eval.evaluate import _retrieval_at_k

    class FakeModel:
        """Query[i] maps to e_i; Doc[j] maps to e_{n-1-j}.

        With n=4, no query matches its doc at rank 1.
        """

        def __init__(self) -> None:
            self._call = 0

        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            n = len(texts)
            self._call += 1
            if self._call == 1:
                # query embeddings: identity (query[i] = e_i)
                return np.eye(n)
            else:
                # doc embeddings: reversed identity (doc[j] = e_{n-1-j})
                vecs = np.zeros((n, n))
                for j in range(n):
                    vecs[j, n - 1 - j] = 1.0
                return vecs

    # n=4: no query[i] matches doc[i] at rank 1 (doc[0]→e3, doc[1]→e2, doc[2]→e1, doc[3]→e0)
    result = _retrieval_at_k(FakeModel(), ["a", "b", "c", "d"], ["a", "b", "c", "d"], k=1)
    assert result["recall@1"] == 0.0


def test_load_test_pairs_normalizes_sentences(tmp_path) -> None:
    from langembed.eval.evaluate import _load_test_pairs

    test_path = tmp_path / "test.jsonl"
    test_path.write_text(
        '{"sentence_a": "  hello   world  ", "sentence_b": "foo\\tbar", "score": 4}\n',
        encoding="utf-8",
    )
    sa, sb, scores = _load_test_pairs(str(test_path), score_scale=5.0)
    assert sa == ["hello world"]
    assert sb == ["foo bar"]
    assert scores == [0.8]


def _ru_model_available() -> bool:
    try:
        import spacy

        spacy.load("ru_core_news_sm", exclude=["ner", "parser"])
        return True
    except Exception:
        return False


requires_ru_model = pytest.mark.skipif(
    not _ru_model_available(), reason="ru_core_news_sm spaCy model not installed"
)


@requires_ru_model
def test_load_test_pairs_applies_spacy_model(tmp_path) -> None:
    from langembed.eval.evaluate import _load_test_pairs

    test_path = tmp_path / "test.jsonl"
    test_path.write_text(
        json.dumps({"sentence_a": "она купила пять яблок", "sentence_b": "x", "score": 5}) + "\n",
        encoding="utf-8",
    )
    sa, sb, _ = _load_test_pairs(
        str(test_path), score_scale=5.0, lang="ru", spacy_model="ru_core_news_sm"
    )
    assert sa == ["pron1 купить ordinal1 яблоко"]


@requires_ru_model
def test_assert_no_leakage_uses_spacy_model_consistently(tmp_path) -> None:
    """Same leakage-consistency proof as build_corpus.py's guard, for evaluate.py's copy."""
    from langembed.eval.evaluate import assert_no_leakage

    test_path = tmp_path / "test.jsonl"
    test_path.write_text(
        json.dumps({"sentence_a": "она купила пять яблок", "sentence_b": "x", "score": 5}) + "\n",
        encoding="utf-8",
    )
    train_path = tmp_path / "train.txt"
    train_path.write_text("она купила пять яблок\n", encoding="utf-8")

    with pytest.raises(RuntimeError):
        assert_no_leakage(str(test_path), [str(train_path)], "ru", "ru_core_news_sm")
