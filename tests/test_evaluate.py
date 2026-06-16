"""Unit tests for evaluate helpers."""
from __future__ import annotations

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
        """Query[i] maps to e_i; Doc[j] maps to e_{n-1-j}. With n=4, no query matches its doc at rank 1."""

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
