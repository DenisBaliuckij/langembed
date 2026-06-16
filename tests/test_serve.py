"""Contract tests for the /embed serving endpoint (Phase 7)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient


class _FakeModel:
    """Deterministic fake encoder: returns L2-normalised all-ones vectors of dim 4."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def encode(
        self, texts: list[str], normalize_embeddings: bool = True, **kwargs: Any
    ) -> np.ndarray:
        n = len(texts)
        vecs = np.ones((n, self.dim))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms


def _make_client(monkeypatch: pytest.MonkeyPatch, dim: int = 4) -> TestClient:
    import langembed.serving.serve as srv

    fake = _FakeModel(dim=dim)
    monkeypatch.setattr(srv, "_model", None)
    monkeypatch.setattr(srv, "_get_model", lambda: fake)
    return TestClient(srv.app)


def test_embed_returns_correct_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch)
    resp = client.post("/embed", json={"texts": ["hello", "world", "foo"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["dim"] == 4
    assert len(data["embeddings"]) == 3
    assert len(data["embeddings"][0]) == 4


def test_embed_normalize_applied_before_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    """normalize() must be called on input before passing to the model."""
    import langembed.serving.serve as srv
    from langembed.preprocess import normalize

    captured: list[str] = []

    class _CapturingModel:
        def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
            captured.extend(texts)
            return np.ones((len(texts), 4))

    monkeypatch.setattr(srv, "_model", None)
    monkeypatch.setattr(srv, "_get_model", lambda: _CapturingModel())

    client = TestClient(srv.app)
    raw = "  hello   world  "
    client.post("/embed", json={"texts": [raw]})
    assert captured[0] == normalize(raw)


def test_embed_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same input must produce identical embeddings on successive calls."""
    client = _make_client(monkeypatch)
    r1 = client.post("/embed", json={"texts": ["test sentence"]})
    r2 = client.post("/embed", json={"texts": ["test sentence"]})
    assert r1.json()["embeddings"] == r2.json()["embeddings"]


def test_embed_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch)
    resp = client.post("/embed", json={"texts": []})
    assert resp.status_code == 200
    assert resp.json()["embeddings"] == []
