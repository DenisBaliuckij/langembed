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


def test_embed_passes_lang_and_spacy_model_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """LANGEMBED_LANG and LANGEMBED_SPACY_MODEL must both be forwarded to normalize()."""
    import langembed.serving.serve as srv

    captured: list[tuple[str, str, Any]] = []

    def _fake_normalize(text: str, lang: str = "gu", spacy_model: str | None = None) -> str:
        captured.append((text, lang, spacy_model))
        return text

    monkeypatch.setattr(srv, "normalize", _fake_normalize)
    monkeypatch.setenv("LANGEMBED_LANG", "ru")
    monkeypatch.setenv("LANGEMBED_SPACY_MODEL", "ru_core_news_sm")
    monkeypatch.setattr(srv, "_model", None)
    monkeypatch.setattr(srv, "_get_model", lambda: _FakeModel())

    client = TestClient(srv.app)
    client.post("/embed", json={"texts": ["hello"]})
    assert captured == [("hello", "ru", "ru_core_news_sm")]


def test_embed_spacy_model_env_unset_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without LANGEMBED_SPACY_MODEL set, normalize() must receive spacy_model=None."""
    import langembed.serving.serve as srv

    captured: list[tuple[str, str, Any]] = []

    def _fake_normalize(text: str, lang: str = "gu", spacy_model: str | None = None) -> str:
        captured.append((text, lang, spacy_model))
        return text

    monkeypatch.setattr(srv, "normalize", _fake_normalize)
    monkeypatch.delenv("LANGEMBED_SPACY_MODEL", raising=False)
    monkeypatch.setattr(srv, "_model", None)
    monkeypatch.setattr(srv, "_get_model", lambda: _FakeModel())

    client = TestClient(srv.app)
    client.post("/embed", json={"texts": ["hello"]})
    assert captured == [("hello", "gu", None)]
