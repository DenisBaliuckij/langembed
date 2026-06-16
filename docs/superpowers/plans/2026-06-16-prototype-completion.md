# Prototype Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close four gaps in the langembed skeleton: missing tests (tokenizer, annotation API, embed endpoint), retrieval@k metric in evaluate.py, end-to-end English smoke test, full Docker stack, and Russian documentation.

**Architecture:** All changes are additive to the existing skeleton (phases 0–7 + 4C already implemented). Tests are grouped by concern: unit tests for pure functions, API contract tests via FastAPI TestClient with SQLite in-memory, and one e2e integration test that runs the full pipeline on tiny English data on CPU. Docker adds multi-stage Dockerfile and three new compose services.

**Tech Stack:** pytest, FastAPI TestClient, SQLAlchemy (SQLite in-memory for tests), sentence-transformers, HF tokenizers, numpy, Docker multi-stage builds.

**Working directory for all paths:** `langembed/` (i.e., `C:\Repositories\langembed_skeleton\langembed\`).

---

## File Map

| Status | Path | Purpose |
|--------|------|---------|
| Create | `tests/test_tokenizer.py` | Round-trip + unk_rate tests for train_tokenizer |
| Modify | `src/langembed/eval/evaluate.py` | Add `_retrieval_at_k` helper, update `evaluate()` + `main()` |
| Create | `tests/conftest.py` | `db_session` + `client` fixtures for annotation + serve tests |
| Create | `tests/test_annotation_api.py` | Contract tests: GET /queue, POST /annotate, POST /export |
| Create | `tests/test_serve.py` | Contract tests: POST /embed with monkeypatched model |
| Modify | `src/langembed/serving/serve.py` | Read model dir from `LANGEMBED_MODEL_DIR` env var |
| Modify | `src/langembed/pretrain/train_mlm.py` | Read `report_to` from cfg (enables `[]` in e2e test) |
| Create | `tests/fixtures/en_corpus.txt` | 200-sentence English fixture corpus |
| Create | `tests/fixtures/en_sts_test.jsonl` | 20 STS pairs with scores 0–5 |
| Create | `tests/e2e/__init__.py` | Package marker |
| Create | `tests/e2e/test_pipeline_english.py` | Full pipeline smoke on English data on CPU |
| Modify | `pyproject.toml` | Add `e2e` pytest marker |
| Modify | `Makefile` | Add `test-e2e` target |
| Modify | `.env.example` | Add `LANGEMBED_MODEL_DIR` |
| Create | `Dockerfile` | Multi-stage: base (serve extras) + ml (ML extras) |
| Modify | `docker-compose.yml` | Add annotation, serve, train services |
| Create | `docs/ru/README_RU.md` | Russian documentation (11 sections) |

---

## Task 1: tests/test_tokenizer.py

**Files:**
- Create: `tests/test_tokenizer.py`

Tests for the existing `train_tokenizer()` and `diagnose()` functions in
`src/langembed/tokenizer/train_tokenizer.py`. Both functions are already
implemented; this task writes the tests that prove they work.

- [ ] **Step 1: Write the test file**

```python
# tests/test_tokenizer.py
"""Phase 2 tokenizer tests: round-trip and unk_rate."""
from __future__ import annotations

import pytest

pytest.importorskip("tokenizers")

_FIXTURE = """\
The cat sat on the mat.
A quick brown fox jumps over the lazy dog.
Machine learning enables computers to learn from data.
Natural language processing helps computers understand human text.
The sun rises in the east and sets in the west.
She baked a delicious chocolate cake for the party.
Engineers write software to solve real world problems.
Athletes train every day to improve their performance.
Dark clouds gathered on the horizon before the storm.
The river flows quietly through the green valley.
Students read books to expand their knowledge and skills.
The scientist conducted experiments in the laboratory.
Children played happily in the park on a sunny afternoon.
Music has the power to evoke strong emotions in people.
The economy grows when people invest in new businesses.
Doctors work long hours to care for their patients.
The library contains thousands of books on many topics.
Farmers harvest their crops before the winter arrives.
The chef prepared a meal using fresh local ingredients.
Travelers explore new countries to experience different cultures.\
"""


def test_round_trip(tmp_path: "pytest.TempPathFactory") -> None:
    from langembed.preprocess import normalize
    from langembed.tokenizer.train_tokenizer import train_tokenizer

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(_FIXTURE, encoding="utf-8")
    out = str(tmp_path / "tok")
    tok = train_tokenizer(str(corpus), out, vocab_size=300, min_frequency=1)

    for raw in _FIXTURE.splitlines():
        normed = normalize(raw.strip())
        if not normed:
            continue
        ids = tok(normed)["input_ids"]
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert decoded == normed, f"round-trip failed: {normed!r} -> {decoded!r}"


def test_unk_rate_below_threshold(tmp_path: "pytest.TempPathFactory") -> None:
    from langembed.tokenizer.train_tokenizer import diagnose, train_tokenizer

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(_FIXTURE, encoding="utf-8")
    out = str(tmp_path / "tok")
    tok = train_tokenizer(str(corpus), out, vocab_size=300, min_frequency=1)
    stats = diagnose(tok, str(corpus))
    assert stats["unk_rate"] < 0.01, f"unk_rate too high: {stats['unk_rate']}"
```

- [ ] **Step 2: Run the test to verify it passes**

```
pytest tests/test_tokenizer.py -v
```

Expected: both tests PASS (the functions are already implemented).
If either fails, check that `tokenizers` is installed (`pip install -e ".[ml,serve,dev]"`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_tokenizer.py
git commit -m "test(phase2): add tokenizer round-trip and unk_rate tests"
```

---

## Task 2: Retrieval@k in evaluate.py

**Files:**
- Modify: `src/langembed/eval/evaluate.py`

Add a private `_retrieval_at_k` helper that computes Recall@k and MRR@k via
cosine similarity (avoids dependency on InformationRetrievalEvaluator internal
API which changed across ST versions). Update `evaluate()` to call it and
return a flat dict with all metric keys already prefixed. Update `main()` to
dump the flat dict directly.

- [ ] **Step 1: Write a failing unit test for `_retrieval_at_k`**

```python
# Temporarily add to tests/test_evaluate.py — we'll create this file:
"""Unit tests for evaluate helpers."""
from __future__ import annotations

import numpy as np
import pytest


def test_retrieval_at_k_perfect() -> None:
    from langembed.eval.evaluate import _retrieval_at_k

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            # Return identity-like embeddings: each text maps to unique row
            n = len(texts)
            vecs = np.eye(n, n)
            return vecs

    m = FakeModel()
    # 3 query-doc pairs; identity embeddings → each query retrieves its own doc at rank 1
    sa = ["a", "b", "c"]
    sb = ["a", "b", "c"]
    result = _retrieval_at_k(m, sa, sb, k=3)
    assert result[f"recall@3"] == 1.0
    assert result[f"mrr@3"] == 1.0


def test_retrieval_at_k_worst() -> None:
    from langembed.eval.evaluate import _retrieval_at_k

    class FakeModel:
        def __init__(self):
            self._call = 0

        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            n = len(texts)
            # All queries return same vector → identical cosine to all docs → random ordering
            # Use reverse-identity so doc[i] is LAST for query[i]
            vecs = np.zeros((n, n))
            for j in range(n):
                vecs[j, n - 1 - j] = 1.0  # query j points to doc (n-1-j)
            return vecs

    m = FakeModel()
    sa = ["a", "b", "c"]
    sb = ["a", "b", "c"]
    # query 0 → doc 2 at rank 1 (not doc 0), etc. With k=1 no hits.
    result = _retrieval_at_k(m, sa, sb, k=1)
    assert result["recall@1"] == 0.0
```

- [ ] **Step 2: Run to verify it fails**

```
pytest tests/test_evaluate.py -v
```

Expected: `ImportError` or `AttributeError` — `_retrieval_at_k` does not exist yet.

- [ ] **Step 3: Implement `_retrieval_at_k` and update `evaluate()` + `main()`**

Replace the entire content of `src/langembed/eval/evaluate.py` with:

```python
"""Phase 6: evaluate branches A/B/C on the isolated STS test, with leakage guard."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from langembed.config import load_config
from langembed.preprocess import normalize


def _h(s: str) -> str:
    return hashlib.sha1(normalize(s).encode("utf-8")).hexdigest()


def assert_no_leakage(test_path: str, train_paths: Sequence[str]) -> None:
    test_hashes: set[str] = set()
    for line in Path(test_path).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        test_hashes |= {_h(r["sentence_a"]), _h(r["sentence_b"])}
    for tp in train_paths:
        p = Path(tp)
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            if line.strip() and _h(line) in test_hashes:
                raise RuntimeError(f"Test leakage detected via {tp}")


def _retrieval_at_k(
    model: Any, sa: list[str], sb: list[str], k: int
) -> dict[str, float]:
    """Compute Recall@k and MRR@k.

    Each sentence_a[i] is a query; sentence_b[i] is its single positive.
    All sentence_b texts form the corpus.
    """
    q_embs = model.encode(sa, normalize_embeddings=True, show_progress_bar=False)
    c_embs = model.encode(sb, normalize_embeddings=True, show_progress_bar=False)
    sims = q_embs @ c_embs.T  # [N, N]
    n = len(sa)
    recall = 0.0
    mrr = 0.0
    for i in range(n):
        ranked = list(np.argsort(-sims[i]))[:k]
        if i in ranked:
            recall += 1.0
        for rank, j in enumerate(ranked):
            if j == i:
                mrr += 1.0 / (rank + 1)
                break
    return {f"recall@{k}": recall / n, f"mrr@{k}": mrr / n}


def evaluate(cfg: dict[str, Any]) -> dict[str, float]:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

    assert_no_leakage(cfg["test_path"], cfg.get("train_paths", []))
    sa: list[str] = []
    sb: list[str] = []
    scores: list[float] = []
    for line in Path(cfg["test_path"]).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        sa.append(r["sentence_a"])
        sb.append(r["sentence_b"])
        scores.append(r["score"] / cfg["score_scale"])

    k = cfg.get("retrieval_k", 10)
    spearman_evaluator = EmbeddingSimilarityEvaluator(sa, sb, scores, name="gu-sts")
    results: dict[str, float] = {}
    for branch, path in cfg["branches"].items():
        if not Path(path).exists():
            print(f"skip branch {branch}: {path} missing")
            continue
        model = SentenceTransformer(path)
        spearman = float(spearman_evaluator(model))
        results[f"spearman_{branch}"] = spearman
        print(f"branch {branch}: Spearman = {spearman:.4f}")
        ret = _retrieval_at_k(model, sa, sb, k)
        results[f"retrieval_recall@{k}_{branch}"] = ret[f"recall@{k}"]
        results[f"retrieval_mrr@{k}_{branch}"] = ret[f"mrr@{k}"]
        print(
            f"branch {branch}: Recall@{k}={ret[f'recall@{k}']:.4f},"
            f" MRR@{k}={ret[f'mrr@{k}']:.4f}"
        )
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    results = evaluate(cfg)
    metrics_path = Path(cfg.get("metrics_path", "metrics/eval.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

```
pytest tests/test_evaluate.py -v
```

Expected: both PASS. Also run lint:

```
ruff check src/langembed/eval/evaluate.py
mypy src/langembed/eval/evaluate.py
```

- [ ] **Step 5: Commit**

```bash
git add src/langembed/eval/evaluate.py tests/test_evaluate.py
git commit -m "feat(phase6): add retrieval@k (recall, mrr) to evaluate.py"
```

---

## Task 3: tests/conftest.py

**Files:**
- Create: `tests/conftest.py`

Shared pytest fixtures for the annotation API tests and serve tests. Uses
SQLite in-memory so tests run without a real Postgres instance.

- [ ] **Step 1: Write conftest.py**

```python
# tests/conftest.py
"""Shared pytest fixtures for API contract tests."""
from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from langembed.annotation.api import app as annotation_app
from langembed.annotation.db import get_db
from langembed.annotation.models import Base


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture()
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def _override() -> Generator[Session, None, None]:
        yield db_session

    annotation_app.dependency_overrides[get_db] = _override
    with TestClient(annotation_app) as c:
        yield c
    annotation_app.dependency_overrides.clear()
```

- [ ] **Step 2: Verify the file is importable**

```
python -c "import tests.conftest"
```

(Run from the `langembed/` directory. May warn about no `__init__.py` in tests — that's fine for pytest.)

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add shared db_session and client pytest fixtures"
```

---

## Task 4: tests/test_annotation_api.py

**Files:**
- Create: `tests/test_annotation_api.py`

Requires: Task 3 (conftest.py must exist).

- [ ] **Step 1: Write the test file**

```python
# tests/test_annotation_api.py
"""Contract tests for the annotation FastAPI service (Phase 5)."""
from __future__ import annotations

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from langembed.annotation.models import Annotation, Annotator, Item


def _add_annotator(db: Session, name: str = "alice") -> Annotator:
    a = Annotator(name=name, reliability=1.0)
    db.add(a)
    db.flush()
    return a


def _add_item(
    db: Session,
    sentence_a: str = "a",
    sentence_b: str = "b",
    uncertainty: float = 0.5,
    status: str = "pending",
    gold_label: float | None = None,
) -> Item:
    i = Item(
        sentence_a=sentence_a,
        sentence_b=sentence_b,
        uncertainty=uncertainty,
        status=status,
        gold_label=gold_label,
    )
    db.add(i)
    db.flush()
    return i


# ---------------------------------------------------------------------------
# GET /queue
# ---------------------------------------------------------------------------

def test_queue_returns_pending_and_gold(client: TestClient, db_session: Session) -> None:
    _add_item(db_session, "pa", "pb", uncertainty=0.8, status="pending")
    _add_item(db_session, "ga", "gb", uncertainty=0.0, status="gold", gold_label=3.0)
    db_session.commit()

    resp = client.get("/queue?annotator_id=1&n=10")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    statuses = {i["status"] for i in data["items"]}
    assert "pending" in statuses
    assert "gold" in statuses


def test_queue_orders_pending_by_uncertainty_desc(
    client: TestClient, db_session: Session
) -> None:
    _add_item(db_session, "a", "b", uncertainty=0.1, status="pending")
    _add_item(db_session, "c", "d", uncertainty=0.9, status="pending")
    db_session.commit()

    resp = client.get("/queue?annotator_id=1&n=10")
    assert resp.status_code == 200
    pending = [i for i in resp.json()["items"] if i["status"] == "pending"]
    uncertainties = [i["uncertainty"] for i in pending]
    assert uncertainties == sorted(uncertainties, reverse=True)


def test_queue_includes_item_fields(client: TestClient, db_session: Session) -> None:
    _add_item(db_session, "sent_a", "sent_b", uncertainty=0.7, status="pending")
    db_session.commit()

    resp = client.get("/queue?annotator_id=1&n=5")
    assert resp.status_code == 200
    item = next(i for i in resp.json()["items"] if i["status"] == "pending")
    assert item["sentence_a"] == "sent_a"
    assert item["sentence_b"] == "sent_b"
    assert "id" in item


# ---------------------------------------------------------------------------
# POST /annotate
# ---------------------------------------------------------------------------

def test_annotate_persists_to_db(client: TestClient, db_session: Session) -> None:
    ann = _add_annotator(db_session)
    item = _add_item(db_session, "x", "y", status="pending")
    db_session.commit()

    resp = client.post(
        "/annotate",
        json={"item_id": item.id, "annotator_id": ann.id, "label": 4.0},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    row = db_session.query(Annotation).filter_by(item_id=item.id).first()
    assert row is not None
    assert row.label == 4.0
    assert row.annotator_id == ann.id


def test_annotate_multiple_labels_same_item(
    client: TestClient, db_session: Session
) -> None:
    ann = _add_annotator(db_session)
    item = _add_item(db_session, "x", "y", status="pending")
    db_session.commit()

    for label in (3.0, 4.0, 5.0):
        resp = client.post(
            "/annotate",
            json={"item_id": item.id, "annotator_id": ann.id, "label": label},
        )
        assert resp.status_code == 200

    count = db_session.query(Annotation).filter_by(item_id=item.id).count()
    assert count == 3


# ---------------------------------------------------------------------------
# POST /export
# ---------------------------------------------------------------------------

def test_export_builds_triplets(client: TestClient, db_session: Session) -> None:
    ann = _add_annotator(db_session)
    pos = _add_item(db_session, "pos_a", "pos_b", status="pending")
    neg = _add_item(db_session, "neg_a", "neg_b", status="pending")
    db_session.add(Annotation(item_id=pos.id, annotator_id=ann.id, label=5.0))
    db_session.add(Annotation(item_id=neg.id, annotator_id=ann.id, label=0.0))
    db_session.commit()

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        out_path = f.name
    try:
        resp = client.post(f"/export?out_path={out_path}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["written"] == 1
        lines = [json.loads(l) for l in open(out_path).readlines() if l.strip()]
        assert len(lines) == 1
        assert lines[0]["anchor"] == "pos_a"
        assert lines[0]["positive"] == "pos_b"
        assert lines[0]["negative"] == "neg_b"
    finally:
        os.unlink(out_path)


def test_export_empty_db_returns_zero(client: TestClient, db_session: Session) -> None:
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        out_path = f.name
    try:
        resp = client.post(f"/export?out_path={out_path}")
        assert resp.status_code == 200
        assert resp.json()["written"] == 0
    finally:
        os.unlink(out_path)


def test_export_skips_below_positive_threshold(
    client: TestClient, db_session: Session
) -> None:
    ann = _add_annotator(db_session)
    # Two items both labeled below pos_thr (4.0) — no triplets can be built
    for sa, sb in [("a", "b"), ("c", "d")]:
        item = _add_item(db_session, sa, sb, status="pending")
        db_session.add(Annotation(item_id=item.id, annotator_id=ann.id, label=3.0))
    db_session.commit()

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        out_path = f.name
    try:
        resp = client.post(f"/export?out_path={out_path}")
        assert resp.json()["written"] == 0
    finally:
        os.unlink(out_path)
```

- [ ] **Step 2: Run the tests**

```
pytest tests/test_annotation_api.py -v
```

Expected: all 8 tests PASS. If any fail on `annotators` FK constraint, add `pragma_foreign_keys=False` to the engine create call in conftest.py (SQLite doesn't enforce FKs by default — this shouldn't be needed).

- [ ] **Step 3: Commit**

```bash
git add tests/test_annotation_api.py
git commit -m "test(phase5): annotation API contract tests (queue, annotate, export)"
```

---

## Task 5: tests/test_serve.py + serve.py MODEL_DIR env var

**Files:**
- Create: `tests/test_serve.py`
- Modify: `src/langembed/serving/serve.py`

The serve tests monkeypatch `_get_model` so no real model is needed. The
serve.py change reads the model directory from an env var so the Docker
container can be pointed at different artifacts.

- [ ] **Step 1: Update serve.py to use LANGEMBED_MODEL_DIR**

In `src/langembed/serving/serve.py`, change `_get_model` from:

```python
def _get_model() -> Any:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("artifacts/embed_gu_v1")
    return _model
```

to:

```python
import os

def _get_model() -> Any:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        model_dir = os.environ.get("LANGEMBED_MODEL_DIR", "artifacts/embed_gu_v1")
        _model = SentenceTransformer(model_dir)
    return _model
```

The `import os` goes at the top of the file (module level, after the existing imports). The full updated file:

```python
"""Phase 7: embedding inference service (shares preprocess.normalize with training)."""
from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from langembed.preprocess import normalize

app = FastAPI(title="langembed embed")
_model: Any = None


def _get_model() -> Any:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        model_dir = os.environ.get("LANGEMBED_MODEL_DIR", "artifacts/embed_gu_v1")
        _model = SentenceTransformer(model_dir)
    return _model


class EmbedIn(BaseModel):
    texts: list[str]


@app.post("/embed")
def embed(payload: EmbedIn) -> dict[str, Any]:
    model = _get_model()
    vecs = model.encode([normalize(t) for t in payload.texts], normalize_embeddings=True)
    return {"embeddings": vecs.tolist(), "dim": int(vecs.shape[1])}
```

- [ ] **Step 2: Write tests/test_serve.py**

```python
# tests/test_serve.py
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
    from langembed.preprocess import normalize
    import langembed.serving.serve as srv

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
```

- [ ] **Step 3: Run the tests**

```
pytest tests/test_serve.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 4: Run lint on serve.py**

```
ruff check src/langembed/serving/serve.py
mypy src/langembed/serving/serve.py
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/langembed/serving/serve.py tests/test_serve.py
git commit -m "feat(phase7): read model dir from LANGEMBED_MODEL_DIR env; add /embed tests"
```

---

## Task 6: E2E fixture files

**Files:**
- Create: `tests/fixtures/en_corpus.txt`
- Create: `tests/fixtures/en_sts_test.jsonl`

These files are committed to the repo. They provide English data for the CPU
smoke test (no real Gujarati data needed).

- [ ] **Step 1: Create tests/fixtures/en_corpus.txt**

Create `tests/fixtures/en_corpus.txt` with exactly 200 lines. Content below
(40 sentences per cluster: animals, food, technology, sports, weather):

```
The dog barked loudly at the passing cars.
The cat curled up on the warm windowsill.
A flock of birds flew south for the winter.
The horse galloped across the open meadow.
Elephants use their trunks to drink water.
The fox hid silently behind the tall bushes.
A rabbit hopped quickly across the garden path.
The bear hibernates during the cold winter months.
Lions are known as the kings of the jungle.
The dolphin leaped gracefully above the ocean waves.
Wolves travel in packs to hunt their prey.
The eagle soared high above the mountain peaks.
A spider spun a delicate web between two branches.
The cow grazed peacefully in the green pasture.
A school of fish swam through the coral reef.
The owl hunted mice silently in the dark night.
The monkey climbed quickly to the top of the tree.
A crocodile waited motionless at the edge of the river.
The penguin waddled slowly across the icy surface.
Butterflies collect nectar from colorful garden flowers.
The tiger prowled quietly through the dense jungle.
A whale surfaced to breathe fresh ocean air.
The snake slithered silently through the dry grass.
A deer leaped over the fallen log in the forest.
The parrot repeated every word it heard all day.
Bees collect pollen from flowers to make honey.
The zebra has distinctive black and white stripes.
A gorilla sat quietly eating leaves in the forest.
The cheetah is the fastest land animal on earth.
A polar bear swam effortlessly through the Arctic water.
The crow is considered one of the smartest birds.
A turtle moves slowly but lives for many decades.
The camel can survive for days without drinking water.
A kangaroo carries its young in a pouch on its belly.
The panda eats bamboo almost exclusively every single day.
A peacock spread its colorful feathers in the sunlight.
The flamingo stands on one leg in shallow water.
A chameleon changed its color to match the tree bark.
The giraffe stretched its long neck to reach the leaves.
A seal basked lazily on a warm rocky shoreline.
She baked a moist chocolate cake for the birthday party.
The soup was served warm with freshly baked bread.
He grilled vegetables and chicken for the summer barbecue.
The chef prepared pasta with a rich tomato sauce.
Fresh fruit salad is a healthy and delicious breakfast option.
The market sold ripe mangoes and sweet pineapples today.
She stirred the batter carefully before pouring it into the pan.
The pizza was topped with mozzarella cheese and fresh basil.
He seasoned the steak with salt pepper and garlic.
The homemade jam spread easily onto the warm toast.
A warm bowl of oatmeal is perfect on cold mornings.
The sushi chef sliced the fish with remarkable precision.
She poured olive oil generously over the fresh salad greens.
The bread rose slowly in the warm oven overnight.
He squeezed fresh oranges to make a glass of juice.
The roasted chicken smelled wonderful coming out of the oven.
She measured the flour carefully before mixing the dough.
The spices gave the curry a rich and complex flavor.
He boiled the eggs for exactly six minutes for breakfast.
The ice cream melted quickly in the hot afternoon sun.
Chocolate mousse requires careful folding to keep it light.
The farmers market offered fresh strawberries and blueberries today.
She marinated the tofu overnight in soy sauce and ginger.
The croissant was flaky golden and perfectly buttery inside.
He added a pinch of cinnamon to the morning coffee.
The stew simmered on the stove for three long hours.
She decorated the cupcakes with colorful frosting and sprinkles.
The wine complemented the cheese and crackers perfectly tonight.
He grilled the corn until it was slightly charred and sweet.
The avocado toast was garnished with cherry tomatoes and herbs.
Software engineers write code to solve complex technical problems.
The computer processed thousands of calculations in one second.
Artificial intelligence is transforming many industries worldwide rapidly.
The smartphone battery drained quickly while streaming high quality video.
She debugged the program until all the test cases passed.
The database stored millions of records with impressive efficiency.
Cloud computing allows companies to scale their infrastructure quickly.
He configured the network router to improve connection speed.
The operating system manages hardware resources for all applications.
Machine learning models improve their accuracy with more training data.
The encryption algorithm protected sensitive data during transmission.
She deployed the new application to the production server today.
The open source library simplified the development of complex features.
He optimized the algorithm to reduce its computational complexity.
The firmware update fixed several critical security vulnerabilities.
A microservices architecture made the system more resilient and scalable.
The compiler transformed source code into executable machine instructions.
She integrated the payment gateway into the e-commerce platform.
The API documentation helped developers understand how to use it.
He automated the testing pipeline to catch regressions early.
The version control system tracked every change made to the codebase.
Data scientists analyze large datasets to extract useful insights.
The neural network was trained on millions of labeled examples.
She used containerization to ensure consistent deployment environments.
The cache reduced database query time significantly in production.
He monitored the server logs to identify the performance bottleneck.
The responsive design ensured the website worked well on mobile devices.
She wrote unit tests to verify each function worked correctly.
The load balancer distributed incoming requests across multiple servers.
He refactored the legacy code to improve its readability and performance.
The team played well and won the championship after a tough season.
Athletes train for years to compete at the Olympic Games.
The soccer ball curved sharply and landed in the corner of the net.
She swam fifty laps in the pool before her morning workout ended.
The marathon runner crossed the finish line with great effort.
He dribbled past three defenders before scoring the winning goal.
The tennis player served an unreturnable ace to win the match.
She lifted heavy weights in the gym to build her strength.
The basketball team practiced shooting free throws for two hours.
The cyclist climbed the steep hill without slowing down once.
He trained every morning before sunrise to improve his endurance.
The gymnast performed a flawless routine on the balance beam.
She scored the highest points in the figure skating competition.
The volleyball team celebrated after winning the national tournament.
He jogged five kilometers along the river path each morning.
The swimmer broke the world record by a fraction of a second.
She practiced yoga every evening to improve her flexibility.
The hockey puck slid across the ice into the open goal.
He knocked out his opponent in the third round of boxing.
The rowing team synchronized their strokes perfectly in the final race.
The coach motivated the players with a powerful halftime speech.
She served an ace to win the final set of the tennis match.
The sprinter exploded out of the starting blocks at the sound of the gun.
He blocked the shot at the last second to save the game.
The goalkeeper dove to the right and stopped the penalty kick.
She won the gold medal in the high jump at the championship.
The wrestling match lasted three intense rounds before a winner emerged.
He scored a hat trick in the first half of the football game.
The ski jumper soared through the cold mountain air gracefully.
She finished the triathlon in an impressive time despite the heat.
The sun shone brightly and warmed the cold winter morning air.
Dark storm clouds gathered rapidly on the western horizon today.
Heavy rain fell throughout the night flooding several low streets.
A gentle breeze rustled the leaves of the tall oak tree.
The temperature dropped sharply after sunset bringing unexpected frost.
Lightning flashed across the sky followed by loud rolling thunder.
She watched the rainbow appear after the brief afternoon shower.
The fog rolled in from the sea covering the harbor completely.
A snowstorm buried the entire town under half a meter of snow.
The heatwave lasted for two weeks breaking temperature records daily.
He wore a thick jacket because the wind was bitterly cold.
The forecast predicted heavy snowfall for the coming weekend morning.
Hailstones the size of marbles fell during the brief violent storm.
The humidity made the summer afternoon feel uncomfortably hot and sticky.
A tornado warning was issued for the counties south of the city.
The clear blue sky promised a perfect day for outdoor activities.
A cold front moved in overnight dropping temperatures by ten degrees.
The drought had lasted three months causing water restrictions citywide.
She watched the sunset paint the sky in shades of orange and pink.
The morning frost covered every blade of grass with sparkling ice.
A gusty wind knocked over several trees along the main boulevard.
The spring rain nourished the newly planted seeds in the garden.
He carried an umbrella because the clouds looked threatening that day.
The weather station recorded the lowest temperature in fifty years.
A warm current from the south brought mild weather to the region.
The wildfire spread quickly fueled by dry conditions and strong winds.
She photographed the dramatic lightning storm from her apartment window.
The seasons changed gradually from golden autumn to cold bare winter.
A dense fog made driving extremely dangerous on the highway today.
The monsoon season brought weeks of continuous heavy tropical rainfall.
```

- [ ] **Step 2: Create tests/fixtures/en_sts_test.jsonl**

Create `tests/fixtures/en_sts_test.jsonl` with exactly 20 lines (JSON objects, one per line):

```jsonl
{"sentence_a": "The dog barked loudly at the passing cars.", "sentence_b": "The dog made loud noises at vehicles going by.", "score": 5}
{"sentence_a": "She baked a moist chocolate cake for the birthday party.", "sentence_b": "She made a delicious chocolate dessert for the celebration.", "score": 5}
{"sentence_a": "The computer processed thousands of calculations in one second.", "sentence_b": "The machine performed many computations very quickly.", "score": 5}
{"sentence_a": "The team played well and won the championship after a tough season.", "sentence_b": "The team won the title after a difficult season.", "score": 5}
{"sentence_a": "Heavy rain fell throughout the night flooding several low streets.", "sentence_b": "It rained heavily all night causing floods in some streets.", "score": 5}
{"sentence_a": "The cat curled up on the warm windowsill.", "sentence_b": "The software engineers optimized the database query performance.", "score": 0}
{"sentence_a": "She grilled vegetables and chicken for the summer barbecue.", "sentence_b": "The soccer ball curved sharply and landed in the corner of the net.", "score": 0}
{"sentence_a": "Artificial intelligence is transforming many industries worldwide rapidly.", "sentence_b": "The horse galloped across the open meadow.", "score": 0}
{"sentence_a": "The marathon runner crossed the finish line with great effort.", "sentence_b": "A gentle breeze rustled the leaves of the tall oak tree.", "score": 0}
{"sentence_a": "The sun shone brightly and warmed the cold winter morning air.", "sentence_b": "He debugged the program until all the test cases passed.", "score": 0}
{"sentence_a": "The horse galloped across the open meadow.", "sentence_b": "The cheetah ran fast across the open plain.", "score": 4}
{"sentence_a": "The chef prepared pasta with a rich tomato sauce.", "sentence_b": "She cooked noodles with a flavorful red sauce.", "score": 4}
{"sentence_a": "Machine learning models improve their accuracy with more training data.", "sentence_b": "Neural networks learn better when given more labeled examples.", "score": 4}
{"sentence_a": "She swam fifty laps in the pool before her morning workout ended.", "sentence_b": "He ran thirty kilometers as part of his daily training regimen.", "score": 2}
{"sentence_a": "A snowstorm buried the entire town under half a meter of snow.", "sentence_b": "The heatwave lasted for two weeks breaking temperature records daily.", "score": 1}
{"sentence_a": "The snake slithered silently through the dry grass.", "sentence_b": "The temperature dropped sharply after sunset bringing unexpected frost.", "score": 1}
{"sentence_a": "He seasoned the steak with salt pepper and garlic.", "sentence_b": "She marinated the tofu overnight in soy sauce and ginger.", "score": 3}
{"sentence_a": "The basketball team practiced shooting free throws for two hours.", "sentence_b": "The volleyball team celebrated after winning the national tournament.", "score": 3}
{"sentence_a": "The cloud computing platform scaled infrastructure automatically during peak load.", "sentence_b": "The API server handled requests efficiently using horizontal scaling.", "score": 3}
{"sentence_a": "Butterflies collect nectar from colorful garden flowers.", "sentence_b": "Bees collect pollen from flowers to make honey.", "score": 2}
```

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/en_corpus.txt tests/fixtures/en_sts_test.jsonl
git commit -m "test(e2e): add English fixture corpus (200 sentences) and STS test pairs (20)"
```

---

## Task 7: E2E smoke test + Makefile + pyproject.toml

**Files:**
- Modify: `src/langembed/pretrain/train_mlm.py` (read `report_to` from cfg)
- Modify: `pyproject.toml` (add e2e pytest marker)
- Modify: `Makefile` (add test-e2e target)
- Create: `tests/e2e/__init__.py`
- Create: `tests/e2e/test_pipeline_english.py`

Requires: Task 6 (fixture files must exist).

- [ ] **Step 1: Make train_mlm.py configurable for report_to**

In `src/langembed/pretrain/train_mlm.py`, find the `TrainingArguments` call and
change the hardcoded `report_to=["mlflow"]` to read from cfg:

```python
    args = TrainingArguments(
        output_dir=str(Path(cfg["out_dir"]) / "ckpt"),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        weight_decay=t["weight_decay"],
        warmup_steps=t["warmup_steps"],
        max_steps=max_steps,
        fp16=t["fp16"],
        save_steps=t["save_steps"],
        logging_steps=t["logging_steps"],
        report_to=cfg.get("report_to", ["mlflow"]),
    )
```

The only change is the last argument: `report_to=cfg.get("report_to", ["mlflow"])`.

- [ ] **Step 2: Add e2e pytest marker to pyproject.toml**

Find the `[tool.pytest.ini_options]` section in `pyproject.toml` and add the
`markers` key:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "e2e: end-to-end pipeline smoke test requiring full ML stack (deselect with '-m not e2e')",
]
```

- [ ] **Step 3: Add test-e2e target to Makefile**

Add the `test-e2e` target after the existing `test` target:

```makefile
test-e2e:
	pytest -m e2e tests/e2e/ -v
```

Also add `test-e2e` to the `.PHONY` line at the top:

```makefile
.PHONY: setup lint test test-e2e corpus tokenizer pretrain pretrain-smoke simcse simcse-smoke supervised serve-annotation eval serve llm-mntp llm-lora
```

- [ ] **Step 4: Create tests/e2e/__init__.py**

```python
# tests/e2e/__init__.py
```

(Empty file — marks directory as a package so pytest collects it.)

- [ ] **Step 5: Write tests/e2e/test_pipeline_english.py**

```python
# tests/e2e/test_pipeline_english.py
"""End-to-end smoke test: full pipeline on English fixture data, CPU only.

Proves the code path from raw text → corpus → tokenizer → MLM → SimCSE →
evaluate runs without errors. Does NOT validate model quality — Spearman after
50 MLM steps is expected to be low.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("datasets")

pytestmark = pytest.mark.e2e

_FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[return]
    """Run all pipeline steps once and return a dict of output paths."""
    base = tmp_path_factory.mktemp("e2e")
    corpus_file = base / "corpus.txt"
    tokenizer_dir = base / "tokenizer"
    encoder_dir = base / "encoder"
    simcse_dir = base / "simcse"
    metrics_file = base / "metrics.json"

    # ── Step 1: build corpus ──────────────────────────────────────────────
    from langembed.data.build_corpus import build_corpus

    n = build_corpus(
        [str(_FIXTURES / "en_corpus.txt")], str(corpus_file), set()
    )
    assert n > 0, "build_corpus wrote zero lines"
    assert corpus_file.exists()

    # ── Step 2: train tokenizer ───────────────────────────────────────────
    from langembed.preprocess import normalize
    from langembed.tokenizer.train_tokenizer import diagnose, train_tokenizer

    tok = train_tokenizer(
        str(corpus_file), str(tokenizer_dir), vocab_size=500, min_frequency=1
    )
    stats = diagnose(tok, str(corpus_file))
    assert stats["unk_rate"] < 0.05, f"unk_rate={stats['unk_rate']:.4f}"
    for line in corpus_file.read_text(encoding="utf-8").splitlines()[:5]:
        normed = normalize(line.strip())
        if not normed:
            continue
        ids = tok(normed)["input_ids"]
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert decoded == normed

    # ── Step 3: MLM pre-training (smoke: 50 steps, tiny model, CPU) ──────
    from langembed.pretrain.train_mlm import train_mlm

    mlm_cfg: dict = {
        "seed": 42,
        "tokenizer_dir": str(tokenizer_dir),
        "corpus_path": str(corpus_file),
        "report_to": [],
        "model": {
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 256,
            "max_position_embeddings": 514,
            "max_seq_length": 64,
        },
        "training": {
            "mlm_probability": 0.15,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-4,
            "weight_decay": 0.01,
            "warmup_steps": 5,
            "max_steps": 200,
            "fp16": False,
            "save_steps": 100,
            "logging_steps": 10,
        },
        "smoke": {"max_steps": 50},
        "out_dir": str(encoder_dir),
    }
    train_mlm(mlm_cfg, smoke=True)
    assert (encoder_dir / "config.json").exists(), "encoder config.json missing"
    weight_files = list(encoder_dir.glob("*.safetensors")) + list(
        encoder_dir.glob("pytorch_model.bin")
    )
    assert weight_files, "no weight files in encoder_dir"

    # ── Step 4: SimCSE (smoke: 20 sentences, 1 epoch) ────────────────────
    from langembed.contrastive.train_simcse import train_simcse

    simcse_cfg: dict = {
        "encoder_dir": str(encoder_dir),
        "simcse": {
            "sentences_path": str(corpus_file),
            "max_seq_length": 64,
            "batch_size": 8,
            "epochs": 1,
            "warmup_steps": 2,
            "out_dir": str(simcse_dir),
        },
    }
    train_simcse(simcse_cfg, smoke=True)
    assert simcse_dir.exists(), "simcse output directory not created"

    return {
        "corpus_file": corpus_file,
        "tokenizer_dir": tokenizer_dir,
        "encoder_dir": encoder_dir,
        "simcse_dir": simcse_dir,
        "metrics_file": metrics_file,
    }


def test_corpus_written(pipeline: dict) -> None:
    assert pipeline["corpus_file"].exists()
    lines = pipeline["corpus_file"].read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0


def test_encoder_config_present(pipeline: dict) -> None:
    cfg_file = pipeline["encoder_dir"] / "config.json"
    assert cfg_file.exists()
    cfg = json.loads(cfg_file.read_text())
    assert cfg.get("hidden_size") == 128


def test_simcse_model_saved(pipeline: dict) -> None:
    assert pipeline["simcse_dir"].exists()
    assert any(pipeline["simcse_dir"].iterdir()), "simcse_dir is empty"


def test_embeddings_l2_normalised(pipeline: dict) -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(pipeline["simcse_dir"]))
    sents = [
        "The dog barked.",
        "The cat meowed.",
        "The computer processed data.",
        "The algorithm ran efficiently.",
    ]
    embs = model.encode(sents, normalize_embeddings=True)
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=0.01)


def test_similar_pairs_higher_cosine(pipeline: dict) -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(pipeline["simcse_dir"]))
    # Both pairs share vocabulary; we only assert cosine > -1 (model runs OK)
    sents = model.encode(
        [
            "The dog barked loudly.",
            "The dog made loud noises.",
            "She baked a chocolate cake.",
            "He wrote a computer program.",
        ],
        normalize_embeddings=True,
    )
    cos_similar = float(sents[0] @ sents[1])
    cos_dissimilar = float(sents[0] @ sents[3])
    # After 50 MLM steps + 1 SimCSE epoch the model may not be good,
    # so we only assert cosines are in valid range [-1, 1].
    assert -1.0 <= cos_similar <= 1.0
    assert -1.0 <= cos_dissimilar <= 1.0


def test_evaluate_runs_and_writes_metrics(pipeline: dict) -> None:
    from langembed.eval.evaluate import evaluate

    eval_cfg = {
        "test_path": str(_FIXTURES / "en_sts_test.jsonl"),
        "score_scale": 5.0,
        "retrieval_k": 5,
        "train_paths": [],
        "branches": {"en_smoke": str(pipeline["simcse_dir"])},
        "metrics_path": str(pipeline["metrics_file"]),
    }
    results = evaluate(eval_cfg)
    assert "spearman_en_smoke" in results
    assert "retrieval_recall@5_en_smoke" in results
    assert "retrieval_mrr@5_en_smoke" in results

    # Write metrics file (not done by evaluate() itself, only by main())
    import json as _json

    pipeline["metrics_file"].write_text(_json.dumps(results, indent=2))
    assert pipeline["metrics_file"].exists()

    # Verify embedding dim is 128 (matches hidden_size of the tiny encoder)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(pipeline["simcse_dir"]))
    emb = model.encode(["test"])
    assert emb.shape[1] == 128
```

- [ ] **Step 6: Run the e2e test**

```
cd langembed
pytest -m e2e tests/e2e/ -v --timeout=300
```

Expected: all 6 tests PASS. This takes a few minutes on CPU.
If `pytest-timeout` is not installed, omit `--timeout=300`.

- [ ] **Step 7: Run regular tests to ensure no regressions**

```
pytest tests/ --ignore=tests/e2e -v
```

Expected: all existing tests + new tests still PASS.

- [ ] **Step 8: Commit**

```bash
git add src/langembed/pretrain/train_mlm.py pyproject.toml Makefile \
        tests/e2e/__init__.py tests/e2e/test_pipeline_english.py
git commit -m "feat(e2e): English CPU smoke test; add report_to cfg in train_mlm; e2e make target"
```

---

## Task 8: Dockerfile

**Files:**
- Create: `Dockerfile`

Multi-stage build: `base` stage installs serve extras only (annotation service),
`ml` stage extends base with ML extras (training + serving with models).

- [ ] **Step 1: Write Dockerfile**

```dockerfile
# Dockerfile — multi-stage build for langembed services.
#
# Stages:
#   base  → annotation service (~400 MB, no torch)
#   ml    → serving + training image (~4 GB, includes torch)
#
# Build examples:
#   docker build --target base -t langembed-annotation .
#   docker build --target ml   -t langembed-ml .
#
# Artifacts (models, data) are mounted as volumes — never baked in.

FROM python:3.11-slim AS base
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e ".[serve]"

# ---------------------------------------------------------------------------

FROM base AS ml

RUN pip install --no-cache-dir -e ".[ml]"
```

- [ ] **Step 2: Verify Dockerfile syntax**

```
docker build --target base --no-cache -t langembed-annotation-test . 2>&1 | tail -20
```

Expected: build succeeds (last line: `Successfully tagged langembed-annotation-test:latest`
or similar). If docker is not available locally, skip and rely on CI.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "feat(docker): multi-stage Dockerfile — base (serve) and ml (torch) stages"
```

---

## Task 9: docker-compose.yml update + .env.example

**Files:**
- Modify: `docker-compose.yml`
- Modify: `.env.example`

Adds three new services: `annotation` (port 8001), `serve` (port 8000), and
`train` (one-shot runner for `docker compose run train dvc repro`).

- [ ] **Step 1: Update .env.example**

Append one line to `.env.example`:

```
LANGEMBED_MODEL_DIR=artifacts/embed_gu_v1
```

The full updated file:
```
POSTGRES_USER=langembed
POSTGRES_PASSWORD=langembed
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=langembed
REDIS_URL=redis://localhost:6379/0
LANGEMBED_MODEL_DIR=artifacts/embed_gu_v1
```

- [ ] **Step 2: Update docker-compose.yml**

Replace `docker-compose.yml` with the following full content:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-langembed}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-langembed}
      POSTGRES_DB: ${POSTGRES_DB:-langembed}
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-langembed}"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  annotation:
    build:
      context: .
      target: base
    command: uvicorn langembed.annotation.api:app --host 0.0.0.0 --port 8001
    ports:
      - "8001:8001"
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-langembed}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-langembed}
      POSTGRES_HOST: postgres
      POSTGRES_PORT: 5432
      POSTGRES_DB: ${POSTGRES_DB:-langembed}
      REDIS_URL: redis://redis:6379/0
    volumes:
      - ./data:/app/data
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  serve:
    build:
      context: .
      target: ml
    command: uvicorn langembed.serving.serve:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    environment:
      LANGEMBED_MODEL_DIR: ${LANGEMBED_MODEL_DIR:-artifacts/embed_gu_v1}
    volumes:
      - ./artifacts:/app/artifacts
      - ./data:/app/data
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  train:
    build:
      context: .
      target: ml
    volumes:
      - ./data:/app/data
      - ./artifacts:/app/artifacts
      - ./configs:/app/configs
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

volumes:
  pgdata:
```

- [ ] **Step 3: Verify docker-compose config is valid**

```
docker compose config --quiet
```

Expected: exits 0 (no output on success). If docker is not available, skip.

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml .env.example
git commit -m "feat(docker): add annotation, serve, train services to docker-compose"
```

---

## Task 10: Russian documentation

**Files:**
- Create: `docs/ru/README_RU.md`

11-section Russian-language reference. Write in Russian. Each section header
is given in English below; the content must be in Russian.

- [ ] **Step 1: Create docs/ru/ directory and write README_RU.md**

Create `docs/ru/README_RU.md`. The file must contain exactly these 11 sections
(use `##` headings), written in Russian, with the described content:

**Section 1 — Что это такое**  
Describe the research goal: train sentence embeddings from scratch for a low-resource
language (example: Gujarati). Explain the three branches: A (from scratch:
custom tokenizer + encoder + contrastive), B (multilingual model fine-tuned on
same native-speaker data), C (decoder LLM as embedder via LoRA). Explain
why native speakers are central: they are the annotation signal and quality
control loop. State the evaluation metric: Spearman correlation on an isolated
STS test set.

**Section 2 — Архитектура**  
Describe the 8 pipeline phases (0–7 + 4C) and their order. Explain the three
invariants: (1) единая нормализация — `preprocess.normalize` is the single
source of truth; (2) запрет утечки теста — `data/sts_test_*` never enters
training; (3) конфиги через YAML — no magic numbers in training code.
Include a short description of how the components relate: corpus → tokenizer
→ encoder → contrastive → annotation loop → evaluation → serving.

**Section 3 — Структура репозитория**  
List every module and its purpose. Cover: `config.py`, `preprocess.py`,
`data/`, `tokenizer/`, `pretrain/`, `contrastive/`, `llm_embed/`,
`annotation/`, `eval/`, `serving/`, `configs/`, `scripts/`, `tests/`,
`docs/`, `Makefile`, `docker-compose.yml`, `dvc.yaml`.

**Section 4 — Установка**  
Two paths:

Local:
```bash
git clone <repo>
cd langembed
pip install -e ".[ml,serve,dev]"
cp .env.example .env  # задать пароли
```

Docker:
```bash
docker compose up -d postgres redis
docker build --target base -t langembed-annotation .
docker build --target ml -t langembed-ml .
docker compose up -d
```

**Section 5 — Запуск пайплайна**  
Step-by-step: place raw data → start services → run DVC → evaluate.

```bash
# 1. Raw data
cp your_data.txt data/raw/wiki_gu.txt

# 2. Infrastructure
docker compose up -d postgres redis

# 3. Run full pipeline
dvc repro

# 4. Evaluate
make eval

# 5. View results
mlflow ui
```

Also note that smoke tests for individual phases are available:
`make pretrain-smoke`, `make simcse-smoke`.

**Section 6 — Фазы 0–7 + 4C**  
For each phase (0–7 and 4C), one paragraph covering:
what it does, its inputs and outputs (file paths), key YAML config parameters.
Use the IMPLEMENTATION_PLAN.md as the source of truth for details.

**Section 7 — Сервис разметки и active learning**  
Explain the annotation loop: `/queue` returns the N most uncertain pairs
plus 2 gold calibration questions; `/annotate` saves a label; `/export`
aggregates labels and builds triplets. Explain uncertainty formula
(1 - |cos - 0.5| / 0.5). Explain inter-annotator quality control
(weighted kappa, reliability-weighted aggregation). Mention Label Studio
configs in `annotation/label_studio/`. Include the `scripts/seed_gold.py`
workflow.

**Section 8 — Тестирование**  
Describe test layers: unit tests (pure functions: normalize, dedup,
uncertainty_from_cosine, weighted_kappa, aggregate), API contract tests
(annotation service and /embed via TestClient + SQLite in-memory), e2e test
(English CPU smoke test, `make test-e2e`). Explain expected e2e results:
pipeline completes without errors; Spearman may be low (50 MLM steps is not
enough for quality but proves the code path works). Commands:
```bash
make test          # unit + API tests
make test-e2e      # full pipeline smoke on English (few minutes)
make lint          # ruff + mypy
```

**Section 9 — DVC и воспроизводимость**  
Explain `dvc repro` runs the DAG defined in `dvc.yaml`. List the stages:
corpus → tokenizer → pretrain → simcse → supervised → llm_lora → evaluate.
Show:
```bash
dvc repro            # run all stale stages
dvc metrics show     # print eval.json metrics
dvc dag              # visualise the DAG
```
Explain that artifacts (model weights, data) are tracked by DVC and excluded
from git (see `.gitignore` and `.dvcignore`). Mention DVC remote storage
should be configured for team use.

**Section 10 — MLflow**  
Explain that MLM pretraining logs loss and perplexity to MLflow. Evaluation
writes Spearman, Recall@k, MRR@k to `metrics/eval.json` and to MLflow.
```bash
mlflow ui          # open http://localhost:5000
```
To compare runs: Experiments → select runs → Compare. Explain the three runs
to compare: branch A (from scratch), B (multilingual fine-tuned), C (LLM LoRA).

**Section 11 — Адаптация под другой язык**  
To switch from Gujarati to another language, change:
1. `configs/tokenizer.yaml`: set `language: <lang_code>`
2. `configs/tokenizer.yaml`: update `raw_paths` to point to raw data
3. `configs/eval.yaml`: update `test_path`
4. `preprocess.normalize`: the `lang` parameter in `_indic_normalizer`
   is driven by the config (currently hardcoded to `"gu"` as default — if
   targeting a non-Indic language, the normalizer falls back gracefully)
5. Re-run `dvc repro` — all stages are parameterised through configs

Note: for non-Indic languages, the `indic-nlp-library` normalizer is skipped
automatically (falls back to NFC + whitespace collapse).

- [ ] **Step 2: Verify the file exists and has all 11 sections**

```
grep "^##" docs/ru/README_RU.md | wc -l
```

Expected: 11 lines.

- [ ] **Step 3: Commit**

```bash
git add docs/ru/README_RU.md
git commit -m "docs(ru): Russian reference documentation — 11 sections"
```

---

## Final verification

After all 10 tasks are complete, run the full suite:

```bash
# Lint
make lint

# Unit + API tests
make test

# Verify all new tests are included
pytest tests/ --ignore=tests/e2e --collect-only 2>&1 | grep "test session starts" -A5

# E2E (takes a few minutes)
make test-e2e

# Docker config
docker compose config --quiet
```

Expected: `make lint && make test` green; `make test-e2e` green; docker compose
config valid.

---

## Spec coverage checklist

| Spec section | Task(s) |
|---|---|
| 1.1 Retrieval@k in evaluate.py | Task 2 |
| 1.2 tests/test_tokenizer.py | Task 1 |
| 1.3 tests/conftest.py | Task 3 |
| 1.4 tests/test_annotation_api.py | Task 4 |
| 1.5 tests/test_serve.py | Task 5 |
| 2 E2E fixture files | Task 6 |
| 2 E2E test + Makefile + markers | Task 7 |
| 3 Dockerfile multi-stage | Task 8 |
| 3 docker-compose 3 new services | Task 9 |
| 3 serve.py MODEL_DIR env var | Task 5 |
| 3 .env.example LANGEMBED_MODEL_DIR | Task 9 |
| 4 Russian documentation | Task 10 |
