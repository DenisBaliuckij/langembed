"""Contract tests for the annotation FastAPI service (Phase 5)."""

from __future__ import annotations

import json
import os
import tempfile

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


def test_queue_orders_pending_by_uncertainty_desc(client: TestClient, db_session: Session) -> None:
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


def test_annotate_multiple_labels_same_item(client: TestClient, db_session: Session) -> None:
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
        lines = [json.loads(line) for line in open(out_path) if line.strip()]
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


def test_export_skips_below_positive_threshold(client: TestClient, db_session: Session) -> None:
    ann = _add_annotator(db_session)
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
