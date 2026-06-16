"""Phase 5: native-speaker annotation API (queue / annotate / export)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from langembed.annotation.db import get_db
from langembed.annotation.models import Annotation, Item
from langembed.annotation.quality import aggregate

app = FastAPI(title="langembed annotation")


class AnnotateIn(BaseModel):
    item_id: int
    annotator_id: int
    label: float


def _item(i: Item) -> dict[str, Any]:
    return {
        "id": i.id,
        "sentence_a": i.sentence_a,
        "sentence_b": i.sentence_b,
        "uncertainty": i.uncertainty,
        "status": i.status,
    }


@app.get("/queue")
def get_queue(annotator_id: int, n: int = 20, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return the n most informative pending pairs plus 2 hidden gold questions."""
    pending = db.scalars(
        select(Item).where(Item.status == "pending").order_by(Item.uncertainty.desc()).limit(n)
    ).all()
    gold = db.scalars(
        select(Item).where(Item.status == "gold").order_by(func.random()).limit(2)
    ).all()
    return {"items": [_item(i) for i in [*pending, *gold]]}


@app.post("/annotate")
def annotate(payload: AnnotateIn, db: Session = Depends(get_db)) -> dict[str, bool]:
    db.add(
        Annotation(item_id=payload.item_id, annotator_id=payload.annotator_id, label=payload.label)
    )
    db.commit()
    return {"ok": True}


@app.post("/export")
def export(
    out_path: str = "data/native_triplets.jsonl", db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Aggregate labels per item and emit (anchor, positive, negative) triplets."""
    rows = db.execute(select(Annotation.item_id, Annotation.label)).all()
    by_item: dict[int, list[float]] = defaultdict(list)
    for item_id, label in rows:
        by_item[item_id].append(label)
    scored = {iid: aggregate(v, [1.0] * len(v)) for iid, v in by_item.items()}
    triplets = _build_triplets(db, scored)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    return {"written": len(triplets), "path": out_path}


def _build_triplets(
    db: Session, scored: dict[int, float], pos_thr: float = 4.0, neg_thr: float = 1.0
) -> list[dict[str, str]]:
    items = {i.id: i for i in db.scalars(select(Item)).all()}
    pos = [items[i] for i, s in scored.items() if s >= pos_thr and i in items]
    neg = [items[i] for i, s in scored.items() if s <= neg_thr and i in items]
    out: list[dict[str, str]] = []
    for p, nneg in zip(pos, neg, strict=False):
        out.append({"anchor": p.sentence_a, "positive": p.sentence_b, "negative": nneg.sentence_b})
    return out
