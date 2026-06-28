"""Seed the gold-question set used for annotator-reliability control (Phase 5)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langembed.annotation.db import get_db
from langembed.annotation.models import Item


def seed(path: str) -> int:
    gen = get_db()
    db = next(gen)
    n = 0
    try:
        for line in Path(path).open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            db.add(
                Item(
                    sentence_a=r["sentence_a"],
                    sentence_b=r["sentence_b"],
                    status="gold",
                    gold_label=float(r["score"]),
                )
            )
            n += 1
        db.commit()
    finally:
        gen.close()
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/gold_gu.jsonl")
    args = ap.parse_args()
    print("seeded gold items:", seed(args.path))


if __name__ == "__main__":
    main()
