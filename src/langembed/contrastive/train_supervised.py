"""Phase 4: supervised contrastive fine-tuning on native-speaker triplets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from langembed.config import load_config


def train_supervised(cfg: dict[str, Any]) -> None:
    from sentence_transformers import (  # type: ignore[attr-defined]
        InputExample,
        SentenceTransformer,
        losses,
    )
    from torch.utils.data import DataLoader

    s = cfg["supervised"]
    path = Path(s["triplets_path"])
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(
            f"No triplets at {path}. Run Phase 5 (annotation service) and POST /export first."
        )
    model = SentenceTransformer(s["in_dir"])
    examples = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        examples.append(InputExample(texts=[r["anchor"], r["positive"], r["negative"]]))
    loader: DataLoader = DataLoader(examples, batch_size=s["batch_size"], shuffle=True)  # type: ignore[arg-type]
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=s["epochs"],
        warmup_steps=s["warmup_steps"],
        output_path=s["out_dir"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train_supervised(load_config(args.config))


if __name__ == "__main__":
    main()
