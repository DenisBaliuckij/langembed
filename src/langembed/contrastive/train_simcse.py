"""Phase 4: unsupervised SimCSE on top of the from-scratch encoder."""
from __future__ import annotations

import argparse
from typing import Any

from langembed.config import load_config


def train_simcse(cfg: dict[str, Any], smoke: bool = False) -> None:
    from sentence_transformers import InputExample, SentenceTransformer, losses, models
    from torch.utils.data import DataLoader

    s = cfg["simcse"]
    word = models.Transformer(cfg["encoder_dir"], max_seq_length=s["max_seq_length"])
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
    model = SentenceTransformer(modules=[word, pool])

    with open(s["sentences_path"], encoding="utf-8") as f:
        sents = [ln.strip() for ln in f if ln.strip()]
    if smoke:
        sents = sents[:256]
    examples = [InputExample(texts=[x, x]) for x in sents]  # dropout gives the positive
    loader = DataLoader(examples, batch_size=s["batch_size"], shuffle=True)
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
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    train_simcse(load_config(args.config), smoke=args.smoke)


if __name__ == "__main__":
    main()
