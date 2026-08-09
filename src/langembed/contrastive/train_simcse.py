"""Phase 4: unsupervised SimCSE on top of the from-scratch encoder."""

from __future__ import annotations

import argparse
import random
import warnings
from typing import Any

from langembed.config import load_config

# Bounds SimCSE training data size (and therefore peak memory) regardless of
# raw corpus size. gu's ~13.8M-sentence corpus OOM-killed this step when the
# whole corpus was materialized twice in memory (once as strings, again
# wrapped in InputExample); pa's ~6.3M-sentence corpus trained fine, so this
# stays comfortably below that already-proven scale.
MAX_TRAIN_EXAMPLES = 2_000_000
SMOKE_EXAMPLES = 256


def _reservoir_sample(path: str, k: int, seed: int) -> list[str]:
    """Uniformly sample up to k non-empty lines from `path` in one streaming
    pass, without ever holding the full file in memory (Algorithm R). A file
    with fewer than k lines is returned in full, in its original order."""
    rng = random.Random(seed)
    reservoir: list[str] = []
    with open(path, encoding="utf-8") as f:
        i = 0
        for line in f:
            s = line.strip()
            if not s:
                continue
            if i < k:
                reservoir.append(s)
            else:
                j = rng.randint(0, i)
                if j < k:
                    reservoir[j] = s
            i += 1
    return reservoir


def train_simcse(cfg: dict[str, Any], smoke: bool = False) -> None:
    # Pre-load transitive deps in correct order to avoid DLL-init crash
    # (sentence_transformers → pandas → pyarrow segfaults on Windows + Python 3.14
    # unless torch/datasets/pyarrow are imported first in the same process).
    import datasets  # noqa: F401
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    import torch  # noqa: F401
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.base.modules import Transformer
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from sentence_transformers.sentence_transformer.modules import Pooling
    from sentence_transformers.sentence_transformer.readers import InputExample
    from torch.utils.data import DataLoader

    s = cfg["simcse"]
    word = Transformer(cfg["encoder_dir"], max_seq_length=s["max_seq_length"])
    pool = Pooling(word.get_embedding_dimension(), pooling_mode="mean")
    model = SentenceTransformer(modules=[word, pool])

    n_target = SMOKE_EXAMPLES if smoke else MAX_TRAIN_EXAMPLES
    sents = _reservoir_sample(s["sentences_path"], n_target, cfg.get("seed", 42))
    examples = [InputExample(texts=[x, x]) for x in sents]  # dropout gives the positive
    loader: DataLoader = DataLoader(examples, batch_size=s["batch_size"], shuffle=True)
    loss = MultipleNegativesRankingLoss(model)
    # fit() uses deprecated ST 5.x internals that hard-code dataloader_pin_memory=True
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="'pin_memory'", category=UserWarning)
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
