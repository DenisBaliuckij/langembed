"""Seed candidate STS pairs from the ru corpus, ranked by SimCSE uncertainty (Phase 5, ru track)."""

from __future__ import annotations

import argparse
import random

import numpy as np

from langembed.annotation.active_learning import uncertainty
from langembed.annotation.db import get_db
from langembed.annotation.models import Item
from langembed.config import load_config
from langembed.data.reservoir_sample import reservoir_sample

# build_candidates() makes one adjacent pair per sentence handed to it, then uncertainty()
# encodes every pair on top of that -- for a multi-million-line corpus that's millions of
# CPU-bound encode calls just to pick `n` candidates for human review. Bound the pool first.
MAX_CANDIDATE_SENTENCES = 5000


def build_candidates(sentences: list[str], n_random: int, seed: int) -> list[tuple[str, str]]:
    """Adjacent pairs (likely related) + random distant pairs (likely unrelated)."""
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []
    for i in range(len(sentences) - 1):
        pairs.append((sentences[i], sentences[i + 1]))
    idx = list(range(len(sentences)))
    for _ in range(n_random):
        a, b = rng.sample(idx, 2)
        pairs.append((sentences[a], sentences[b]))
    return pairs


def select_pairs(
    pairs: list[tuple[str, str]],
    scores: np.ndarray,
    n: int,
    max_reuse: int = 2,
) -> list[tuple[str, str, float]]:
    """Greedily take the highest-scored pairs, capping how often any sentence repeats."""
    order = np.argsort(-scores)
    usage: dict[str, int] = {}
    selected: list[tuple[str, str, float]] = []
    for i in order:
        a, b = pairs[i]
        if usage.get(a, 0) >= max_reuse or usage.get(b, 0) >= max_reuse:
            continue
        selected.append((a, b, float(scores[i])))
        usage[a] = usage.get(a, 0) + 1
        usage[b] = usage.get(b, 0) + 1
        if len(selected) >= n:
            break
    return selected


def seed(
    config_path: str,
    n: int = 60,
    n_random_pool: int = 400,
    max_reuse: int = 2,
    seed_value: int = 42,
    max_candidate_sentences: int = MAX_CANDIDATE_SENTENCES,
) -> int:
    import datasets  # noqa: F401
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    import torch  # noqa: F401
    from sentence_transformers import SentenceTransformer

    cfg = load_config(config_path)
    sentences = reservoir_sample(
        cfg["simcse"]["sentences_path"], max_candidate_sentences, seed_value
    )

    candidates = build_candidates(sentences, n_random_pool, seed_value)
    model = SentenceTransformer(cfg["simcse"]["out_dir"])
    scores = uncertainty(candidates, model)
    chosen = select_pairs(candidates, scores, n, max_reuse)

    gen = get_db()
    db = next(gen)
    written = 0
    try:
        for a, b, score in chosen:
            db.add(Item(sentence_a=a, sentence_b=b, uncertainty=score, status="pending"))
            written += 1
        db.commit()
    finally:
        gen.close()
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    print("seeded items:", seed(args.config, n=args.n))


if __name__ == "__main__":
    main()
