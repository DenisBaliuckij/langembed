"""Convert scored sentence pairs into (anchor, positive, negative) training
triplets -- see docs/superpowers/specs/2026-08-10-supervised-finetune-pass-design.md."""

from __future__ import annotations

import random

import numpy as np


def build_triplets_from_pairs(
    pairs: list[tuple[str, str, float]],
    positive_percentile: float = 0.7,
    negative_percentile: float = 0.3,
    seed: int = 42,
) -> list[tuple[str, str, str]]:
    """Convert scored (sentence_a, sentence_b, score) pairs into (anchor, positive,
    negative) triplets. Pairs scoring at or above the `positive_percentile` of this
    batch's own score distribution become positive candidates; pairs at or below
    `negative_percentile` become negative candidates. Positive and negative
    candidates are shuffled independently (seeded) and zipped, so the returned
    triplet count is min(len(positive_candidates), len(negative_candidates)).

    Percentile-based rather than a fixed absolute threshold (unlike
    langembed.annotation.api's _build_triplets) because some label methods produce
    continuous, uniformly-distributed scores where a fixed high threshold could
    starve the positive bucket -- a percentile split adapts to whatever
    distribution a given method actually produces.
    """
    if not pairs:
        return []

    scores = [score for _, _, score in pairs]
    pos_cutoff = float(np.percentile(scores, positive_percentile * 100))
    neg_cutoff = float(np.percentile(scores, negative_percentile * 100))

    positive = [(a, b) for a, b, score in pairs if score >= pos_cutoff]
    negative = [(a, b) for a, b, score in pairs if score <= neg_cutoff]

    rng = random.Random(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)

    triplets: list[tuple[str, str, str]] = []
    for (anchor, pos_sentence), (_, neg_sentence) in zip(positive, negative, strict=False):
        if pos_sentence != neg_sentence:
            triplets.append((anchor, pos_sentence, neg_sentence))
    return triplets
