"""Annotation quality control: inter-annotator agreement, aggregation, reliability (Phase 5)."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score


def weighted_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Quadratic-weighted Cohen's kappa between two annotators."""
    return float(cohen_kappa_score(a, b, weights="quadratic"))


def aggregate(labels: Sequence[float], reliabilities: Sequence[float]) -> float:
    """Reliability-weighted mean of labels (not plain majority)."""
    return float(
        np.average(
            np.asarray(labels, dtype=float),
            weights=np.asarray(reliabilities, dtype=float),
        )
    )


def update_reliability(correct_on_gold: int, total_gold: int, prior: float = 2.0) -> float:
    """Smoothed accuracy on gold questions, mapped to a weight in [0, 1]."""
    return (correct_on_gold + prior) / (total_gold + 2 * prior)
