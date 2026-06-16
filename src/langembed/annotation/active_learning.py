"""Active-learning sampler: prioritize the most informative pairs (Phase 5)."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def uncertainty_from_cosine(cos: np.ndarray) -> np.ndarray:
    """Uncertainty peaks at the decision boundary (cos ~ 0.5); clipped to [0, 1]."""
    cos = np.asarray(cos, dtype=float)
    return np.clip(1.0 - np.abs(cos - 0.5) / 0.5, 0.0, 1.0)


def uncertainty(pairs: Sequence[tuple[str, str]], model: object) -> np.ndarray:
    """Score pairs by model uncertainty. `model` is a SentenceTransformer."""
    from sentence_transformers import util

    a = model.encode(  # type: ignore[attr-defined]
        [p[0] for p in pairs], convert_to_tensor=True, normalize_embeddings=True
    )
    b = model.encode(  # type: ignore[attr-defined]
        [p[1] for p in pairs], convert_to_tensor=True, normalize_embeddings=True
    )
    cos = util.pairwise_cos_sim(a, b).cpu().numpy()
    return uncertainty_from_cosine(cos)
