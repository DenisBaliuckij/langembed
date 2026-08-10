"""Streaming reservoir sampling: uniformly sample a bounded number of lines from a
possibly-huge file without ever holding the full file in memory. Shared by any pipeline
step that needs a bounded-memory subsample of a corpus (e.g. train_simcse, run_pipeline's
SVD auto-labeling)."""

from __future__ import annotations

import random


def reservoir_sample(path: str, k: int, seed: int) -> list[str]:
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
