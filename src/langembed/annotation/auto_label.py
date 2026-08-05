"""Silver-standard STS pair generation via back-translation — no human labeler needed."""

from __future__ import annotations

import json
import random
from pathlib import Path

from langembed.data.backtranslate import back_translate, load_cache

PARAPHRASE_SCORE = 4.8
ADJACENT_SCORE = 2.0
RANDOM_SCORE = 0.3


def build_auto_sts_pairs(
    sentences: list[str],
    n: int,
    providers: list[str],
    pivot_lang: str,
    source_lang: str,
    cache_path: str | Path,
    requests_per_minute: float = 20.0,
    seed: int = 42,
) -> list[tuple[str, str, float]]:
    """Silver STS pairs in three tiers, evenly split across `n`: back-translated
    paraphrases (high similarity), adjacent corpus sentences (mid similarity), and
    random distant sentence pairs (low similarity). Pairs where every translation
    provider fails are dropped, not padded, so the paraphrase tier may end up smaller
    than the other two.
    """
    if len(sentences) < 2:
        return []

    rng = random.Random(seed)
    cache = load_cache(cache_path)
    delay = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
    n_each = max(1, n // 3)

    pairs: list[tuple[str, str, float]] = []

    anchors = rng.sample(sentences, min(n_each, len(sentences)))
    for s in anchors:
        para = back_translate(s, providers, pivot_lang, source_lang, cache, cache_path, delay=delay)
        if para and para != s:
            pairs.append((s, para, PARAPHRASE_SCORE))

    adjacent_idx = list(range(len(sentences) - 1))
    for i in rng.sample(adjacent_idx, min(n_each, len(adjacent_idx))):
        pairs.append((sentences[i], sentences[i + 1], ADJACENT_SCORE))

    for _ in range(n_each):
        a, b = rng.sample(range(len(sentences)), 2)
        pairs.append((sentences[a], sentences[b], RANDOM_SCORE))

    rng.shuffle(pairs)
    return pairs[:n]


def write_sts_pairs(pairs: list[tuple[str, str, float]], out_path: str | Path) -> int:
    """Write (sentence_a, sentence_b, score) triples as STS-test JSONL, matching
    `annotation.api.export_sts`'s schema exactly."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for a, b, score in pairs:
            f.write(
                json.dumps({"sentence_a": a, "sentence_b": b, "score": score}, ensure_ascii=False)
                + "\n"
            )
    return len(pairs)
