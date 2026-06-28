"""Single source of truth for text normalization (used by both train and serve).

Indic-specific normalization is applied when indic-nlp-library is installed; if
it is missing the function falls back to NFC + whitespace collapse so the module
stays importable and testable.
"""

from __future__ import annotations

import functools
import re
import unicodedata

_WS = re.compile(r"\s+")


@functools.lru_cache(maxsize=4)
def _indic_normalizer(lang: str) -> object | None:
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

        return IndicNormalizerFactory().get_normalizer(lang)
    except Exception:
        return None


def normalize(text: str, lang: str = "gu") -> str:
    """Normalize text deterministically. Idempotent: normalize(normalize(x)) == normalize(x)."""
    text = unicodedata.normalize("NFC", text)
    norm = _indic_normalizer(lang)
    if norm is not None:
        text = norm.normalize(text)  # type: ignore[attr-defined]
    return _WS.sub(" ", text).strip()
