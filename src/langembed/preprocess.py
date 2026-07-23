"""Single source of truth for text normalization (used by both train and serve).

Indic-specific normalization is applied when indic-nlp-library is installed; if
it is missing the function falls back to NFC + whitespace collapse so the module
stays importable and testable. Linguistic text preparation (lemmatization +
POS-token substitution) is applied when a spaCy model name is given and spaCy /
that model are installed; otherwise it is skipped the same way.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from typing import Any

_WS = re.compile(r"\s+")

_POS_TOKENS = {"PROPN": "person1", "PRON": "pron1", "NUM": "ordinal1"}
_ABBR_TOKEN = "abbr1"
_RESERVED_TOKENS = frozenset({*_POS_TOKENS.values(), _ABBR_TOKEN})


@functools.lru_cache(maxsize=4)
def _indic_normalizer(lang: str) -> object | None:
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

        return IndicNormalizerFactory().get_normalizer(lang)
    except Exception:
        return None


@functools.lru_cache(maxsize=4)
def _spacy_pipeline(model_name: str) -> object | None:
    try:
        import spacy

        return spacy.load(model_name, exclude=["ner", "parser"])
    except Exception:
        return None


def _looks_like_abbreviation(text: str) -> bool:
    return "." in text and any(ch.isalpha() for ch in text)


def _prepare_tokens(text: str, model_name: str) -> str | None:
    """Lemmatize + substitute PROPN/PRON/NUM/abbreviation tokens via spaCy.

    Returns None (leaving `text` untouched) if spaCy or `model_name` isn't available.
    """
    nlp: Any = _spacy_pipeline(model_name)
    if nlp is None:
        return None
    out: list[str] = []
    for tok in nlp(text):
        if tok.is_space:
            continue
        if tok.text in _RESERVED_TOKENS:
            out.append(tok.text)  # already-prepared text is a fixed point (idempotency)
        elif _looks_like_abbreviation(tok.text):
            out.append(_ABBR_TOKEN)
        elif tok.pos_ in _POS_TOKENS:
            out.append(_POS_TOKENS[tok.pos_])
        else:
            out.append(tok.lemma_.lower())
    return " ".join(out)


def normalize(text: str, lang: str = "gu", spacy_model: str | None = None) -> str:
    """Normalize text deterministically. Idempotent: normalize(normalize(x)) == normalize(x)."""
    text = unicodedata.normalize("NFC", text)
    norm = _indic_normalizer(lang)
    if norm is not None:
        text = norm.normalize(text)  # type: ignore[attr-defined]
    if spacy_model:
        prepared = _prepare_tokens(text, spacy_model)
        if prepared is not None:
            text = prepared
    return _WS.sub(" ", text).strip()
