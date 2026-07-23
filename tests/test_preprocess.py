"""Unit tests for preprocess.normalize()."""

from __future__ import annotations

import pytest

from langembed.preprocess import normalize


def test_idempotent():
    s = "  hello    world   "
    once = normalize(s)
    assert normalize(once) == once


def test_collapses_and_strips():
    assert normalize("a   b\tc\n") == "a b c"


def test_gu_unaffected_without_spacy_model():
    """No spacy_model given: identical to the pre-spaCy-feature behavior."""
    assert normalize("  hello    world   ", "gu") == "hello world"


def test_spacy_model_missing_falls_back_gracefully():
    """An undownloaded/nonexistent spaCy model must not raise."""
    result = normalize("Мама мыла раму.", "ru", "nonexistent-model-xyz")
    assert result == "Мама мыла раму."


def _ru_model_available() -> bool:
    try:
        import spacy

        spacy.load("ru_core_news_sm", exclude=["ner", "parser"])
        return True
    except Exception:
        return False


requires_ru_model = pytest.mark.skipif(
    not _ru_model_available(), reason="ru_core_news_sm spaCy model not installed"
)


@requires_ru_model
def test_spacy_lemmatizes():
    assert normalize("кошки бежали быстро", "ru", "ru_core_news_sm") == "кошка бежать быстро"


@requires_ru_model
def test_spacy_substitutes_pronoun_and_numeral():
    result = normalize("она купила пять яблок", "ru", "ru_core_news_sm")
    assert result == "pron1 купить ordinal1 яблоко"


@requires_ru_model
def test_spacy_substitutes_proper_noun():
    result = normalize("Пушкин написал роман", "ru", "ru_core_news_sm")
    assert result == "person1 написать роман"


@requires_ru_model
def test_spacy_substitutes_abbreviations():
    result = normalize("г. Москва и др. города", "ru", "ru_core_news_sm")
    assert result == "abbr1 person1 abbr1 город"


@requires_ru_model
def test_spacy_idempotent():
    once = normalize("она купила пять яблок у Пушкина", "ru", "ru_core_news_sm")
    twice = normalize(once, "ru", "ru_core_news_sm")
    assert twice == once
