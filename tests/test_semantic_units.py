"""Unit tests for phrase-level semantic-unit chunking (semantic graph /
unit-level embeddings granularity)."""

from __future__ import annotations

from langembed.data.semantic_units import extract_semantic_units, regex_units, spacy_units


def test_regex_units_drops_stopword_and_punctuation_runs() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    result = regex_units(text)
    # no syntactic awareness: a content-word run merges across a
    # noun/verb boundary when nothing separates them (here "fox jumps")
    assert result == ["quick brown fox jumps", "lazy dog"]


def test_regex_units_empty_for_all_stopwords() -> None:
    assert regex_units("the a an of to in on") == []


def test_spacy_units_splits_noun_phrase_and_verb() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    result = spacy_units(text)
    # noun_chunks legitimately include their determiner ("The quick brown fox")
    assert "The quick brown fox" in result
    assert "the lazy dog" in result
    assert "jumps" in result
    # "over" (bare preposition, not absorbed into any phrase) never appears
    # as its own unit
    assert "over" not in result


def test_spacy_units_keeps_idiom_as_single_unit() -> None:
    text = "This model is the state of the art for this task."
    result = spacy_units(text)
    assert "state of the art" in result
    # the idiom's words must not also appear split up as separate units
    assert "state" not in result
    assert "the art" not in result


def test_spacy_units_absorbs_particle_into_verb_phrase() -> None:
    text = "We need to figure out the answer."
    result = spacy_units(text)
    assert any(u.lower() == "figure out" for u in result)


def test_spacy_units_every_content_word_covered_exactly_once() -> None:
    text = "Researchers trained a large model on diverse multilingual data."
    result = spacy_units(text)
    joined = " ".join(result).lower()
    for content_word in (
        "researchers",
        "trained",
        "large",
        "model",
        "diverse",
        "multilingual",
        "data",
    ):
        assert content_word in joined, f"{content_word!r} missing from units: {result}"


def test_extract_semantic_units_dispatches_by_method() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    assert extract_semantic_units(text, method="spacy") == spacy_units(text)
    assert extract_semantic_units(text, method="regex") == regex_units(text)


def test_extract_semantic_units_rejects_unknown_method() -> None:
    try:
        extract_semantic_units("hello", method="bogus")
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown method")
