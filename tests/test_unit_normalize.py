"""Unit tests for the v2 per-article graph unit normalization (colleague feedback 2026-10-11:
duplicates, stop words, citations, surnames, foreign example lines)."""

from __future__ import annotations

from langembed.data.unit_normalize import (
    cited_names,
    clean_article_text,
    is_foreign_line,
    merge_units,
    normalized_units,
    strip_citations,
)


def test_determiners_dropped_and_lemma_key_shared() -> None:
    a = dict(normalized_units("The sentence is true."))
    b = dict(normalized_units("We evaluate a sentence and two sentences."))
    assert "sentence" in a
    assert a["sentence"] == "sentence"
    assert "sentence" in b  # "a sentence" and "two sentences" share the key


def test_copulas_light_verbs_and_stop_words_are_not_units() -> None:
    keys = [k for k, _ in normalized_units("It is clear that this was shown to be the case.")]
    assert "be" not in keys
    assert "show" not in keys
    assert "case" not in keys  # generic academic word on its own


def test_cited_names_are_dropped() -> None:
    keys = [k for k, _ in normalized_units("Musan discusses the past tense.", drop_names={"Musan"})]
    assert "musan" not in keys
    assert "past tense" in keys


def test_short_abbreviations_and_numbering_dropped() -> None:
    keys = [k for k, _ in normalized_units("See A.3 and e.g. the results in 17b for details.")]
    assert all(k not in keys for k in ("a.3", "e.g.", "17b"))


def test_strip_citations() -> None:
    s = strip_citations("Lifetime effects (Musan 1997; Kratzer 1995) are defeasible [12].")
    assert s == "Lifetime effects are defeasible."


def test_cited_names_from_text_and_reference_list() -> None:
    lines = [
        "As McCloskey (2002) shows, ...",
        "References",
        "Lepikhin, Melvin Johnson, and others. 2020.",
    ]
    names = cited_names(lines)
    assert {"McCloskey", "Lepikhin"} <= names


def test_clean_article_text_cuts_references_but_keeps_appendix() -> None:
    lines = (
        ["Abstract"]
        + [f"body {i}" for i in range(10)]
        + ["References", "Smith, J. 2000.", "Appendix", "A.1 Setup", "more"]
    )
    kept = clean_article_text(lines)
    assert "Smith, J. 2000." not in kept
    assert "A.1 Setup" in kept and "body 0" in kept
    assert "Abstract" not in kept  # front matter up to the heading is dropped


def test_is_foreign_line() -> None:
    lexicon = {"the", "door", "they", "say", "think", "key"}
    assert is_foreign_line("an doras deir siad mheasann sibh bhfuil eochair", lexicon)
    assert not is_foreign_line("the door they say they think the key", lexicon)


def test_merge_units_counts_variants_and_contexts() -> None:
    occ = [
        {
            "article_id": "a",
            "sent_idx": 0,
            "key": "sentence",
            "surface": "sentence",
            "parent_sentence": "S0",
        },
        {
            "article_id": "a",
            "sent_idx": 3,
            "key": "sentence",
            "surface": "sentences",
            "parent_sentence": "S3",
        },
        {
            "article_id": "a",
            "sent_idx": 3,
            "key": "live",
            "surface": "living",
            "parent_sentence": "S3",
        },
        {
            "article_id": "a",
            "sent_idx": 4,
            "key": "living",
            "surface": "living",
            "parent_sentence": "S4",
        },
    ]
    merged = {m["key"]: m for m in merge_units(occ)}
    assert merged["sentence"]["count"] == 2
    assert merged["sentence"]["variants"] == ["sentence", "sentences"]
    assert [c["sent_idx"] for c in merged["sentence"]["contexts"]] == [0, 3]
    # "living" (noun) and "live" (verb lemma) share a surface form -> one unit
    assert len([m for m in merged.values() if m["text"] == "living"]) == 1
