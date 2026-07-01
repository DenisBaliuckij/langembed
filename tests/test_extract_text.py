"""Unit tests for the PDF-to-sentences extraction module (ru track, Phase 1)."""

from __future__ import annotations

from langembed.data.extract_text import split_sentences


def test_split_sentences_basic() -> None:
    text = "Это первое предложение книги. Это второе предложение книги."
    result = split_sentences(text)
    assert result == [
        "Это первое предложение книги.",
        "Это второе предложение книги.",
    ]


def test_split_sentences_quotation_marks() -> None:
    text = "Он сказал ей тихо: «Пойдём скорее домой». Она согласилась молча и тихо."
    result = split_sentences(text)
    assert result[0] == "Он сказал ей тихо: «Пойдём скорее домой»."
    assert result[1] == "Она согласилась молча и тихо."


def test_split_sentences_multiple_terminators() -> None:
    text = "Неужели это была правда?! Никто не мог поверить в случившееся."
    result = split_sentences(text)
    assert result[0] == "Неужели это была правда?!"
    assert result[1] == "Никто не мог поверить в случившееся."


def test_split_sentences_filters_junk_and_short_fragments() -> None:
    text = "42\nЭто нормальное предложение книги, длиннее лимита символов."
    result = split_sentences(text)
    assert result == ["Это нормальное предложение книги, длиннее лимита символов."]


def test_split_sentences_dehyphenates_wrapped_words() -> None:
    text = "Это слово было перене-\nсено на новую строку книги, вот так вот."
    result = split_sentences(text)
    assert result == ["Это слово было перенесено на новую строку книги, вот так вот."]
