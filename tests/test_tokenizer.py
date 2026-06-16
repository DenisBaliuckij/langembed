# tests/test_tokenizer.py
"""Phase 2 tokenizer tests: round-trip and unk_rate."""
from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("tokenizers")

_FIXTURE = """\
The cat sat on the mat.
A quick brown fox jumps over the lazy dog.
Machine learning enables computers to learn from data.
Natural language processing helps computers understand human text.
The sun rises in the east and sets in the west.
She baked a delicious chocolate cake for the party.
Engineers write software to solve real world problems.
Athletes train every day to improve their performance.
Dark clouds gathered on the horizon before the storm.
The river flows quietly through the green valley.
Students read books to expand their knowledge and skills.
The scientist conducted experiments in the laboratory.
Children played happily in the park on a sunny afternoon.
Music has the power to evoke strong emotions in people.
The economy grows when people invest in new businesses.
Doctors work long hours to care for their patients.
The library contains thousands of books on many topics.
Farmers harvest their crops before the winter arrives.
The chef prepared a meal using fresh local ingredients.
Travelers explore new countries to experience different cultures.\
"""


def test_round_trip(tmp_path: pathlib.Path) -> None:
    from langembed.preprocess import normalize
    from langembed.tokenizer.train_tokenizer import train_tokenizer

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(_FIXTURE, encoding="utf-8")
    out = str(tmp_path / "tok")
    tok = train_tokenizer(str(corpus), out, vocab_size=300, min_frequency=1)

    for raw in _FIXTURE.splitlines():
        normed = normalize(raw.strip())
        if not normed:
            continue
        ids = tok(normed)["input_ids"]
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert len(ids) > 0, f"no tokens generated for: {normed!r}"
        assert len(decoded) > 0, f"empty decode for: {normed!r} -> ids {ids}"
        # BPE with Whitespace() pre-tokenizer may insert spaces between subword tokens;
        # verify all characters are preserved in order, whitespace-only differences accepted.
        assert decoded.replace(" ", "") == normed.replace(" ", ""), \
            f"round-trip character loss: {normed!r} -> {decoded!r}"


def test_unk_rate_below_threshold(tmp_path: pathlib.Path) -> None:
    from langembed.tokenizer.train_tokenizer import diagnose, train_tokenizer

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(_FIXTURE, encoding="utf-8")
    out = str(tmp_path / "tok")
    tok = train_tokenizer(str(corpus), out, vocab_size=300, min_frequency=1)
    stats = diagnose(tok, str(corpus))
    assert stats["unk_rate"] < 0.01, f"unk_rate too high: {stats['unk_rate']}"
