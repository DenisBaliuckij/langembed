import json

import pytest

pytest.importorskip("datasketch")

from langembed.data.build_corpus import build_corpus, load_test_hashes  # noqa: E402


def test_guard_raises_on_leakage(tmp_path):
    raw = tmp_path / "raw.txt"
    raw.write_text("hello world\nfoo bar baz\n", encoding="utf-8")
    test = tmp_path / "test.jsonl"
    test.write_text(
        json.dumps({"sentence_a": "hello world", "sentence_b": "x", "score": 5}) + "\n",
        encoding="utf-8",
    )
    th = load_test_hashes(str(test))
    with pytest.raises(RuntimeError):
        build_corpus([str(raw)], str(tmp_path / "out.txt"), th)


def test_writes_corpus(tmp_path):
    raw = tmp_path / "raw.txt"
    raw.write_text("alpha beta gamma\ndelta epsilon zeta\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    n = build_corpus([str(raw)], str(out), set())
    assert n == 2
    assert out.exists()


def test_leakage_guard_uses_spacy_model_consistently(tmp_path):
    """A corpus line and a test sentence that are identical before lemmatization must still
    collide as leaked even though lemmatization changes their surface form, proving _h() and
    build_corpus() hash with the same lang/spacy_model rather than silently defaulting to "gu".
    """
    pytest.importorskip("spacy")
    import spacy

    try:
        spacy.load("ru_core_news_sm", exclude=["ner", "parser"])
    except Exception:
        pytest.skip("ru_core_news_sm spaCy model not installed")

    raw = tmp_path / "raw.txt"
    raw.write_text("кошки бежали быстро\n", encoding="utf-8")
    test = tmp_path / "test.jsonl"
    test.write_text(
        json.dumps({"sentence_a": "кошки бежали быстро", "sentence_b": "x", "score": 5}) + "\n",
        encoding="utf-8",
    )
    th = load_test_hashes(str(test), "ru", "ru_core_news_sm")
    with pytest.raises(RuntimeError):
        build_corpus([str(raw)], str(tmp_path / "out.txt"), th, "ru", "ru_core_news_sm")


def test_leakage_guard_lang_default_backward_compatible(tmp_path):
    """Existing callers that don't pass lang/spacy_model keep working exactly as before."""
    raw = tmp_path / "raw.txt"
    raw.write_text("hello world\nfoo bar baz\n", encoding="utf-8")
    test = tmp_path / "test.jsonl"
    test.write_text(
        json.dumps({"sentence_a": "hello world", "sentence_b": "x", "score": 5}) + "\n",
        encoding="utf-8",
    )
    th = load_test_hashes(str(test))
    with pytest.raises(RuntimeError):
        build_corpus([str(raw)], str(tmp_path / "out.txt"), th)
