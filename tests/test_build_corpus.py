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
