import json

from langembed.annotation import auto_label


def _fake_paraphrase(*args, **kwargs):
    return "PARA:" + args[0]


def test_build_auto_sts_pairs_three_tiers(monkeypatch, tmp_path):
    monkeypatch.setattr(auto_label, "back_translate", _fake_paraphrase)
    sentences = [f"sentence {i}" for i in range(12)]

    pairs = auto_label.build_auto_sts_pairs(
        sentences,
        n=9,
        providers=["google"],
        pivot_lang="en",
        source_lang="ru",
        cache_path=tmp_path / "cache.jsonl",
        seed=1,
    )

    assert len(pairs) == 9
    scores = {p[2] for p in pairs}
    assert scores == {
        auto_label.PARAPHRASE_SCORE,
        auto_label.ADJACENT_SCORE,
        auto_label.RANDOM_SCORE,
    }


def test_build_auto_sts_pairs_drops_failed_paraphrases(monkeypatch, tmp_path):
    monkeypatch.setattr(auto_label, "back_translate", lambda *a, **k: None)
    sentences = [f"sentence {i}" for i in range(12)]

    pairs = auto_label.build_auto_sts_pairs(
        sentences,
        n=9,
        providers=["google"],
        pivot_lang="en",
        source_lang="ru",
        cache_path=tmp_path / "cache.jsonl",
        seed=1,
    )

    assert auto_label.PARAPHRASE_SCORE not in {p[2] for p in pairs}
    assert len(pairs) > 0


def test_build_auto_sts_pairs_warns_on_total_paraphrase_failure(monkeypatch, tmp_path, capsys):
    """A total translation outage (every anchor fails) must print a visible warning to
    stderr instead of silently shrinking the paraphrase tier to zero unnoticed."""
    monkeypatch.setattr(auto_label, "back_translate", lambda *a, **k: None)
    sentences = [f"sentence {i}" for i in range(12)]

    auto_label.build_auto_sts_pairs(
        sentences,
        n=9,
        providers=["google"],
        pivot_lang="en",
        source_lang="ru",
        cache_path=tmp_path / "cache.jsonl",
        seed=1,
    )

    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "paraphrase" in captured.err


def test_build_auto_sts_pairs_no_warning_when_paraphrases_succeed(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(auto_label, "back_translate", _fake_paraphrase)
    sentences = [f"sentence {i}" for i in range(12)]

    auto_label.build_auto_sts_pairs(
        sentences,
        n=9,
        providers=["google"],
        pivot_lang="en",
        source_lang="ru",
        cache_path=tmp_path / "cache.jsonl",
        seed=1,
    )

    captured = capsys.readouterr()
    assert "WARNING" not in captured.err


def test_build_auto_sts_pairs_deterministic(monkeypatch, tmp_path):
    monkeypatch.setattr(auto_label, "back_translate", _fake_paraphrase)
    sentences = [f"sentence {i}" for i in range(12)]
    kwargs = dict(
        sentences=sentences,
        n=9,
        providers=["google"],
        pivot_lang="en",
        source_lang="ru",
        cache_path=tmp_path / "cache.jsonl",
        seed=7,
    )

    assert auto_label.build_auto_sts_pairs(**kwargs) == auto_label.build_auto_sts_pairs(**kwargs)


def test_build_auto_sts_pairs_too_few_sentences_returns_empty(tmp_path):
    pairs = auto_label.build_auto_sts_pairs(
        ["only one sentence"],
        n=9,
        providers=["google"],
        pivot_lang="en",
        source_lang="ru",
        cache_path=tmp_path / "cache.jsonl",
    )
    assert pairs == []


def test_write_sts_pairs_schema(tmp_path):
    out = tmp_path / "sts.jsonl"
    n = auto_label.write_sts_pairs([("a", "b", 5.0)], out)

    assert n == 1
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row == {"sentence_a": "a", "sentence_b": "b", "score": 5.0}
