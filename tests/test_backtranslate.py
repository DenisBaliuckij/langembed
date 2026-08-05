import json

import pytest

pytest.importorskip("deep_translator")

from langembed.data.backtranslate import (  # noqa: E402
    append_cache,
    back_translate,
    load_cache,
)


class _StepTranslator:
    """Fake deep_translator backend: returns queued responses in order, records calls."""

    _responses: list[str] = []
    _calls: list[str] = []

    def __init__(self, source=None, target=None):
        self.source = source
        self.target = target

    def translate(self, text: str) -> str:
        self._calls.append(text)
        return self._responses.pop(0)


class _AlwaysFails:
    def __init__(self, source=None, target=None):
        pass

    def translate(self, text: str) -> str:
        raise RuntimeError("provider unavailable")


def test_back_translate_round_trip(monkeypatch, tmp_path):
    import deep_translator

    _StepTranslator._responses = ["hello there", "hi there"]
    _StepTranslator._calls = []
    monkeypatch.setattr(deep_translator, "GoogleTranslator", _StepTranslator)

    cache_path = tmp_path / "cache.jsonl"
    result = back_translate("привет", ["google"], "en", "ru", {}, cache_path)

    assert result == "hi there"
    assert _StepTranslator._calls == ["привет", "hello there"]


def test_back_translate_caches_to_disk(monkeypatch, tmp_path):
    import deep_translator

    _StepTranslator._responses = ["hello there", "hi there"]
    _StepTranslator._calls = []
    monkeypatch.setattr(deep_translator, "GoogleTranslator", _StepTranslator)

    cache_path = tmp_path / "cache.jsonl"
    cache: dict[str, str] = {}
    back_translate("привет", ["google"], "en", "ru", cache, cache_path)

    assert len(cache) == 1
    reloaded = load_cache(cache_path)
    assert reloaded == cache


def test_back_translate_cache_hit_skips_network(monkeypatch, tmp_path):
    import deep_translator

    monkeypatch.setattr(deep_translator, "GoogleTranslator", _AlwaysFails)

    cache_path = tmp_path / "cache.jsonl"

    # Manually construct the exact key back_translate would compute, so the
    # cache lookup hits without ever calling the (always-failing) provider.
    from langembed.data.backtranslate import _cache_key

    key = _cache_key("привет", "google", "en", "ru")
    prewarmed: dict[str, str] = {key: "cached value"}

    result = back_translate("привет", ["google"], "en", "ru", prewarmed, cache_path)
    assert result == "cached value"


def test_back_translate_provider_fallback(monkeypatch, tmp_path):
    import deep_translator

    monkeypatch.setattr(deep_translator, "GoogleTranslator", _AlwaysFails)
    _StepTranslator._responses = ["hello there", "hi there"]
    _StepTranslator._calls = []
    monkeypatch.setattr(deep_translator, "MyMemoryTranslator", _StepTranslator)

    cache_path = tmp_path / "cache.jsonl"
    result = back_translate("привет", ["google", "mymemory"], "en", "ru", {}, cache_path)

    assert result == "hi there"


def test_back_translate_all_providers_fail_returns_none(monkeypatch, tmp_path):
    import deep_translator

    monkeypatch.setattr(deep_translator, "GoogleTranslator", _AlwaysFails)
    monkeypatch.setattr(deep_translator, "MyMemoryTranslator", _AlwaysFails)

    cache_path = tmp_path / "cache.jsonl"
    result = back_translate(
        "привет", ["google", "mymemory"], "en", "ru", {}, cache_path, max_retries=0
    )

    assert result is None


def test_back_translate_propagates_value_error(monkeypatch, tmp_path):
    """A bad provider name (ValueError from _translate_one) is a config error, not a
    transient failure -- it must propagate instead of being retried and swallowed into
    a silent None (see finding: silent metric corruption)."""
    import langembed.data.backtranslate as backtranslate_module

    def _raise_value_error(text, provider, source_lang, target_lang):
        raise ValueError(f"unknown translation provider: {provider!r}")

    monkeypatch.setattr(backtranslate_module, "_translate_one", _raise_value_error)

    cache_path = tmp_path / "cache.jsonl"
    with pytest.raises(ValueError):
        back_translate("привет", ["bogus"], "en", "ru", {}, cache_path, max_retries=2)


def test_back_translate_propagates_import_error(monkeypatch, tmp_path):
    """Missing deep_translator (ImportError) must propagate immediately rather than
    being retried/swallowed by the transient-failure retry loop."""
    import langembed.data.backtranslate as backtranslate_module

    def _raise_import_error(text, provider, source_lang, target_lang):
        raise ImportError("No module named 'deep_translator'")

    monkeypatch.setattr(backtranslate_module, "_translate_one", _raise_import_error)

    cache_path = tmp_path / "cache.jsonl"
    with pytest.raises(ImportError):
        back_translate("привет", ["google"], "en", "ru", {}, cache_path, max_retries=2)


def test_back_translate_none_result_not_cached_and_retries(monkeypatch, tmp_path):
    """A provider returning a falsy result (e.g. None) for one attempt must not be cached
    -- it should be treated as a failure and retried, not permanently poison the cache."""
    import langembed.data.backtranslate as backtranslate_module

    calls = {"n": 0}

    def _flaky(text, provider, source_lang, target_lang):
        calls["n"] += 1
        # First round trip's second call returns None; second round trip succeeds.
        if calls["n"] == 2:
            return None
        return "ok"

    monkeypatch.setattr(backtranslate_module, "_translate_one", _flaky)

    cache_path = tmp_path / "cache.jsonl"
    cache: dict[str, str] = {}
    result = back_translate("привет", ["google"], "en", "ru", cache, cache_path, max_retries=1)

    assert result == "ok"
    assert len(cache) == 1
    assert None not in cache.values()
    # Re-running with a fresh call counter proves the cache holds the real value, not None.
    assert load_cache(cache_path) == cache


def test_back_translate_splits_delay_across_both_calls(monkeypatch, tmp_path):
    """--translate-rpm politeness delay must be spent across both real API calls in a
    round trip (source->pivot, pivot->source), not slept once as a single lump after
    both calls have already fired back-to-back."""
    import langembed.data.backtranslate as backtranslate_module

    sleeps: list[float] = []
    monkeypatch.setattr(backtranslate_module.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(
        backtranslate_module,
        "_translate_one",
        lambda text, provider, source_lang, target_lang: "ok",
    )

    cache_path = tmp_path / "cache.jsonl"
    back_translate("привет", ["google"], "en", "ru", {}, cache_path, delay=3.0)

    assert sleeps == [1.5, 1.5]


def test_load_cache_missing_file_returns_empty(tmp_path):
    assert load_cache(tmp_path / "does_not_exist.jsonl") == {}


def test_append_and_load_cache_round_trip(tmp_path):
    path = tmp_path / "cache.jsonl"
    append_cache(path, "k1", "v1")
    append_cache(path, "k2", "v2")

    assert load_cache(path) == {"k1": "v1", "k2": "v2"}
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0]) == {"key": "k1", "value": "v1"}
