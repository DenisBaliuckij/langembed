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


def test_load_cache_missing_file_returns_empty(tmp_path):
    assert load_cache(tmp_path / "does_not_exist.jsonl") == {}


def test_append_and_load_cache_round_trip(tmp_path):
    path = tmp_path / "cache.jsonl"
    append_cache(path, "k1", "v1")
    append_cache(path, "k2", "v2")

    assert load_cache(path) == {"k1": "v1", "k2": "v2"}
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0]) == {"key": "k1", "value": "v1"}
