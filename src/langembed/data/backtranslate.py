"""Cached, multi-provider round-trip (back-)translation using free, keyless MT backends."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path


def _cache_key(text: str, provider: str, pivot_lang: str, source_lang: str) -> str:
    raw = f"{provider}|{source_lang}|{pivot_lang}|{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_cache(path: str | Path) -> dict[str, str]:
    """Load a JSONL cache file (`{"key": ..., "value": ...}` per line) into a dict."""
    p = Path(path)
    cache: dict[str, str] = {}
    if not p.exists():
        return cache
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[row["key"]] = row["value"]
    return cache


def append_cache(path: str | Path, key: str, value: str) -> None:
    """Append one cache entry to the JSONL cache file (creates parent dirs as needed)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def _translate_one(text: str, provider: str, source_lang: str, target_lang: str) -> str:
    if provider == "google":
        from deep_translator import GoogleTranslator

        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    if provider == "mymemory":
        from deep_translator import MyMemoryTranslator

        return MyMemoryTranslator(source=source_lang, target=target_lang).translate(text)
    raise ValueError(f"unknown translation provider: {provider!r}")


def back_translate(
    text: str,
    providers: Sequence[str],
    pivot_lang: str,
    source_lang: str,
    cache: dict[str, str],
    cache_path: str | Path,
    max_retries: int = 2,
    delay: float = 0.0,
) -> str | None:
    """Round-trip `text` through source_lang -> pivot_lang -> source_lang using the first
    provider in `providers` that succeeds. Returns None if every provider fails after
    `max_retries` retries each. Successful results are memoized into `cache` and appended
    to `cache_path` immediately, so a re-run of a partially-completed job skips
    already-translated text instead of re-spending free-tier quota.

    `ImportError` (missing `deep_translator`) and `ValueError` (unknown provider name) are
    programming/config errors, not transient network failures -- they propagate immediately
    instead of being retried and silently swallowed. Only genuinely transient errors (network,
    timeout, rate limiting, ...) are retried. A provider call that returns a falsy result
    (`None` or empty string) is treated as a failure and is never cached, so it can't
    permanently short-circuit future retries.

    `delay` is the target seconds-per-round-trip (derived from a requests-per-minute budget);
    it is split in half and slept after each of the two real `_translate_one` calls so the
    two outbound requests that make up one round trip are themselves spaced apart, rather than
    firing back-to-back.
    """
    half_delay = delay / 2 if delay else 0.0
    for provider in providers:
        key = _cache_key(text, provider, pivot_lang, source_lang)
        if key in cache:
            return cache[key]
        for _attempt in range(max_retries + 1):
            try:
                pivot_text = _translate_one(text, provider, source_lang, pivot_lang)
            except (ImportError, ValueError):
                raise
            except Exception:
                if delay:
                    time.sleep(delay)
                continue
            if half_delay:
                time.sleep(half_delay)
            try:
                back = _translate_one(pivot_text, provider, pivot_lang, source_lang)
            except (ImportError, ValueError):
                raise
            except Exception:
                if half_delay:
                    time.sleep(half_delay)
                continue
            if half_delay:
                time.sleep(half_delay)
            if not back:
                continue
            cache[key] = back
            append_cache(cache_path, key, back)
            return back
    return None
