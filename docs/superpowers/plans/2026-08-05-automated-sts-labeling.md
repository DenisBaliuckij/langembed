# Automated STS Labeling via Back-Translation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `scripts/run_pipeline.py` run start-to-finish with zero human interaction by adding an opt-in `--auto-label` flag that generates silver-standard STS pairs via free-provider back-translation instead of blocking on the manual `/label` web form.

**Architecture:** Two new leaf modules — `langembed/data/backtranslate.py` (round-trip translation with caching, retries, and multi-provider fallback) and `langembed/annotation/auto_label.py` (builds three-tier silver STS pairs: back-translated paraphrase / adjacent-sentence / random-sentence) — plus a small extraction inside `scripts/run_pipeline.py` (`generate_auto_sts()`) that wires the new modules into step 5, branching on a new `--auto-label` CLI flag. The existing manual-labeling branch is untouched.

**Tech Stack:** Python 3.11, `deep-translator` (new optional dependency, `GoogleTranslator` + `MyMemoryTranslator` backends — free, keyless), `pytest` with `monkeypatch` and `tmp_path`.

## Global Constraints

- Line length 100 (`ruff` + project convention); `ruff check`/`ruff format --check`/`mypy` must stay green (`make lint`).
- Heavy/optional imports (here: `deep_translator`) go inside function bodies, never at module top level — see `CLAUDE.md` "Тяжёлые зависимости".
- No hardcoded hyperparameters outside function defaults that are themselves CLI-overridable; this feature has no `configs/*.yaml` surface by design (see spec, "special configuration... at job startup" = CLI flags on `run_pipeline.py`, matching existing `--skip-eval`/`--spacy-model` precedent).
- Manual labeling stays the default behavior of `run_pipeline.py`; `--auto-label` must be explicitly passed to opt in.
- `write_sts_pairs` output schema must exactly match `annotation/api.py::export_sts`'s JSONL schema: `{"sentence_a": str, "sentence_b": str, "score": float}` — `evaluate.py` reads this format unchanged.
- Tests that need `deep_translator` installed use `pytest.importorskip("deep_translator")` (matches existing `pytest.importorskip("datasketch")` / `pytest.importorskip("spacy")` pattern in `tests/test_build_corpus.py`); tests that only need `auto_label`'s pure logic mock `back_translate` directly and need no such skip.
- One git commit per task, message style `feat(<scope>): <summary>`, matching this repo's history (`feat(ru): ...`, `feat(serve): ...`).

---

### Task 1: `langembed/data/backtranslate.py` — cached, multi-provider round-trip translation

**Files:**
- Create: `src/langembed/data/backtranslate.py`
- Modify: `pyproject.toml` (add `translate` optional-dependency group)
- Test: `tests/test_backtranslate.py`

**Interfaces:**
- Consumes: nothing from this codebase (leaf module); `deep_translator.GoogleTranslator` / `deep_translator.MyMemoryTranslator` (both constructed as `Cls(source=..., target=...)` with a `.translate(text: str) -> str` method).
- Produces (used by Task 2):
  - `load_cache(path: str | Path) -> dict[str, str]`
  - `back_translate(text: str, providers: Sequence[str], pivot_lang: str, source_lang: str, cache: dict[str, str], cache_path: str | Path, max_retries: int = 2, delay: float = 0.0) -> str | None`

- [ ] **Step 1: Add the `translate` optional-dependency group**

Edit `pyproject.toml`, immediately after the existing `dev = [...]` block (after line 51, before `[tool.setuptools.packages.find]`):

```toml
translate = [
  "deep-translator>=1.11",
]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_backtranslate.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_backtranslate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'langembed.data.backtranslate'` (or `ImportError`). If `deep-translator` is not installed in the dev environment yet, install it first: `pip install -e ".[translate]"` — the tests should then fail on the import above, not on the `importorskip`.

- [ ] **Step 4: Implement `src/langembed/data/backtranslate.py`**

```python
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
    for line in p.open(encoding="utf-8"):
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
    """
    for provider in providers:
        key = _cache_key(text, provider, pivot_lang, source_lang)
        if key in cache:
            return cache[key]
        for _attempt in range(max_retries + 1):
            try:
                pivot_text = _translate_one(text, provider, source_lang, pivot_lang)
                back = _translate_one(pivot_text, provider, pivot_lang, source_lang)
            except Exception:
                if delay:
                    time.sleep(delay)
                continue
            cache[key] = back
            append_cache(cache_path, key, back)
            if delay:
                time.sleep(delay)
            return back
    return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_backtranslate.py -v`
Expected: PASS (8 tests)

- [ ] **Step 6: Lint**

Run: `ruff check src/langembed/data/backtranslate.py tests/test_backtranslate.py && ruff format --check src/langembed/data/backtranslate.py tests/test_backtranslate.py && mypy src/langembed/data/backtranslate.py`
Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/langembed/data/backtranslate.py tests/test_backtranslate.py
git commit -m "feat(data): cached multi-provider back-translation"
```

---

### Task 2: `langembed/annotation/auto_label.py` — silver STS pair generation

**Files:**
- Create: `src/langembed/annotation/auto_label.py`
- Test: `tests/test_auto_label.py`

**Interfaces:**
- Consumes: `langembed.data.backtranslate.back_translate` (Task 1, signature above), `langembed.data.backtranslate.load_cache`.
- Produces (used by Task 3):
  - `build_auto_sts_pairs(sentences: list[str], n: int, providers: list[str], pivot_lang: str, source_lang: str, cache_path: str | Path, requests_per_minute: float = 20.0, seed: int = 42) -> list[tuple[str, str, float]]`
  - `write_sts_pairs(pairs: list[tuple[str, str, float]], out_path: str | Path) -> int`
  - Module constants: `PARAPHRASE_SCORE = 4.8`, `ADJACENT_SCORE = 2.0`, `RANDOM_SCORE = 0.3`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_auto_label.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_auto_label.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'langembed.annotation.auto_label'`

- [ ] **Step 3: Implement `src/langembed/annotation/auto_label.py`**

```python
"""Silver-standard STS pair generation via back-translation — no human labeler needed."""

from __future__ import annotations

import json
import random
from pathlib import Path

from langembed.data.backtranslate import back_translate, load_cache

PARAPHRASE_SCORE = 4.8
ADJACENT_SCORE = 2.0
RANDOM_SCORE = 0.3


def build_auto_sts_pairs(
    sentences: list[str],
    n: int,
    providers: list[str],
    pivot_lang: str,
    source_lang: str,
    cache_path: str | Path,
    requests_per_minute: float = 20.0,
    seed: int = 42,
) -> list[tuple[str, str, float]]:
    """Silver STS pairs in three tiers, evenly split across `n`: back-translated
    paraphrases (high similarity), adjacent corpus sentences (mid similarity), and
    random distant sentence pairs (low similarity). Pairs where every translation
    provider fails are dropped, not padded, so the paraphrase tier may end up smaller
    than the other two.
    """
    if len(sentences) < 2:
        return []

    rng = random.Random(seed)
    cache = load_cache(cache_path)
    delay = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
    n_each = max(1, n // 3)

    pairs: list[tuple[str, str, float]] = []

    anchors = rng.sample(sentences, min(n_each, len(sentences)))
    for s in anchors:
        para = back_translate(s, providers, pivot_lang, source_lang, cache, cache_path, delay=delay)
        if para and para != s:
            pairs.append((s, para, PARAPHRASE_SCORE))

    adjacent_idx = list(range(len(sentences) - 1))
    for i in rng.sample(adjacent_idx, min(n_each, len(adjacent_idx))):
        pairs.append((sentences[i], sentences[i + 1], ADJACENT_SCORE))

    for _ in range(n_each):
        a, b = rng.sample(range(len(sentences)), 2)
        pairs.append((sentences[a], sentences[b], RANDOM_SCORE))

    rng.shuffle(pairs)
    return pairs[:n]


def write_sts_pairs(pairs: list[tuple[str, str, float]], out_path: str | Path) -> int:
    """Write (sentence_a, sentence_b, score) triples as STS-test JSONL, matching
    `annotation.api.export_sts`'s schema exactly."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for a, b, score in pairs:
            f.write(
                json.dumps({"sentence_a": a, "sentence_b": b, "score": score}, ensure_ascii=False)
                + "\n"
            )
    return len(pairs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_auto_label.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Lint**

Run: `ruff check src/langembed/annotation/auto_label.py tests/test_auto_label.py && ruff format --check src/langembed/annotation/auto_label.py tests/test_auto_label.py && mypy src/langembed/annotation/auto_label.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/langembed/annotation/auto_label.py tests/test_auto_label.py
git commit -m "feat(annotation): auto-generate silver STS pairs via back-translation"
```

---

### Task 3: Wire `--auto-label` into `scripts/run_pipeline.py`

**Files:**
- Modify: `scripts/run_pipeline.py`
- Test: `tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: `langembed.annotation.auto_label.build_auto_sts_pairs`, `langembed.annotation.auto_label.write_sts_pairs` (Task 2).
- Produces: new module-level function `generate_auto_sts(corpus_path: str, sts_test_path: str, lang: str, providers: list[str], pivot_lang: str, requests_per_minute: float, n_labels: int) -> int` in `scripts/run_pipeline.py`; new CLI flags `--auto-label`, `--translate-providers`, `--pivot-lang`, `--translate-rpm`.

Note on testing approach: `main()` in this file already runs a long, unmocked subprocess pipeline (extraction, tokenizer, pretrain, ...) with no existing test coverage — there is no `tests/test_run_pipeline.py` today. Rather than mock that entire chain, this task extracts the auto-label branch's logic into a standalone function (`generate_auto_sts`) that takes plain arguments and has no docker/server/`input()` dependency by construction, and tests that function directly. This is fast, isolated, and gives real coverage of the new logic without inventing subprocess mocks for unrelated, pre-existing code.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_pipeline.py`:

```python
import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_pipeline", Path(__file__).resolve().parent.parent / "scripts" / "run_pipeline.py"
)
assert _SPEC is not None and _SPEC.loader is not None
run_pipeline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_pipeline)


def test_generate_auto_sts_writes_pairs(monkeypatch, tmp_path):
    from langembed.annotation import auto_label

    monkeypatch.setattr(auto_label, "back_translate", lambda *a, **k: "PARA:" + a[0])

    corpus = tmp_path / "corpus_ru.txt"
    corpus.write_text("\n".join(f"sentence {i}" for i in range(12)), encoding="utf-8")
    sts_out = tmp_path / "sts_test_ru.jsonl"

    n = run_pipeline.generate_auto_sts(str(corpus), str(sts_out), "ru", ["google"], "en", 1000.0, 9)

    assert n == 9
    lines = sts_out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 9
    row = json.loads(lines[0])
    assert set(row.keys()) == {"sentence_a", "sentence_b", "score"}


def test_generate_auto_sts_uses_per_language_cache_path(monkeypatch, tmp_path):
    from langembed.annotation import auto_label

    seen_cache_paths = []

    def fake_build(*args, **kwargs):
        seen_cache_paths.append(kwargs["cache_path"])
        return []

    monkeypatch.setattr(auto_label, "build_auto_sts_pairs", fake_build)

    corpus = tmp_path / "corpus_fr.txt"
    corpus.write_text("a\nb\n", encoding="utf-8")
    sts_out = tmp_path / "sts_test_fr.jsonl"

    run_pipeline.generate_auto_sts(str(corpus), str(sts_out), "fr", ["google"], "en", 20.0, 6)

    assert seen_cache_paths == ["data/backtranslation_cache_fr.jsonl"]


def test_auto_label_cli_flag_defaults():
    ap = run_pipeline.build_arg_parser()
    args = ap.parse_args(["--lang", "ru", "--input", "book.pdf"])

    assert args.auto_label is False
    assert args.translate_providers == ["google", "mymemory"]
    assert args.pivot_lang == "en"
    assert args.translate_rpm == 20.0


def test_auto_label_cli_flag_set():
    ap = run_pipeline.build_arg_parser()
    args = ap.parse_args(
        ["--lang", "ru", "--input", "book.pdf", "--auto-label", "--pivot-lang", "hi"]
    )

    assert args.auto_label is True
    assert args.pivot_lang == "hi"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_run_pipeline.py -v`
Expected: FAIL — `AttributeError: module 'run_pipeline' has no attribute 'generate_auto_sts'` (and `build_arg_parser` also does not exist yet)

- [ ] **Step 3: Extract `build_arg_parser()` and add `generate_auto_sts()` in `scripts/run_pipeline.py`**

`main()` currently builds its `argparse.ArgumentParser` inline (lines 109-139 in the current file) and reads `args.input` name that would collide with a name inside `build_arg_parser`. Factor the parser construction out into its own function so the new CLI flags are unit-testable without invoking `main()`'s subprocess chain, and add the `generate_auto_sts` helper.

Replace the start of `main()`:

```python
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lang", required=True, help="language code, e.g. ru, fr, de")
    ap.add_argument(
        "--input", nargs="+", required=True, type=Path, help="one or more PDF corpus files"
    )
    ap.add_argument(
        "--output", default=Path("output"), type=Path, help="final deliverable directory"
    )
    ap.add_argument(
        "--n-labels", type=int, default=60, help="STS candidate pairs to seed for labeling"
    )
    ap.add_argument(
        "--pretrain-minutes", type=float, default=25.0, help="target MLM pretrain wall time"
    )
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument(
        "--spacy-model",
        default=None,
        help=(
            "spaCy model for text preparation (lemmatize + POS-token "
            "substitution), e.g. ru_core_news_sm; omit to skip"
        ),
    )
    ap.add_argument(
        "--skip-eval",
        action="store_true",
        help="skip labeling/eval; only produce corpus, encoder, SimCSE model and embeddings",
    )
    args = ap.parse_args()
```

with:

```python
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lang", required=True, help="language code, e.g. ru, fr, de")
    ap.add_argument(
        "--input", nargs="+", required=True, type=Path, help="one or more PDF corpus files"
    )
    ap.add_argument(
        "--output", default=Path("output"), type=Path, help="final deliverable directory"
    )
    ap.add_argument(
        "--n-labels", type=int, default=60, help="STS candidate pairs to seed for labeling"
    )
    ap.add_argument(
        "--pretrain-minutes", type=float, default=25.0, help="target MLM pretrain wall time"
    )
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument(
        "--spacy-model",
        default=None,
        help=(
            "spaCy model for text preparation (lemmatize + POS-token "
            "substitution), e.g. ru_core_news_sm; omit to skip"
        ),
    )
    ap.add_argument(
        "--skip-eval",
        action="store_true",
        help="skip labeling/eval; only produce corpus, encoder, SimCSE model and embeddings",
    )
    ap.add_argument(
        "--auto-label",
        action="store_true",
        help=(
            "skip the manual /label step; generate silver STS pairs via "
            "back-translation instead (no human, no docker/postgres needed for this step)"
        ),
    )
    ap.add_argument(
        "--translate-providers",
        nargs="+",
        default=["google", "mymemory"],
        help="free translation backends for back-translation (deep-translator provider names)",
    )
    ap.add_argument(
        "--pivot-lang", default="en", help="pivot language for the back-translation round-trip"
    )
    ap.add_argument(
        "--translate-rpm",
        type=float,
        default=20.0,
        help="max back-translation requests/minute (politeness limit for free MT APIs)",
    )
    return ap


def generate_auto_sts(
    corpus_path: str,
    sts_test_path: str,
    lang: str,
    providers: list[str],
    pivot_lang: str,
    requests_per_minute: float,
    n_labels: int,
) -> int:
    """Auto-label branch of pipeline step 5: build silver STS pairs via back-translation
    and write them to `sts_test_path`. Returns the number of pairs written. Unlike the
    manual-labeling branch, this has no docker/server/human-input dependency.
    """
    from langembed.annotation.auto_label import build_auto_sts_pairs, write_sts_pairs

    with open(corpus_path, encoding="utf-8") as f:
        sentences = [ln.strip() for ln in f if ln.strip()]
    cache_path = f"data/backtranslation_cache_{lang}.jsonl"
    pairs = build_auto_sts_pairs(
        sentences,
        n=n_labels,
        providers=providers,
        pivot_lang=pivot_lang,
        source_lang=lang,
        cache_path=cache_path,
        requests_per_minute=requests_per_minute,
    )
    return write_sts_pairs(pairs, sts_test_path)


def main() -> None:
    args = build_arg_parser().parse_args()
```

- [ ] **Step 4: Branch step 5 on `args.auto_label`**

In `main()`, replace the `if not args.skip_eval:` block (the human-labeling step, currently running unconditionally inside that `if`) so the existing body moves into an `else`, with a new `if args.auto_label:` branch before it, and the trailing `eval_cfg` block (already present, unchanged) stays common to both branches:

```python
    if not args.skip_eval:
        if args.auto_label:
            print(f"=== [{lang}] 5/6 auto-label STS pairs (back-translation, no human) ===")
            n_written = generate_auto_sts(
                corpus_path,
                sts_test_path,
                lang,
                args.translate_providers,
                args.pivot_lang,
                args.translate_rpm,
                args.n_labels,
            )
            print(f"  wrote {n_written} auto-labeled STS pairs -> {sts_test_path}")
        else:
            print(f"=== [{lang}] 5/6 human-in-the-loop STS labeling + eval ===")
            run(["docker", "compose", "up", "-d", "postgres"])
            run(
                [
                    sys.executable,
                    "scripts/seed_sts_pairs.py",
                    "--config",
                    str(contrastive_path),
                    "--n",
                    str(args.n_labels),
                ]
            )

            label_port = free_port(8001)
            server = start_server("langembed.annotation.api:app", label_port)
            try:
                input(
                    f"\nLabel pairs at http://localhost:{label_port}/label (rate 1-5), "
                    "then press Enter here to continue...\n"
                )
                import httpx

                resp = httpx.get(
                    f"http://localhost:{label_port}/export-sts",
                    params={"out_path": sts_test_path},
                    timeout=30,
                )
                resp.raise_for_status()
                print(" ", resp.json())
            finally:
                stop_server(server)

        eval_cfg = {
            "language": lang,
            "spacy_model": args.spacy_model,
            "test_path": sts_test_path,
            "score_scale": 5.0,
            "retrieval_k": 5,
            "branches": {"A": f"artifacts/simcse_{lang}"},
            "train_paths": [],
            "metrics_path": f"metrics/eval_{lang}.json",
        }
        eval_path = cfg_dir / "eval.yaml"
        write_yaml(eval_path, eval_cfg)
        run([sys.executable, "-m", "langembed.eval.evaluate", "--config", str(eval_path)])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_run_pipeline.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Full test suite + lint**

Run: `pytest tests/ -v --ignore=tests/e2e` then `ruff check src scripts tests && ruff format --check src scripts tests && mypy src`
Expected: all green, no new failures in previously-passing tests (`tests/test_backtranslate.py`, `tests/test_auto_label.py` from Tasks 1-2 included)

- [ ] **Step 7: Commit**

```bash
git add scripts/run_pipeline.py tests/test_run_pipeline.py
git commit -m "feat(pipeline): add --auto-label flag for unattended STS labeling"
```

---

## Definition of Done

- [ ] `make lint` green.
- [ ] `pytest tests/ --ignore=tests/e2e` green, including the three new test files.
- [ ] `python scripts/run_pipeline.py --lang <lang> --input <pdf> --auto-label` reaches step 6 without any `docker compose`, `input()`, or annotation-server call in step 5.
- [ ] Manual labeling (flag omitted) behaves byte-for-byte as before — verified by reading the diff: the `else` branch is the original code, unmodified.
