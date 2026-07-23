# Generic Linguistic Text Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `preprocess.normalize()` with an opt-in, language-agnostic spaCy-based text-preparation stage (lemmatize + PROPN/PRON/NUM/abbreviation token substitution) matching the manual pipeline doc's "Предобработка" step, and fix two correctness gaps this change would otherwise expose.

**Architecture:** One new optional parameter (`spacy_model: str | None`) threaded through the existing single normalization entry point (`preprocess.normalize`) and every one of its three call sites (`build_corpus.py`, `evaluate.py`, `serve.py`), plus a matching `--spacy-model` flag in `scripts/run_pipeline.py`. No new call sites, no duplicated normalization logic. Two pre-existing correctness gaps found during design are fixed as part of this work: `_h()`/`load_test_hashes()` in `build_corpus.py` and `evaluate.py` never threaded `lang` through to `normalize()` (defaulting to `"gu"` regardless of the real language), and `evaluate.py` never normalized STS test sentences before scoring them.

**Tech Stack:** Python 3.11, spaCy `>=3.7` (new, `ml` extra), `ru_core_news_sm` spaCy model (downloaded separately, not a pip dependency), existing `pytest`/`ruff`/`mypy` tooling.

## Global Constraints

- **Единая нормализация**: all text preparation logic lives in `langembed.preprocess.normalize` — no duplicate logic in any other module. Every call site passes `lang` and (where applicable) `spacy_model` consistently.
- **No language-specific code paths**: the POS-substitution and abbreviation-detection logic uses only spaCy's Universal POS tagset and a plain string check — no `if lang == "ru"` branches anywhere in `preprocess.py` or elsewhere.
- **Idempotency**: `normalize(normalize(x, lang, spacy_model), lang, spacy_model) == normalize(x, lang, spacy_model)` must hold for every `lang`/`spacy_model` combination — verify explicitly in tests.
- **Backward compatibility**: every function whose signature changes (`normalize`, `_h`, `load_test_hashes`, `build_corpus`, `assert_no_leakage`) gets its new parameters appended with defaults (`lang: str = "gu"`, `spacy_model: str | None = None`) so existing call sites and existing tests that don't pass them keep working unmodified.
- **Style**: `ruff format` + `ruff check src tests`; `mypy src` clean; line length 100 (`[tool.ruff] line-length = 100`).
- **Graceful degradation**: a missing `spacy` install or undownloaded model must never raise — `normalize()` falls back to its pre-spaCy output, matching the existing `_indic_normalizer` pattern.
- **One commit per task**, per `CLAUDE.md`: run each task's acceptance commands before moving to the next task.

## Repo facts locked in during planning

- `spacy>=3.7` and the `ru_core_news_sm` model are already installed in this dev environment (verified: `python -m spacy download ru_core_news_sm` succeeded). `ru_core_news_sm`'s pipeline is `['tok2vec', 'morphologizer', 'attribute_ruler', 'lemmatizer']` — no `ner`/`parser` components exist, but `spacy.load(model, exclude=["ner", "parser"])` is a no-op for absent components (verified, does not raise).
- **Verified empirically** (see `docs/superpowers/specs/2026-07-23-text-preparation-design.md`'s correction note): `ru_core_news_sm` never sets `Abbr=Yes` morphology. Abbreviation detection uses a text-shape check instead: `"." in token.text and any(ch.isalpha() for ch in token.text)`. Verified against `"т.д. и т.п."` → `т.д.` tagged `PUNCT`, `"г. Москва и др. города"` → `г.` tagged `NOUN`, `и др.` tagged `PUNCT` — the text-shape check catches all three; POS-tag-based checks would not have caught `г.` or `и др.`.
- **Verified empirically**, exact input/output pairs for `ru_core_news_sm` (used verbatim in Task 1's tests):
  - `"кошки бежали быстро"` → `"кошка бежать быстро"`
  - `"она купила пять яблок"` → `"pron1 купить ordinal1 яблоко"`
  - `"Пушкин написал роман"` → `"person1 написать роман"`
  - `"г. Москва и др. города"` → `"abbr1 person1 abbr1 город"`
  - All five confirmed idempotent under a second `normalize()` pass.
- `configs/ru/eval.yaml` currently has `train_paths: []` (not `[data/corpus_ru.txt]`) with a comment explaining the STS pairs overlap the corpus by construction — Task 5 only adds `language`/`spacy_model` fields, `train_paths` is untouched.
- `tests/test_evaluate.py` already exists (tests `_retrieval_at_k` only) — Task 3 extends it, doesn't create it.
- `tests/test_serve.py` already exists with a `_make_client`/monkeypatch pattern — Task 4 follows the same pattern.
- Retraining `artifacts/{tokenizer,encoder,simcse}_ru` and regenerating `metrics/eval_ru.json` is explicitly **out of scope** for this plan (per the spec) — a separate follow-up run of `run_pipeline.py --lang ru --spacy-model ru_core_news_sm` after these tasks land.

---

### Task 1: `preprocess.normalize()` — spaCy-based text preparation

**Files:**
- Modify: `src/langembed/preprocess.py` (full rewrite, file is only 34 lines)
- Modify: `pyproject.toml` (add `spacy>=3.7` to the `ml` extra)
- Test: `tests/test_preprocess.py`

**Interfaces:**
- Produces: `normalize(text: str, lang: str = "gu", spacy_model: str | None = None) -> str` — the new third parameter, consumed by Tasks 2, 3, 4.
- Consumes: nothing new from other tasks.

- [ ] **Step 1: Confirm the dev environment has spaCy + the Russian model**

Run:
```bash
python -c "import spacy; print(spacy.__version__)"
python -c "import spacy; spacy.load('ru_core_news_sm', exclude=['ner','parser']); print('ok')"
```
Expected: a version string, then `ok`. If either fails:
```bash
pip install "spacy>=3.7"
python -m spacy download ru_core_news_sm
```

- [ ] **Step 2: Write the failing tests**

Replace `tests/test_preprocess.py` with:

```python
"""Unit tests for preprocess.normalize()."""

from __future__ import annotations

import pytest

from langembed.preprocess import normalize


def test_idempotent():
    s = "  hello    world   "
    once = normalize(s)
    assert normalize(once) == once


def test_collapses_and_strips():
    assert normalize("a   b\tc\n") == "a b c"


def test_gu_unaffected_without_spacy_model():
    """No spacy_model given: identical to the pre-spaCy-feature behavior."""
    assert normalize("  hello    world   ", "gu") == "hello world"


def test_spacy_model_missing_falls_back_gracefully():
    """An undownloaded/nonexistent spaCy model must not raise."""
    result = normalize("Мама мыла раму.", "ru", "nonexistent-model-xyz")
    assert result == "Мама мыла раму."


def _ru_model_available() -> bool:
    try:
        import spacy

        spacy.load("ru_core_news_sm", exclude=["ner", "parser"])
        return True
    except Exception:
        return False


requires_ru_model = pytest.mark.skipif(
    not _ru_model_available(), reason="ru_core_news_sm spaCy model not installed"
)


@requires_ru_model
def test_spacy_lemmatizes():
    assert normalize("кошки бежали быстро", "ru", "ru_core_news_sm") == "кошка бежать быстро"


@requires_ru_model
def test_spacy_substitutes_pronoun_and_numeral():
    result = normalize("она купила пять яблок", "ru", "ru_core_news_sm")
    assert result == "pron1 купить ordinal1 яблоко"


@requires_ru_model
def test_spacy_substitutes_proper_noun():
    result = normalize("Пушкин написал роман", "ru", "ru_core_news_sm")
    assert result == "person1 написать роман"


@requires_ru_model
def test_spacy_substitutes_abbreviations():
    result = normalize("г. Москва и др. города", "ru", "ru_core_news_sm")
    assert result == "abbr1 person1 abbr1 город"


@requires_ru_model
def test_spacy_idempotent():
    once = normalize("она купила пять яблок у Пушкина", "ru", "ru_core_news_sm")
    twice = normalize(once, "ru", "ru_core_news_sm")
    assert twice == once
```

- [ ] **Step 3: Run tests to verify the new ones fail**

Run: `pytest tests/test_preprocess.py -v`
Expected: `test_idempotent`, `test_collapses_and_strips`, `test_gu_unaffected_without_spacy_model` PASS (unchanged behavior); every other test FAILs with `TypeError: normalize() takes from 1 to 2 positional arguments but 3 were given`.

- [ ] **Step 4: Write the implementation**

Replace `src/langembed/preprocess.py` entirely with:

```python
"""Single source of truth for text normalization (used by both train and serve).

Indic-specific normalization is applied when indic-nlp-library is installed; if
it is missing the function falls back to NFC + whitespace collapse so the module
stays importable and testable. Linguistic text preparation (lemmatization +
POS-token substitution) is applied when a spaCy model name is given and spaCy /
that model are installed; otherwise it is skipped the same way.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from typing import Any

_WS = re.compile(r"\s+")

_POS_TOKENS = {"PROPN": "person1", "PRON": "pron1", "NUM": "ordinal1"}
_ABBR_TOKEN = "abbr1"
_RESERVED_TOKENS = frozenset({*_POS_TOKENS.values(), _ABBR_TOKEN})


@functools.lru_cache(maxsize=4)
def _indic_normalizer(lang: str) -> object | None:
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

        return IndicNormalizerFactory().get_normalizer(lang)
    except Exception:
        return None


@functools.lru_cache(maxsize=4)
def _spacy_pipeline(model_name: str) -> object | None:
    try:
        import spacy

        return spacy.load(model_name, exclude=["ner", "parser"])
    except Exception:
        return None


def _looks_like_abbreviation(text: str) -> bool:
    return "." in text and any(ch.isalpha() for ch in text)


def _prepare_tokens(text: str, model_name: str) -> str | None:
    """Lemmatize + substitute PROPN/PRON/NUM/abbreviation tokens via spaCy.

    Returns None (leaving `text` untouched) if spaCy or `model_name` isn't available.
    """
    nlp: Any = _spacy_pipeline(model_name)
    if nlp is None:
        return None
    out: list[str] = []
    for tok in nlp(text):
        if tok.is_space:
            continue
        if tok.text in _RESERVED_TOKENS:
            out.append(tok.text)  # already-prepared text is a fixed point (idempotency)
        elif _looks_like_abbreviation(tok.text):
            out.append(_ABBR_TOKEN)
        elif tok.pos_ in _POS_TOKENS:
            out.append(_POS_TOKENS[tok.pos_])
        else:
            out.append(tok.lemma_.lower())
    return " ".join(out)


def normalize(text: str, lang: str = "gu", spacy_model: str | None = None) -> str:
    """Normalize text deterministically. Idempotent: normalize(normalize(x)) == normalize(x)."""
    text = unicodedata.normalize("NFC", text)
    norm = _indic_normalizer(lang)
    if norm is not None:
        text = norm.normalize(text)  # type: ignore[attr-defined]
    if spacy_model:
        prepared = _prepare_tokens(text, spacy_model)
        if prepared is not None:
            text = prepared
    return _WS.sub(" ", text).strip()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_preprocess.py -v`
Expected: 10 passed (or 6 passed + 4 skipped if `ru_core_news_sm` isn't installed in the environment running the tests — both are acceptable per the graceful-degradation constraint, but Step 1 already confirmed it's installed here, so expect all 10 to pass).

- [ ] **Step 6: Add the dependency declaration**

In `pyproject.toml`, change the `ml` extra (currently `"indic-nlp-library>=0.92",` followed by `"mlflow>=2.12",`) to insert spaCy right after indic-nlp-library:

```toml
ml = [
  "torch>=2.2",
  "transformers>=4.40",
  "sentence-transformers>=3.0",
  "tokenizers>=0.19",
  "datasets>=2.19",
  "indic-nlp-library>=0.92",
  "spacy>=3.7",
  "mlflow>=2.12",
  "peft>=0.11",
  "accelerate>=0.30",
  "bitsandbytes>=0.43",
  "sentencepiece>=0.2",
  "scipy>=1.11",
]
```

- [ ] **Step 7: Lint and commit**

```bash
ruff format src tests
ruff check src tests
mypy src
git add src/langembed/preprocess.py tests/test_preprocess.py pyproject.toml
git commit -m "feat(preprocess): generic spaCy-based text preparation (lemmatize + POS token substitution)"
```

---

### Task 2: `build_corpus.py` — thread `lang`/`spacy_model` through leakage hashing

**Files:**
- Modify: `src/langembed/data/build_corpus.py`
- Test: `tests/test_build_corpus.py`

**Interfaces:**
- Consumes: `normalize(text, lang, spacy_model)` from Task 1.
- Produces: `_h(s, lang="gu", spacy_model=None)`, `load_test_hashes(test_path, lang="gu", spacy_model=None)`, `build_corpus(raw_paths, out_path, test_hashes, lang="gu", spacy_model=None)` — the new trailing parameters, consumed by no other task (this file's `main()` is the only caller of the config-driven path).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_corpus.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify the new one fails**

Run: `pytest tests/test_build_corpus.py -v`
Expected: `test_leakage_guard_lang_default_backward_compatible` PASSes (existing behavior, matches the pre-existing `test_guard_raises_on_leakage` test it's modeled on). `test_leakage_guard_uses_spacy_model_consistently` FAILs with `TypeError: load_test_hashes() takes from 1 to 2 positional arguments but 3 were given` (proving the guard doesn't yet detect this leakage case because it isn't passing `spacy_model` through).

- [ ] **Step 3: Write the implementation**

Replace `src/langembed/data/build_corpus.py` entirely with:

```python
"""Build a clean monolingual corpus with a hard guard against test leakage (Phase 1)."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from langembed.config import load_config
from langembed.data.dedup import dedup
from langembed.preprocess import normalize


def _h(s: str, lang: str = "gu", spacy_model: str | None = None) -> str:
    return hashlib.sha1(normalize(s, lang, spacy_model).encode("utf-8")).hexdigest()


def load_test_hashes(
    test_path: str | Path, lang: str = "gu", spacy_model: str | None = None
) -> set[str]:
    """Hash every sentence of the STS test set so we can detect leakage."""
    hashes: set[str] = set()
    p = Path(test_path)
    if not p.exists():
        return hashes
    for line in p.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        for key in ("sentence_a", "sentence_b"):
            if key in r:
                hashes.add(_h(r[key], lang, spacy_model))
    return hashes


def build_corpus(
    raw_paths: Sequence[str],
    out_path: str | Path,
    test_hashes: set[str],
    lang: str = "gu",
    spacy_model: str | None = None,
) -> int:
    """Normalize -> dedup -> guard -> write JSONL-free one-sentence-per-line corpus."""
    docs: list[str] = []
    for rp in raw_paths:
        for line in Path(rp).open(encoding="utf-8"):
            t = normalize(line, lang, spacy_model)
            if t:
                docs.append(t)
    docs = dedup(docs)
    leaked: Iterable[str] = (d for d in docs if _h(d, lang, spacy_model) in test_hashes)
    n_leaked = sum(1 for _ in leaked)
    if n_leaked:
        raise RuntimeError(f"Test leakage: {n_leaked} corpus lines overlap the STS test set")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(d + "\n")
    return len(docs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    spacy_model = cfg.get("spacy_model")
    th = load_test_hashes(cfg["data"]["test_path"], cfg["language"], spacy_model)
    n = build_corpus(
        cfg["data"]["raw_paths"], cfg["data"]["out_path"], th, cfg["language"], spacy_model
    )
    print(f"corpus lines: {n}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_build_corpus.py -v`
Expected: 4 passed (`test_guard_raises_on_leakage`, `test_writes_corpus`, both new tests).

- [ ] **Step 5: Lint and commit**

```bash
ruff format src tests
ruff check src tests
mypy src
git add src/langembed/data/build_corpus.py tests/test_build_corpus.py
git commit -m "fix(build_corpus): thread lang/spacy_model through leakage-hash consistency"
```

---

### Task 3: `evaluate.py` — thread `lang`/`spacy_model` + normalize test sentences before scoring

**Files:**
- Modify: `src/langembed/eval/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `normalize(text, lang, spacy_model)` from Task 1.
- Produces: `_h(s, lang="gu", spacy_model=None)`, `assert_no_leakage(test_path, train_paths, lang="gu", spacy_model=None)`, and a new pure helper `_load_test_pairs(test_path, score_scale, lang="gu", spacy_model=None) -> tuple[list[str], list[str], list[float]]` — consumed by no other task.

- [ ] **Step 1: Write the failing tests**

First, change the top of `tests/test_evaluate.py` from:

```python
"""Unit tests for evaluate helpers."""

from __future__ import annotations

import numpy as np
```

to:

```python
"""Unit tests for evaluate helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest
```

Then append to `tests/test_evaluate.py`:

```python
def test_load_test_pairs_normalizes_sentences(tmp_path) -> None:
    from langembed.eval.evaluate import _load_test_pairs

    test_path = tmp_path / "test.jsonl"
    test_path.write_text(
        '{"sentence_a": "  hello   world  ", "sentence_b": "foo\\tbar", "score": 4}\n',
        encoding="utf-8",
    )
    sa, sb, scores = _load_test_pairs(str(test_path), score_scale=5.0)
    assert sa == ["hello world"]
    assert sb == ["foo bar"]
    assert scores == [0.8]


def _ru_model_available() -> bool:
    try:
        import spacy

        spacy.load("ru_core_news_sm", exclude=["ner", "parser"])
        return True
    except Exception:
        return False


requires_ru_model = pytest.mark.skipif(
    not _ru_model_available(), reason="ru_core_news_sm spaCy model not installed"
)


@requires_ru_model
def test_load_test_pairs_applies_spacy_model(tmp_path) -> None:
    from langembed.eval.evaluate import _load_test_pairs

    test_path = tmp_path / "test.jsonl"
    test_path.write_text(
        json.dumps({"sentence_a": "она купила пять яблок", "sentence_b": "x", "score": 5})
        + "\n",
        encoding="utf-8",
    )
    sa, sb, _ = _load_test_pairs(
        str(test_path), score_scale=5.0, lang="ru", spacy_model="ru_core_news_sm"
    )
    assert sa == ["pron1 купить ordinal1 яблоко"]


@requires_ru_model
def test_assert_no_leakage_uses_spacy_model_consistently(tmp_path) -> None:
    """Same leakage-consistency proof as build_corpus.py's guard, for evaluate.py's copy."""
    from langembed.eval.evaluate import assert_no_leakage

    test_path = tmp_path / "test.jsonl"
    test_path.write_text(
        json.dumps({"sentence_a": "она купила пять яблок", "sentence_b": "x", "score": 5}) + "\n",
        encoding="utf-8",
    )
    train_path = tmp_path / "train.txt"
    train_path.write_text("она купила пять яблок\n", encoding="utf-8")

    with pytest.raises(RuntimeError):
        assert_no_leakage(str(test_path), [str(train_path)], "ru", "ru_core_news_sm")
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `pytest tests/test_evaluate.py -v`
Expected: all three new tests FAIL — `test_load_test_pairs_normalizes_sentences` and `test_load_test_pairs_applies_spacy_model` with `ImportError: cannot import name '_load_test_pairs'`; `test_assert_no_leakage_uses_spacy_model_consistently` with `TypeError: assert_no_leakage() takes from 2 to 2 positional arguments but 4 were given`.

- [ ] **Step 3: Write the implementation**

Replace `src/langembed/eval/evaluate.py` entirely with:

```python
"""Phase 6: evaluate branches A/B/C on the isolated STS test, with leakage guard."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from langembed.config import load_config
from langembed.preprocess import normalize


def _h(s: str, lang: str = "gu", spacy_model: str | None = None) -> str:
    return hashlib.sha1(normalize(s, lang, spacy_model).encode("utf-8")).hexdigest()


def assert_no_leakage(
    test_path: str,
    train_paths: Sequence[str],
    lang: str = "gu",
    spacy_model: str | None = None,
) -> None:
    test_hashes: set[str] = set()
    for line in Path(test_path).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        test_hashes |= {
            _h(r["sentence_a"], lang, spacy_model),
            _h(r["sentence_b"], lang, spacy_model),
        }
    for tp in train_paths:
        p = Path(tp)
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            if line.strip() and _h(line, lang, spacy_model) in test_hashes:
                raise RuntimeError(f"Test leakage detected via {tp}")


def _load_test_pairs(
    test_path: str, score_scale: float, lang: str = "gu", spacy_model: str | None = None
) -> tuple[list[str], list[str], list[float]]:
    sa: list[str] = []
    sb: list[str] = []
    scores: list[float] = []
    for line in Path(test_path).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        sa.append(normalize(r["sentence_a"], lang, spacy_model))
        sb.append(normalize(r["sentence_b"], lang, spacy_model))
        scores.append(r["score"] / score_scale)
    return sa, sb, scores


def _retrieval_at_k(model: Any, sa: list[str], sb: list[str], k: int) -> dict[str, float]:
    """Recall@k and MRR@k: each sa[i] is a query, sb[i] is its single positive."""
    q_embs = model.encode(sa, normalize_embeddings=True, show_progress_bar=False)
    c_embs = model.encode(sb, normalize_embeddings=True, show_progress_bar=False)
    sims = q_embs @ c_embs.T  # [N, N]
    n = len(sa)
    recall = 0.0
    mrr = 0.0
    for i in range(n):
        ranked = list(np.argsort(-sims[i]))[:k]
        if i in ranked:
            recall += 1.0
        for rank, j in enumerate(ranked):
            if j == i:
                mrr += 1.0 / (rank + 1)
                break
    return {f"recall@{k}": recall / n, f"mrr@{k}": mrr / n}


def evaluate(cfg: dict[str, Any]) -> dict[str, float]:
    # Pre-load transitive deps in correct order to avoid DLL-init crash
    # (sentence_transformers → pandas → pyarrow segfaults on Windows + Python 3.14
    # unless torch/datasets/pyarrow are imported first in the same process).
    import datasets  # noqa: F401
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    import torch  # noqa: F401
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator

    lang = cfg.get("language", "gu")
    spacy_model = cfg.get("spacy_model")
    assert_no_leakage(cfg["test_path"], cfg.get("train_paths", []), lang, spacy_model)
    sa, sb, scores = _load_test_pairs(cfg["test_path"], cfg["score_scale"], lang, spacy_model)

    k = cfg.get("retrieval_k", 10)
    spearman_evaluator = EmbeddingSimilarityEvaluator(sa, sb, scores, name="gu-sts")
    results: dict[str, float] = {}
    for branch, path in cfg["branches"].items():
        if not Path(path).exists():
            print(f"skip branch {branch}: {path} missing")
            continue
        model = SentenceTransformer(path)
        raw = spearman_evaluator(model)
        spearman = raw[spearman_evaluator.primary_metric] if isinstance(raw, dict) else float(raw)
        results[f"spearman_{branch}"] = spearman
        print(f"branch {branch}: Spearman = {spearman:.4f}")
        ret = _retrieval_at_k(model, sa, sb, k)
        results[f"retrieval_recall@{k}_{branch}"] = ret[f"recall@{k}"]
        results[f"retrieval_mrr@{k}_{branch}"] = ret[f"mrr@{k}"]
        print(
            f"branch {branch}: Recall@{k}={ret[f'recall@{k}']:.4f}, MRR@{k}={ret[f'mrr@{k}']:.4f}"
        )
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    results = evaluate(cfg)
    metrics_path = Path(cfg.get("metrics_path", "metrics/eval.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluate.py -v`
Expected: 5 passed (`test_retrieval_at_k_perfect`, `test_retrieval_at_k_worst`, plus the 3 new tests).

- [ ] **Step 5: Lint and commit**

```bash
ruff format src tests
ruff check src tests
mypy src
git add src/langembed/eval/evaluate.py tests/test_evaluate.py
git commit -m "fix(evaluate): normalize test sentences before scoring + thread lang/spacy_model through leakage guard"
```

---

### Task 4: `serve.py` — `LANGEMBED_SPACY_MODEL` env var

**Files:**
- Modify: `src/langembed/serving/serve.py`
- Test: `tests/test_serve.py`

**Interfaces:**
- Consumes: `normalize(text, lang, spacy_model)` from Task 1.
- Produces: nothing new consumed elsewhere (serve.py is the terminal call site).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_serve.py`:

```python
def test_embed_passes_lang_and_spacy_model_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """LANGEMBED_LANG and LANGEMBED_SPACY_MODEL must both be forwarded to normalize()."""
    import langembed.serving.serve as srv

    captured: list[tuple[str, str, Any]] = []

    def _fake_normalize(text: str, lang: str = "gu", spacy_model: str | None = None) -> str:
        captured.append((text, lang, spacy_model))
        return text

    monkeypatch.setattr(srv, "normalize", _fake_normalize)
    monkeypatch.setenv("LANGEMBED_LANG", "ru")
    monkeypatch.setenv("LANGEMBED_SPACY_MODEL", "ru_core_news_sm")
    monkeypatch.setattr(srv, "_model", None)
    monkeypatch.setattr(srv, "_get_model", lambda: _FakeModel())

    client = TestClient(srv.app)
    client.post("/embed", json={"texts": ["hello"]})
    assert captured == [("hello", "ru", "ru_core_news_sm")]


def test_embed_spacy_model_env_unset_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without LANGEMBED_SPACY_MODEL set, normalize() must receive spacy_model=None."""
    import langembed.serving.serve as srv

    captured: list[tuple[str, str, Any]] = []

    def _fake_normalize(text: str, lang: str = "gu", spacy_model: str | None = None) -> str:
        captured.append((text, lang, spacy_model))
        return text

    monkeypatch.setattr(srv, "normalize", _fake_normalize)
    monkeypatch.delenv("LANGEMBED_SPACY_MODEL", raising=False)
    monkeypatch.setattr(srv, "_model", None)
    monkeypatch.setattr(srv, "_get_model", lambda: _FakeModel())

    client = TestClient(srv.app)
    client.post("/embed", json={"texts": ["hello"]})
    assert captured == [("hello", "gu", None)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_serve.py -v`
Expected: the two new tests FAIL with `AssertionError` (currently `captured` would be `[("hello", "ru", None)]`-shaped calls missing the third arg, since `serve.py` doesn't read `LANGEMBED_SPACY_MODEL` yet — actual failure is an `AssertionError` comparing tuples of different structure, not a crash, since `_fake_normalize`'s `spacy_model` param defaults to `None` when serve.py calls it with only 2 positional args).

- [ ] **Step 3: Write the implementation**

In `src/langembed/serving/serve.py`, replace the `embed` function:

```python
@app.post("/embed")
def embed(payload: EmbedIn) -> dict[str, Any]:
    model = _get_model()
    lang = os.environ.get("LANGEMBED_LANG", "gu")
    spacy_model = os.environ.get("LANGEMBED_SPACY_MODEL")
    vecs = model.encode(
        [normalize(t, lang, spacy_model) for t in payload.texts], normalize_embeddings=True
    )
    return {"embeddings": vecs.tolist(), "dim": int(vecs.shape[1])}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_serve.py -v`
Expected: 7 passed (5 existing + 2 new).

- [ ] **Step 5: Lint and commit**

```bash
ruff format src tests
ruff check src tests
mypy src
git add src/langembed/serving/serve.py tests/test_serve.py
git commit -m "feat(serve): read LANGEMBED_SPACY_MODEL env var"
```

---

### Task 5: Wire `spacy_model` into `ru` configs + `run_pipeline.py`

**Files:**
- Modify: `configs/ru/tokenizer.yaml`
- Modify: `configs/ru/eval.yaml`
- Modify: `scripts/run_pipeline.py`

**Interfaces:**
- Consumes: `cfg.get("spacy_model")` / `cfg.get("language")` reads added in Tasks 2 and 3.
- Produces: nothing consumed by later tasks (this is the last code task).

- [ ] **Step 1: Add `spacy_model` to the checked-in ru configs**

Edit `configs/ru/tokenizer.yaml` — add a `spacy_model` line after `language: ru`:

```yaml
language: ru
spacy_model: ru_core_news_sm
data:
  raw_paths:
    - data/raw/voina_i_mir_ru.txt
  out_path: data/corpus_ru.txt
  test_path: data/sts_test_ru.jsonl
tokenizer:
  vocab_size: 16000
  min_frequency: 2
  unk_rate_max: 0.01
  out_dir: artifacts/tokenizer_ru
```

Edit `configs/ru/eval.yaml` — add `language` and `spacy_model` lines at the top:

```yaml
language: ru
spacy_model: ru_core_news_sm
test_path: data/sts_test_ru.jsonl
score_scale: 5.0
retrieval_k: 5
branches:
  A: artifacts/simcse_ru
# Leakage guard is a documented no-op here: the ru STS pairs are corpus sentences
# selected via active learning (Task 4), so they overlap data/corpus_ru.txt by design
# — same precedent as configs/smoke/eval.yaml.
train_paths: []
metrics_path: metrics/eval_ru.json
```

(These configs currently describe the `artifacts/tokenizer_ru`/`simcse_ru` models trained *before* this feature — adding these fields here documents intent for the follow-up retrain in Task 5's acceptance note below; it does not retroactively change the already-trained artifacts.)

- [ ] **Step 2: Add the `run_pipeline.py` CLI flag**

In `scripts/run_pipeline.py`, add the new argument after the existing `--vocab-size` argument (inside `main()`'s `ap.add_argument(...)` block):

```python
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument(
        "--spacy-model",
        default=None,
        help="spaCy model for text preparation (lemmatize + POS-token substitution), e.g. ru_core_news_sm; omit to skip",
    )
```

Then thread `args.spacy_model` into the three generated config dicts and the serve env. Change the `tokenizer_cfg` dict definition:

```python
    tokenizer_cfg = {
        "language": lang,
        "spacy_model": args.spacy_model,
        "data": {"raw_paths": raw_paths, "out_path": corpus_path, "test_path": sts_test_path},
        "tokenizer": {
            "vocab_size": args.vocab_size,
            "min_frequency": 2,
            "unk_rate_max": 0.01,
            "out_dir": f"artifacts/tokenizer_{lang}",
        },
    }
```

Change the `eval_cfg` dict definition (inside the `if not args.skip_eval:` block):

```python
        eval_cfg = {
            "language": lang,
            "spacy_model": args.spacy_model,
            "test_path": sts_test_path,
            "score_scale": 5.0,
            "retrieval_k": 5,
            "branches": {"A": f"artifacts/simcse_{lang}"},
            # STS candidates are corpus sentences by construction (active-learning sampled in
            # seed_sts_pairs.py), so they overlap the training corpus by design -- see
            # docs/ru-embeddings-report.pdf, section 3, for why train_paths must stay empty.
            "train_paths": [],
            "metrics_path": f"metrics/eval_{lang}.json",
        }
```

Change the `serve_env` dict definition (near the end of `main()`):

```python
    serve_port = free_port(8000)
    serve_env = {
        **os.environ,
        "LANGEMBED_MODEL_DIR": f"artifacts/simcse_{lang}",
        "LANGEMBED_LANG": lang,
    }
    if args.spacy_model:
        serve_env["LANGEMBED_SPACY_MODEL"] = args.spacy_model
```

- [ ] **Step 3: Verify the config changes parse correctly**

Run:
```bash
python -c "from langembed.config import load_config; print(load_config('configs/ru/tokenizer.yaml'))"
python -c "from langembed.config import load_config; print(load_config('configs/ru/eval.yaml'))"
```
Expected: both print a dict containing `'language': 'ru', 'spacy_model': 'ru_core_news_sm', ...` with no errors.

- [ ] **Step 4: Verify `run_pipeline.py` still parses its arguments correctly**

Run: `python scripts/run_pipeline.py --help`
Expected: help text lists `--spacy-model` with the description from Step 2, exits 0.

- [ ] **Step 5: Lint and commit**

```bash
ruff format src tests scripts
ruff check src tests scripts
mypy src
git add configs/ru/tokenizer.yaml configs/ru/eval.yaml scripts/run_pipeline.py
git commit -m "feat(ru): wire spacy_model into ru configs + run_pipeline.py --spacy-model"
```

---

### Task 6: Documentation — replace the "why no lemmatization" note

**Files:**
- Modify: `README.md`
- Modify: `docs/ru/README_RU.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Replace the English note**

In `README.md`, find this paragraph (added in an earlier conversation, now stale) directly after the "No code change is needed for non-Indic languages..." sentence in the `### Phase 0 — Normalisation` section:

```
**Why there's no lemmatization or POS-token substitution here.** Classical from-scratch embedding recipes (e.g. building static per-word vectors via SVD or word2vec) typically add a much heavier preparation step: lemmatize every word, replace pronouns/numerals with placeholder tokens, optionally strip proper nouns, and filter "garbage" tokens with tf-idf statistics. This project deliberately skips all of that. Every language track here trains a subword BPE tokenizer feeding a transformer (MLM pre-train → SimCSE contrastive), not per-word static vectors — subword tokenization already absorbs case/declension endings statistically, and MLM pre-training needs the full lexical signal (replacing pronouns/numerals with placeholders would remove information the masked-token objective relies on). Adding classical lemmatization would fight this architecture rather than help it, so `normalize()` stays limited to script/Unicode normalisation and whitespace collapse — the same function for every track, per the single-normalisation invariant.
```

Replace it with:

```
**Optional linguistic text preparation.** `normalize()` also accepts a third
parameter, `spacy_model: str | None`. When set to a spaCy model name (e.g.
`ru_core_news_sm`), it additionally lemmatizes every token and substitutes
`PROPN`/`PRON`/`NUM` tokens and dotted abbreviations (e.g. `г.`, `и др.`) with
fixed placeholder tokens (`person1`, `pron1`, `ordinal1`, `abbr1`), matching
the manual reference pipeline's "Предобработка" step. This logic uses only
spaCy's Universal POS tagset and a plain text-shape check for abbreviations —
no per-language code branches — so any spaCy-supported language can opt in
by passing its own model name via `spacy_model` in `configs/<lang>/*.yaml` or
`run_pipeline.py --spacy-model`. Omitting `spacy_model` (the default) leaves
`normalize()`'s output exactly as before — no track is affected unless it
explicitly opts in.
```

- [ ] **Step 2: Replace the Russian note**

In `docs/ru/README_RU.md`, find the mirrored paragraph in `### Фаза 0 — Нормализация`:

```
**Почему здесь нет лемматизации и замены частей речи на токены.** Классические рецепты построения эмбеддингов «с нуля» (например, статические пословные векторы через SVD или word2vec) обычно включают гораздо более тяжёлый шаг предобработки: лемматизация каждого слова, замена местоимений/числительных на токены-заглушки, опционально — удаление имён собственных и фильтрация «мусорных» токенов по tf-idf-статистикам. Этот проект сознательно этого не делает. Каждая языковая ветка здесь обучает субтокенный BPE-токенизатор для трансформера (MLM-предобучение → контрастивное дообучение SimCSE), а не статические пословные векторы — субтокенная токенизация уже статистически поглощает падежные/словоизменительные окончания, а MLM-предобучению нужен полный лексический сигнал (замена местоимений/числительных на заглушки убрала бы информацию, на которую опирается задача предсказания замаскированного токена). Классическая лемматизация здесь скорее мешала бы этой архитектуре, чем помогала, поэтому `normalize()` ограничивается нормализацией скрипта/Unicode и схлопыванием пробелов — одна и та же функция для всех веток, согласно инварианту единой нормализации.
```

Replace it with:

```
**Опциональная лингвистическая предобработка.** `normalize()` также
принимает третий параметр, `spacy_model: str | None`. Если задать имя
модели spaCy (например, `ru_core_news_sm`), функция дополнительно
лемматизирует каждый токен и заменяет токены `PROPN`/`PRON`/`NUM` и
сокращения с точками (например, `г.`, `и др.`) на фиксированные
токены-заглушки (`person1`, `pron1`, `ordinal1`, `abbr1`) — как описано в
шаге «Предобработка» эталонного ручного пайплайна. Эта логика использует
только универсальный набор частей речи spaCy и простую текстовую проверку
для сокращений — никаких языко-специфичных ветвлений кода — поэтому любой
поддерживаемый spaCy язык может подключить эту функцию, указав свою модель
через `spacy_model` в `configs/<lang>/*.yaml` или `run_pipeline.py
--spacy-model`. Если `spacy_model` не задан (по умолчанию), вывод
`normalize()` остаётся точно таким же, как раньше — ни одна ветка не
затрагивается, пока не подключит эту опцию явно.
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/ru/README_RU.md
git commit -m "docs: describe the new opt-in spaCy text-preparation stage"
```

---

## Definition of Done

- [ ] `ruff check`, `ruff format --check`, `mypy src` all clean.
- [ ] `pytest` green, including all new/extended test files (`test_preprocess.py`, `test_build_corpus.py`, `test_evaluate.py`, `test_serve.py`).
- [ ] `normalize(x, "gu")` and `normalize(x)` produce byte-identical output to before this plan (verified by `test_gu_unaffected_without_spacy_model`).
- [ ] `normalize(x, "ru", "ru_core_news_sm")` lemmatizes and substitutes `PROPN`/`PRON`/`NUM`/abbreviation tokens, and is idempotent.
- [ ] The leakage guard in both `build_corpus.py` and `evaluate.py` correctly detects leakage even when `spacy_model` is set (previously would have silently missed it).
- [ ] `evaluate()` normalizes STS test sentences before scoring.
- [ ] `configs/ru/tokenizer.yaml` and `configs/ru/eval.yaml` declare `spacy_model: ru_core_news_sm`; `run_pipeline.py --spacy-model` threads it through for any language.
- [ ] README (English + Russian) describes the new opt-in behavior.
- [ ] **Not done by this plan, explicit follow-up:** retraining `artifacts/{tokenizer,encoder,simcse}_ru` and regenerating `metrics/eval_ru.json` with the new `spacy_model`-aware corpus — run `python scripts/run_pipeline.py --lang ru --input data/raw/voina-i-mir.pdf --spacy-model ru_core_news_sm` (or the `-ru` Makefile-equivalent stages) after this plan's commits land.
