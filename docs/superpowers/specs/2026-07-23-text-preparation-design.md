# Generic Linguistic Text Preparation — Design

**Date:** 2026-07-23
**Status:** Draft (pending user review)

## Problem

`Пайплайн (1).docx` (the manual reference pipeline) describes a "Предобработка"
(text preparation) step that this project's `preprocess.normalize()` currently
skips entirely: lemmatize every word (strip case/declension/conjugation),
replace pronouns and numerals with placeholder tokens, and — marked optional
in the doc, and in scope per the user's decision — also replace proper nouns
and abbreviations. `normalize()` today only does NFC + IndicNLP script
normalization + whitespace collapse (`src/langembed/preprocess.py:27-33`).

The project's stated goal (per the most recent commit, "generic multi-language
pipeline runner") is that `scripts/run_pipeline.py --lang <code>` works for
any language without code changes. A naive implementation of the doc's
Russian-flavored preparation step (e.g. hardcoding `natasha`, a Russian-only
library) would regress that. This design adds the preparation step while
keeping the pipeline language-agnostic, and fixes a related pre-existing gap
in the test-leakage guard and eval scoring that would otherwise silently break
once `normalize()` starts producing meaningfully different output.

## Solution

Extend `preprocess.normalize()` with an optional `spacy_model: str | None`
parameter. When set, it runs the named spaCy pipeline and does:

1. lemmatize every token (`token.lemma_.lower()`)
2. replace `PROPN`/`PRON`/`NUM` tokens with fixed placeholder strings
   (`person1` / `pron1` / `ordinal1`), mirroring the doc's `pos_dict` example
3. replace tokens spaCy's Universal Dependencies morphology tags
   `Abbr=Yes` with `abbr1`

All three rules key off spaCy's **Universal POS/morphology tagset**, which is
identical across every spaCy language model — so the substitution logic
itself has no Russian-specific code. Which model to use (`ru_core_news_sm`,
`fr_core_news_sm`, ...) is never chosen by the library; it comes from
`configs/<lang>/*.yaml` (new `spacy_model` field) or `run_pipeline.py`'s new
`--spacy-model` flag, exactly like `--lang` already works today. A language
with no configured model (e.g. `gu`) is completely unaffected — same
graceful-skip pattern already used for `_indic_normalizer`.

This is opt-in per language and does not change `gu` (or any track without a
`spacy_model`) at all. For `ru`, it does change `normalize()`'s output, which
means `artifacts/tokenizer_ru`, `encoder_ru`, `simcse_ru`, `embeddings_ru`,
and `metrics/eval_ru.json` all become stale. Retraining them is **out of
scope for this spec's acceptance** — it's a separate, explicitly-confirmed
follow-up run of `run_pipeline.py --lang ru --spacy-model ru_core_news_sm`
after this code lands and passes review.

## Components

### 1. `src/langembed/preprocess.py`

```python
_POS_TOKENS = {"PROPN": "person1", "PRON": "pron1", "NUM": "ordinal1"}
_ABBR_TOKEN = "abbr1"
_RESERVED_TOKENS = frozenset({*_POS_TOKENS.values(), _ABBR_TOKEN})


@functools.lru_cache(maxsize=4)
def _spacy_pipeline(model_name: str) -> object | None:
    try:
        import spacy

        return spacy.load(model_name, exclude=["ner", "parser"])
    except Exception:
        return None


def _prepare_tokens(text: str, model_name: str) -> str | None:
    nlp = _spacy_pipeline(model_name)
    if nlp is None:
        return None
    out: list[str] = []
    for tok in nlp(text):
        if tok.is_space:
            continue
        if tok.text in _RESERVED_TOKENS:
            out.append(tok.text)  # already-prepared text stays a fixed point
        elif tok.morph.get("Abbr") == ["Yes"]:
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
        text = norm.normalize(text)
    if spacy_model:
        prepared = _prepare_tokens(text, spacy_model)
        if prepared is not None:
            text = prepared
    return _WS.sub(" ", text).strip()
```

Notes:
- `exclude=["ner", "parser"]` drops pipeline components this feature doesn't
  need, for speed; exact component names to exclude are confirmed against the
  installed model during implementation (`nlp.pipe_names`).
- **Idempotency safeguard:** the `tok.text in _RESERVED_TOKENS` check makes
  placeholder tokens a fixed point — re-running `normalize()` on already-
  prepared text leaves `person1`/`pron1`/`ordinal1`/`abbr1` untouched instead
  of re-tagging them (which would be undefined behavior on an out-of-
  vocabulary token). This preserves the existing
  `normalize(normalize(x)) == normalize(x)` contract, which
  `tests/test_preprocess.py` already asserts.
- Placeholders are lowercase (`person1`, not `PERSON1` as in the doc's literal
  snippet) so they compose cleanly with the rest of the lowercased,
  lemmatized text and don't get re-cased by a second `normalize()` pass.
- Follows the existing `_indic_normalizer` pattern: heavy import inside the
  function, `lru_cache` so the model loads once per process, broad
  `except Exception: return None` so a missing `spacy` install or
  undownloaded model degrades to a no-op rather than crashing callers that
  don't have the full ML stack (matches CLAUDE.md's "ML imports stay inside
  functions" convention).

### 2. Leakage-hash consistency fix

**Bug found during design:** `build_corpus.py` and `evaluate.py` both hash
sentences for the test-leakage guard via a `_h(s)` helper that calls
`normalize(s)` with **no `lang`/`spacy_model` argument** — always the "gu"
default, even when the actual corpus was built with `lang="ru"`,
`spacy_model="ru_core_news_sm"`. Today this is invisible because `normalize()`
without a spaCy model is nearly a no-op either way. Once `ru` gets real
lemmatization, `_h()` would hash raw, un-lemmatized test sentences while
`build_corpus()` hashes lemmatized corpus lines — the leakage guard
(`Запрет утечки теста`, a hard invariant per `CLAUDE.md`) would silently stop
detecting real overlap.

Fix: thread `lang`/`spacy_model` through the hashing path in both files.

```python
# build_corpus.py
def _h(s: str, lang: str = "gu", spacy_model: str | None = None) -> str:
    return hashlib.sha1(normalize(s, lang, spacy_model).encode("utf-8")).hexdigest()

def load_test_hashes(
    test_path: str | Path, lang: str = "gu", spacy_model: str | None = None
) -> set[str]:
    ...  # pass lang, spacy_model to _h()

def build_corpus(..., lang: str = "gu", spacy_model: str | None = None) -> int:
    ...  # pass spacy_model to normalize(line, lang, spacy_model) and to _h(d, lang, spacy_model)

def main() -> None:
    cfg = load_config(args.config)
    spacy_model = cfg.get("spacy_model")
    th = load_test_hashes(cfg["data"]["test_path"], cfg["language"], spacy_model)
    n = build_corpus(cfg["data"]["raw_paths"], cfg["data"]["out_path"], th, cfg["language"], spacy_model)
```

Same shape in `evaluate.py`'s `_h()` / `assert_no_leakage()`, reading
`cfg.get("language", "gu")` / `cfg.get("spacy_model")`.

### 3. `evaluate.py` — normalize test sentences before scoring

**Second gap found during design:** `evaluate()` currently encodes `sa`/`sb`
(the raw STS test sentences) directly — it never calls `normalize()` before
`model.encode(...)`. Harmless today; would badly understate Spearman for `ru`
once the model is trained on lemmatized text but scored on raw text.

```python
def evaluate(cfg: dict[str, Any]) -> dict[str, float]:
    ...
    lang = cfg.get("language", "gu")
    spacy_model = cfg.get("spacy_model")
    assert_no_leakage(cfg["test_path"], cfg.get("train_paths", []), lang, spacy_model)
    ...
    for line in Path(cfg["test_path"]).open(encoding="utf-8"):
        ...
        sa.append(normalize(r["sentence_a"], lang, spacy_model))
        sb.append(normalize(r["sentence_b"], lang, spacy_model))
        ...
```

### 4. Config wiring

`configs/ru/tokenizer.yaml` and `configs/ru/eval.yaml` gain a top-level
`spacy_model` field (sibling to the existing `language` field the former
already has; `eval.yaml` gains both `language` and `spacy_model`, neither of
which it currently declares):

```yaml
language: ru
spacy_model: ru_core_news_sm
data:
  ...
```

`configs/ru/contrastive.yaml` needs no change — `train_simcse.py` and
`embed_corpus.py` both consume `data/corpus_ru.txt` directly, which is
already-normalized text written once by `build_corpus.py`.

### 5. `serve.py`

New optional env var `LANGEMBED_SPACY_MODEL`, read alongside the existing
`LANGEMBED_LANG`:

```python
lang = os.environ.get("LANGEMBED_LANG", "gu")
spacy_model = os.environ.get("LANGEMBED_SPACY_MODEL")
vecs = model.encode([normalize(t, lang, spacy_model) for t in payload.texts], ...)
```

### 6. `scripts/run_pipeline.py`

New optional `--spacy-model` CLI arg (default `None`), written into the
generated `tokenizer_cfg` and `eval_cfg` dicts as `spacy_model`, and passed as
`LANGEMBED_SPACY_MODEL` in the `serve_env` dict used for the skew check.
Omitting the flag reproduces today's behavior exactly for any language.

### 7. Dependency

Add `spacy>=3.7` to the `ml` extra in `pyproject.toml` (same extra as
`indic-nlp-library`, since it's the same category of optional heavy NLP
dependency used only inside `preprocess.py`). The model itself
(`ru_core_news_sm`) is **not** a pip dependency — it's downloaded separately
via `python -m spacy download ru_core_news_sm`, documented in the README next
to the existing install instructions. `_spacy_pipeline`'s `except Exception`
means an undownloaded model degrades to a no-op with no crash, consistent
with how a missing `indic-nlp-library` install already behaves.

## Data Flow

```
raw text
  --(normalize: NFC)-->
  --(normalize: IndicNLP, only if lang is Indic)-->
  --(normalize: spaCy lemmatize + PROPN/PRON/NUM/Abbr substitution, only if spacy_model set)-->
  --(normalize: whitespace collapse)-->
normalized text
```

Same function, same order of operations, called from the same three places
as today (`build_corpus.py`, `evaluate.py`, `serve.py`) — no new call sites,
no duplicated normalization logic anywhere.

## Error Handling

- `spacy` not installed, or `spacy_model` not downloaded: `_spacy_pipeline`
  catches the exception and returns `None`; `normalize()` silently falls back
  to pre-spaCy output (NFC + whitespace collapse) for that call. This means a
  misconfigured environment degrades quality rather than crashing training or
  serving — acceptable because `make lint`/`mypy`/import-time checks don't
  require the model, matching the project's "importable without the full ML
  stack" convention. It also means a broken config (typo'd model name) fails
  *silently* rather than loudly; flagged as an accepted tradeoff below.
- No change to `train_supervised.py`'s existing missing-triplets guard.

## Testing

- `tests/test_preprocess.py`: extend with spaCy-gated cases behind
  `pytest.importorskip("spacy")`, plus a per-test skip if
  `ru_core_news_sm` isn't downloaded (mirrors the `pytest.importorskip`
  pattern already used in `tests/test_build_corpus.py`,
  `tests/test_dedup.py`, `tests/test_tokenizer.py`):
  - lemmatization: an inflected Russian word normalizes to its lemma
  - `PROPN`/`PRON`/`NUM` substitution: a sentence with a name, a pronoun, and
    a number produces `person1`/`pron1`/`ordinal1` in the right positions
  - abbreviation substitution: a token with `Abbr=Yes` becomes `abbr1`
  - **idempotency**: `normalize(normalize(x, "ru", "ru_core_news_sm"), "ru", "ru_core_news_sm") == normalize(x, "ru", "ru_core_news_sm")`
  - graceful skip: `normalize(x, "ru", "nonexistent-model-xyz")` doesn't raise
    and returns the pre-spaCy-stage output
  - `gu` (no `spacy_model`) output is byte-identical to before this change
- `tests/test_build_corpus.py`: extend `_h`/`load_test_hashes`/`build_corpus`
  tests to cover the `lang`/`spacy_model`-threading fix — construct a case
  where a corpus line and a test sentence are identical pre-lemmatization but
  would hash differently if `_h()` ignored `spacy_model`, and assert the
  leakage guard still fires.
- `tests/test_evaluate.py` (already exists): extend with a case asserting
  `evaluate()` calls `normalize()` on `sa`/`sb` before encoding — e.g. via a
  fake model capturing its `encode()` input and checking it matches
  `normalize(raw, lang, spacy_model)`.
- `ruff check`, `ruff format --check`, `mypy src` clean, per existing
  convention.

## Out of scope

- The docx's separate "Очистка от мусора" step (tf-idf-based garbage-token
  filtering across the whole corpus) — a distinct pipeline stage, not part of
  the per-sentence "Предобработка" step this design covers. Not requested;
  flagging here so it's visible on spec review in case it should be included.
- A rule-based fallback lemmatizer for languages with no spaCy model (the
  doc's documented fallback for low-resource languages) — out of scope; those
  languages simply skip this stage today, same as before this change.
- Actually retraining `artifacts/{tokenizer,encoder,simcse}_ru` and
  regenerating `metrics/eval_ru.json` / `artifacts/embeddings_ru` — a
  follow-up operational step (`run_pipeline.py --lang ru --spacy-model
  ru_core_news_sm`), not part of this spec's acceptance criteria. Requires ~
  20-35 min of CPU pretrain plus a manual STS re-labeling pass (or reuse of
  the existing `data/sts_test_ru.jsonl`, since `evaluate()` will now
  normalize it consistently at scoring time), so it should be kicked off
  explicitly and separately after this code lands.
- Performance tuning of per-sentence spaCy calls (e.g. `nlp.pipe()` batching
  inside `build_corpus.py`) — `normalize()`'s public API stays a single-string
  function (required for `serve.py`'s per-request use). For a large corpus,
  corpus-build time will grow noticeably (minutes, not seconds) once
  `spacy_model` is set; not optimized further in this iteration unless it
  proves to be a real blocker during the follow-up retrain.

## Constraints

- No Russian-specific (or any-language-specific) code in `preprocess.py` —
  only spaCy's Universal POS/morphology tagset, which is shared across all
  spaCy language models.
- `gu` track and any track without a `spacy_model` configured: byte-identical
  behavior to today.
- Single normalization function invariant preserved: `normalize()` remains
  the only place text preparation logic lives; `build_corpus.py`,
  `evaluate.py`, and `serve.py` all call it the same way, now consistently
  passing `lang`/`spacy_model` everywhere they previously passed `lang` alone
  (or nothing).

## Commits

1. `feat(preprocess): generic spaCy-based text preparation (lemmatize + POS
   token substitution)` — `preprocess.py` changes + `pyproject.toml`
   dependency + new tests.
2. `fix(leakage): thread lang/spacy_model through build_corpus and evaluate
   hashing + normalize eval sentences before scoring` — the two correctness
   fixes above + their tests.
3. `feat(ru): wire spacy_model into ru configs + run_pipeline.py` —
   `configs/ru/tokenizer.yaml`, `configs/ru/eval.yaml`, `serve.py`,
   `run_pipeline.py --spacy-model` flag.
4. Docs: update `README.md` / `docs/ru/README_RU.md`'s Phase 0 section,
   replacing the "why we don't lemmatize" note added earlier in this
   conversation with a description of the new opt-in behavior.

Retraining the `ru` artifacts is a separate follow-up after these commits
land, not bundled into them.
