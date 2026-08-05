# Automated STS Labeling via Back-Translation — Design

**Date:** 2026-08-05
**Status:** Approved

## Problem

`scripts/run_pipeline.py` step 5/6 (`=== [{lang}] 5/6 human-in-the-loop STS labeling + eval ===`)
blocks on `input()`, waiting for a human to open `http://localhost:{port}/label` and rate STS
pairs 1–5 before the pipeline can build `sts_test_<lang>.jsonl` and continue to eval and final
embeddings. This is the only step in an otherwise unattended pipeline
(extract → corpus → tokenizer → pretrain → SimCSE → eval → serve-skew check) that requires a
person. There is no way to run `run_pipeline.py` start-to-finish without someone present.

## Solution

Add an opt-in `--auto-label` flag to `run_pipeline.py`. When set, step 5 generates silver-standard
STS pairs automatically instead of waiting on the manual `/label` form, using back-translation
through free translation services. Manual labeling stays the default — this is a parallel path,
selected explicitly at job startup, not a replacement.

### 1. `src/langembed/data/backtranslate.py` (new)

Round-trip translation (`source_lang → pivot_lang → source_lang`) through one or more free,
keyless MT backends via the `deep-translator` library (`GoogleTranslator`, `MyMemoryTranslator`).
Multiple providers exist for two reasons: paraphrase diversity (each engine drifts differently on
the round trip) and resilience (a rate-limited or failing provider is skipped in favor of the
next one).

```python
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
    """Round-trip text through the first provider that succeeds; None if every
    provider fails after retries. Successful results are memoized in `cache`
    and appended to `cache_path` immediately (crash-safe, resumable)."""
```

- `_translate_one(text, provider, src, tgt)`: dispatches to the named backend; imports
  `deep_translator` inside the function (heavy/optional dependency, per project convention).
- Cache key: `sha1(f"{provider}|{source_lang}|{pivot_lang}|{text}")`. `load_cache()` reads the
  JSONL cache file into a dict at startup; `append_cache()` writes one line per new entry —
  re-running a partially-completed job skips already-translated sentences instead of re-spending
  free-tier quota.
- `delay` (derived from a requests-per-minute budget) is slept between calls to stay polite to
  free public services.
- No `enabled` flag inside this module — gating happens once, at the `run_pipeline.py` CLI level.

### 2. `src/langembed/annotation/auto_label.py` (new)

Builds silver STS pairs directly from corpus sentences — no database, no server, no human:

```python
def build_auto_sts_pairs(
    sentences: list[str],
    n: int,
    providers: list[str],
    pivot_lang: str,
    source_lang: str,
    cache_path: str | Path,
    requests_per_minute: float = 20.0,
    seed: int = 42,
) -> list[tuple[str, str, float]]: ...

def write_sts_pairs(pairs: list[tuple[str, str, float]], out_path: str | Path) -> int: ...
```

Three tiers of pairs, evenly split across `n`, giving the eval set a graduated spread rather than
two clusters:

| Tier | Construction | Score |
|---|---|---|
| Paraphrase (high similarity) | `(sentence, back_translate(sentence))` | 4.8 |
| Adjacent (mid similarity) | `(sentences[i], sentences[i+1])` — same heuristic already used in `scripts/seed_sts_pairs.py::build_candidates` | 2.0 |
| Random (low similarity) | two sentences sampled independently | 0.3 |

Pairs where back-translation fails for every provider are dropped (not padded with a placeholder)
so failure never corrupts the eval set with garbage pairs — it only shrinks the paraphrase tier,
which the pipeline log reports so it's visible. `write_sts_pairs` writes
`{"sentence_a", "sentence_b", "score"}` JSONL, matching `annotation/api.py::export_sts`'s schema
exactly, so `evaluate.py` needs zero changes — it already normalizes `sentence_a`/`sentence_b` at
load time, so raw MT output is fine going in.

### 3. `scripts/run_pipeline.py`

New CLI flags:

```
--auto-label              skip the manual /label step; generate silver STS pairs instead
--translate-providers     default ["google", "mymemory"]
--pivot-lang              default "en"
--translate-rpm           default 20.0 (politeness rate limit for free MT APIs)
```

Step 5 branches on `args.auto_label`:

- **`--auto-label` set:** load `corpus_path` sentences, call `build_auto_sts_pairs(...)` /
  `write_sts_pairs(...)` directly, write to `sts_test_path`. No `docker compose up postgres`, no
  `seed_sts_pairs.py`, no annotation server, no `input()` — this branch has zero blocking calls.
- **default (unset):** existing flow, unchanged byte-for-byte.

The cache file path is derived automatically as `data/backtranslation_cache_<lang>.jsonl`,
following the project's existing per-language path convention
(`data/corpus_<lang>.txt`, `data/native_triplets_<lang>.jsonl`) — no new config surface to
remember.

### 4. `pyproject.toml`

New optional-dependency group:

```toml
translate = ["deep-translator>=1.11"]
```

Kept separate from `ml`/`serve` since it's neither — only `run_pipeline.py --auto-label` and the
two new modules need it, and the import stays function-local so the rest of the package remains
importable without it installed.

## Data integrity

Auto-generated pairs are drawn from `corpus_<lang>.txt`, exactly like the existing
`seed_sts_pairs.py` candidates — the same caveat already documented in `run_pipeline.py`'s
templated `eval_cfg` applies unchanged: `train_paths` stays `[]` in eval config because STS
sentences overlap the training corpus by construction. No new leakage surface is introduced; the
existing `assert_no_leakage` / `train_paths: []` handling in `evaluate.py` is untouched.

## Testing

- `tests/test_backtranslate.py`: mock `deep_translator.GoogleTranslator` /
  `MyMemoryTranslator.translate` (no real network calls) — round-trip composition, provider
  fallback when the first raises, cache hit skips the network call entirely, cache file is
  appended on success.
- `tests/test_auto_label.py`: `build_auto_sts_pairs` returns `n` pairs split across the three
  tiers with correct fixed scores; a provider returning `None` for every candidate shrinks the
  paraphrase tier without raising; `write_sts_pairs` produces JSONL matching `export_sts`'s exact
  key set (`sentence_a`, `sentence_b`, `score`); determinism under a fixed `seed`.
- `tests/test_run_pipeline.py`: extend (or create, if this lands before the file exists) with a
  test that `--auto-label` selects the auto branch and never calls `docker`/`start_server`/`input`
  (patch `run`/`start_server`/`input` and assert non-invocation).

## Out of scope

- Wiring `train_supervised.py` (triplet-based contrastive fine-tuning) into `run_pipeline.py` —
  confirmed with the user this is not the target; the pipeline currently only runs unsupervised
  SimCSE, and that's unchanged by this feature.
- A "medium-noise" fourth tier (e.g., double round-trip through two pivot languages for a score
  between paraphrase and adjacent) — three tiers give a usable spread for Spearman correlation;
  can be added later if eval results show it's needed.
- Self-hosting LibreTranslate — most public mirrors now require an API key, which conflicts with
  "free, keyless"; `deep-translator`'s Google/MyMemory backends need no key today.
