# SVD-Based Auto-Labeling — Design

**Date:** 2026-08-09
**Status:** Approved

## Problem

`--auto-label` (see `docs/superpowers/specs/2026-08-05-automated-sts-labeling-design.md`) currently
has exactly one implementation: back-translation through external, free MT services
(`deep-translator`). That means every automated labeling run depends on network access to
third-party services, is subject to their rate limits/availability, and only reliably produces one
kind of signal (a paraphrase pair) — the other two tiers (adjacent / random) are positional
heuristics, not measured similarity.

A second, fully offline method gives a fallback when translation services are unavailable or
undesired, and produces a real computed similarity score for every pair instead of heuristic tier
constants. It must not become the default — back-translation stays the default automated method,
matching how `--auto-label` itself stays opt-in relative to manual labeling.

Separately, neither existing automated-labeling method (back-translation or manual) has any
`README.md` documentation today. This spec's documentation task covers both, not just the new one.

## Solution

Add `--auto-label-method {backtranslation,svd}` (default `backtranslation`) to `run_pipeline.py`.
Every existing `--auto-label` invocation with no method flag behaves identically to today.
`--auto-label-method svd` selects a new, fully offline TF-IDF + truncated SVD (LSA) method.

### 1. `src/langembed/annotation/svd_label.py` (new)

Peer module to `auto_label.py`, same shape, no dependency between them:

```python
MAX_FIT_SENTENCES = 200_000  # bounds TF-IDF+SVD fit cost regardless of corpus size

def build_svd_sts_pairs(
    sentences: list[str],
    n: int,
    n_components: int = 100,
    seed: int = 42,
) -> list[tuple[str, str, float]]: ...
```

Steps:

1. If `len(sentences) > MAX_FIT_SENTENCES`, uniformly subsample down to `MAX_FIT_SENTENCES` with
   `random.Random(seed)` before fitting — the same bounded-memory principle just applied to
   `dedup()` (batched `MinHashLSH`) and `train_simcse()` (`MAX_TRAIN_EXAMPLES` reservoir sample),
   both fixed this session after corpus-size-proportional memory blew up gu's pipeline twice.
2. Fit `sklearn.feature_extraction.text.TfidfVectorizer` then
   `sklearn.decomposition.TruncatedSVD(n_components=n_components, random_state=seed)` on the
   (possibly subsampled) sentence set, producing one dense vector per sentence.
3. Sample `n` random pairs (`rng.sample(range(len(fit_sentences)), 2)`, repeated `n` times) from
   that same fitted set — no separate corpus pass, no tiering.
4. Score each pair by cosine similarity of its two SVD vectors, scaled to
   `score = clamp(similarity, 0.0, 1.0) * 5.0` to match `eval.yaml`'s `score_scale: 5.0` (the same
   scale back-translation's fixed tier constants already target). TF-IDF features are
   non-negative, so similarity is effectively always in `[0, 1]`; the clamp exists only as a
   floating-point safety margin, not because negative values are expected.
5. Return `list[(sentence_a, sentence_b, score)]` — same tuple shape `write_sts_pairs` already
   accepts, so no changes needed there.

`build_svd_sts_pairs` returns `[]` if `len(sentences) < 2`, matching `build_auto_sts_pairs`'s
existing guard.

### 2. `scripts/run_pipeline.py`

New CLI flags:

```
--auto-label-method {backtranslation,svd}   default "backtranslation"
--svd-components                            default 100
```

New function, mirroring `generate_auto_sts`:

```python
def generate_svd_sts(
    corpus_path: str,
    sts_test_path: str,
    n_components: int,
    n_labels: int,
) -> int:
    """Auto-label branch of pipeline step 5, SVD variant: build silver STS pairs via
    TF-IDF+SVD cosine similarity and write them to `sts_test_path`. Fully offline —
    no network calls, no docker/server/human dependency."""
```

Step 5 branching becomes three-way instead of two-way:

- `args.auto_label` and `args.auto_label_method == "backtranslation"` (default): existing
  `generate_auto_sts` path, unchanged byte-for-byte.
- `args.auto_label` and `args.auto_label_method == "svd"`: new `generate_svd_sts` path.
  Back-translation-only flags (`--translate-providers`, `--pivot-lang`, `--translate-rpm`) are
  simply unused in this branch — no validation error if the caller passes them anyway, they just
  have no effect, consistent with argparse's normal handling of unused defaulted flags.
- default (no `--auto-label`): existing manual `/label` flow, unchanged.

`eval_cfg` gains one field:

```python
"label_method": args.auto_label_method if args.auto_label else None,
```

placed next to the existing `label_source` field, so eval runs stay traceable to which method
produced the silver labels — SVD's continuous scores have a very different distribution than
back-translation's 3 fixed values, the same reasoning that motivated `label_source` originally.

### 3. `pyproject.toml`

Add to the existing `ml` optional-dependency group (SVD is a core numerical technique, not a
separate concern the way `translate` is):

```toml
ml = [
  ...
  "scikit-learn>=1.4",
]
```

### 4. Documentation — `README.md` and `docs/ru/README_RU.md`

New section, "Automated labeling (no human annotator)" (mirrored in Russian in the RU doc,
matching its existing structure/style), covering:

- What `--auto-label` replaces (the blocking manual `/label` step) and why it exists — currently
  undocumented despite already being merged.
- Both methods side by side: back-translation (needs network access to free MT services, produces
  one measured tier + two heuristic tiers) vs. SVD (fully offline, produces a real computed score
  for every pair).
- Concrete `run_pipeline.py` command examples for each:
  ```bash
  # back-translation (default method once --auto-label is set)
  python scripts/run_pipeline.py --lang gu --raw-input data/raw/gu_nllb.txt --auto-label

  # SVD (fully offline)
  python scripts/run_pipeline.py --lang gu --raw-input data/raw/gu_nllb.txt \
      --auto-label --auto-label-method svd --svd-components 100
  ```
- A short note that `label_method` in the generated `metrics/eval_<lang>.json` records which one
  ran.

## Data integrity

Pairs are drawn from `corpus_<lang>.txt`, exactly like back-translation's and the existing
`seed_sts_pairs.py`'s candidates — the same `train_paths: []` / STS-overlaps-training-corpus
caveat documented in the back-translation spec applies unchanged. No new leakage surface.

## Testing

- `tests/test_svd_label.py`: `build_svd_sts_pairs` returns `n` pairs; every score is in `[0, 5]`;
  a corpus below `MAX_FIT_SENTENCES` uses every sentence unmodified (no subsampling path taken);
  a corpus above it is subsampled (bounded fit-set size) while still returning `n` valid pairs;
  determinism under a fixed `seed`; `len(sentences) < 2` returns `[]`.
- `tests/test_run_pipeline.py`: extend with a test that `--auto-label-method svd` calls
  `generate_svd_sts` (not `generate_auto_sts`) and that omitting `--auto-label-method` after
  `--auto-label` still calls `generate_auto_sts` (default-preserving regression guard).

## Out of scope

- Any pair-selection mode beyond uniform random for SVD (e.g., nearest-neighbor "paraphrase-like"
  tiering) — confirmed with the user during design; real cosine similarity already gives adequate
  score spread without manufacturing tiers.
- A third auto-label method — this spec adds exactly one alternative to back-translation.
- The embeddings-to-LLM flow requested separately — explicitly deferred to its own brainstorming
  cycle after this spec ships (different subsystem: architecture/training, not labeling).
