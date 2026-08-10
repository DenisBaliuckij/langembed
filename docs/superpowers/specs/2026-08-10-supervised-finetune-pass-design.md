# Per-Method Supervised Fine-Tuning Pass — Design

**Date:** 2026-08-10
**Status:** Approved

## Problem

Each language currently produces exactly one final `output/<lang>/embeddings.jsonl`, from
unsupervised SimCSE — which doesn't use STS labels at all, so `--auto-label-method` (`svd` vs
`backtranslation`) and manual labeling today only affect the *eval* step, not the embeddings
themselves. To get three genuinely different embedding files per language — one per label
source — each method's labels need to feed a supervised fine-tuning step on top of the shared
SimCSE model. `src/langembed/contrastive/train_supervised.py` already implements exactly this
(triplet-based `MultipleNegativesRankingLoss` fine-tuning) but was explicitly kept out of
`run_pipeline.py` in an earlier design (`docs/superpowers/plans/2026-08-05-automated-sts-labeling.md`);
`run_pipeline.py` already templates a `supervised` config block for it but never invokes it.

`train_supervised.py` consumes `(anchor, positive, negative)` triplets, but the two automated
label methods (`svd`, `backtranslation`) produce `(sentence_a, sentence_b, score)` pairs, not
triplets — a conversion is needed. The native-speaker method is different: the annotation
service's `/export` endpoint (`src/langembed/annotation/api.py`) already performs its own
pair→triplet conversion (`_build_triplets`, fixed `score>=4.0`/`<=1.0` thresholds) before writing
`data/native_triplets_<lang>.jsonl` — so the native path needs no new conversion logic, only
consumption of a file that depends on real human annotators actually using the (currently
undeployed) annotation service.

## Solution

A new standalone script, `scripts/supervised_finetune_pass.py --lang <lang> --label-method
{svd,backtranslation,native}`, run once per method per language, after that language's base
pipeline (`run_pipeline.py`) has already produced `artifacts/simcse_<lang>` and
`data/corpus_<lang>.txt`. Mirrors `scripts/svd_eval_pass.py`'s established pattern: reuse
already-trained artifacts, do only the method-specific work, no re-running of
corpus/tokenizer/pretrain/SimCSE (none of which depend on label method).

### 1. `src/langembed/annotation/triplets.py` (new)

```python
def build_triplets_from_pairs(
    pairs: list[tuple[str, str, float]],
    positive_percentile: float = 0.7,
    negative_percentile: float = 0.3,
    seed: int = 42,
) -> list[tuple[str, str, str]]:
    """Convert scored (sentence_a, sentence_b, score) pairs into (anchor, positive,
    negative) triplets. Pairs scoring at or above `positive_percentile` (as a
    percentile of this batch's own score distribution, not a fixed absolute
    threshold) become positive candidates; pairs at or below `negative_percentile`
    become negative candidates. Positive and negative candidates are shuffled
    independently (seeded) and zipped, so triplet count is
    min(len(positive_candidates), len(negative_candidates)).

    Percentile-based (not `api.py::_build_triplets`'s fixed score>=4.0/<=1.0
    thresholds) because SVD's scores are continuous and its pairs are uniformly
    random by design (see docs/superpowers/specs/2026-08-09-svd-auto-labeling-design.md)
    -- most random sentence pairs score low-to-mid, so a fixed high threshold could
    starve the positive bucket. A percentile split adapts to whatever distribution a
    method actually produces; back-translation's discrete tiers (4.8/2.0/0.3) still
    split sensibly under it.
    """
```

Uses `numpy.percentile` (already a project dependency, via `scipy`/`sentence-transformers`) to
compute the score cutoffs, then partitions and zips. Independent of `api.py` — no import between
them, no shared DB coupling; `api.py`'s existing `_build_triplets`/`/export` for the native path
is untouched.

### 2. `scripts/supervised_finetune_pass.py` (new)

```python
def get_triplets(lang: str, label_method: str, n_labels: int, n_components: int) -> Path:
    """Returns the path to a triplets JSONL file for `label_method`, generating it
    first for svd/backtranslation, or locating the pre-existing
    data/native_triplets_<lang>.jsonl for native (raising FileNotFoundError with a
    clear message -- mirroring train_supervised.py's own "run Phase 5 and POST
    /export first" error -- if it doesn't exist yet)."""

def run_supervised_finetune_pass(
    lang: str, label_method: str, n_labels: int = 60, n_components: int = 100
) -> None:
    """Full pass: get_triplets -> train_supervised -> embed_corpus, writing
    output/<lang>/embeddings_<label_method>.jsonl."""
```

For `svd`/`backtranslation`: calls the existing `build_svd_sts_pairs`/`build_auto_sts_pairs`
(same defaults as `run_pipeline.py`'s auto-label stage: `n_labels=60`, `n_components=100`), then
`build_triplets_from_pairs`, writing `data/triplets_<lang>_<label_method>.jsonl`.

For `native`: no generation step — `get_triplets` just checks
`data/native_triplets_<lang>.jsonl` exists and returns that path directly.

Then, matching `svd_eval_pass.py`'s YAML-for-inspectability convention:

```python
supervised_cfg = {
    "seed": 42,
    "supervised": {
        "triplets_path": str(triplets_path),
        "in_dir": f"artifacts/simcse_{lang}",
        "out_dir": f"artifacts/embed_{lang}_{label_method}",
        "batch_size": 32,
        "epochs": 3,
        "warmup_steps": 100,
    },
}
```

written to `configs/<lang>/supervised_<label_method>.yaml` (matching `run_pipeline.py`'s existing
`contrastive_cfg["supervised"]` values exactly — same `batch_size`/`epochs`/`warmup_steps` it
already templates but never used), then `train_supervised(supervised_cfg)`.

Final step: `embed_corpus(embed_config_path, out_path=f"output/{lang}/embeddings_{label_method}.jsonl")`
against a config pointing `simcse.out_dir` at the just-fine-tuned `artifacts/embed_<lang>_<label_method>`
(not the shared unsupervised `simcse_<lang>`) and `simcse.sentences_path` at the existing
`data/corpus_<lang>.txt` — reusing `embed_corpus.py`'s existing chunked-encoding memory safety
unchanged.

### Data flow

1. `run_pipeline.py` runs once per language, producing the shared `artifacts/simcse_<lang>`,
   `data/corpus_<lang>.txt` (unchanged from tonight).
2. `supervised_finetune_pass.py --lang <lang> --label-method svd` and
   `--label-method backtranslation` can run any time after step 1 — fully automatable, no human
   dependency.
3. `--label-method native` requires the annotation service to be deployed and real annotators to
   have labeled and exported triplets first — this script's own guard makes that dependency
   explicit and loud rather than silently producing an empty/wrong result.
4. End state per language (once all 3 have run): `output/<lang>/embeddings_svd.jsonl`,
   `embeddings_backtranslation.jsonl`, `embeddings_native.jsonl`, alongside the original
   unsupervised `embeddings.jsonl` from `run_pipeline.py` (kept, not replaced).

## Testing

- `tests/test_triplets.py`: `build_triplets_from_pairs` returns triplets built from
  percentile-selected positive/negative candidates; triplet count equals
  `min(len(positive), len(negative))`; deterministic under a fixed seed; a small input where
  every pair has the same score (no meaningful percentile split) is handled without raising
  (produces whatever the percentile boundaries resolve to, even if candidates overlap/degenerate)
  — assert it doesn't crash rather than asserting a specific count in that edge case.
- `tests/test_supervised_finetune_pass.py`: `get_triplets("native", ...)` raises
  `FileNotFoundError` with a clear message when `data/native_triplets_<lang>.jsonl` is absent;
  `get_triplets("svd"/"backtranslation", ...)` generates and writes a triplets file (mocking
  `build_svd_sts_pairs`/`build_auto_sts_pairs` to avoid real ML work, matching
  `test_svd_eval_pass.py`'s existing mocking convention); `run_supervised_finetune_pass` calls
  `train_supervised` then `embed_corpus` with the right config values (mocked, verifying the
  config dict contents and call order — not a real training run).

## Out of scope

- Deploying the annotation service itself (nginx, per-language Basic Auth, `docker-compose.pilot.yml`)
  — that infrastructure was already designed and prepared earlier this session; deploying it is a
  separate operational task, tracked independently, not part of this spec.
- Wiring this into `run_pipeline.py`'s main flow as a flag — deliberately a separate script invoked
  per method, consistent with `svd_eval_pass.py`'s standalone pattern, since not every language run
  needs all 3 (or any) supervised passes and the native method's human-dependency doesn't fit a
  single synchronous pipeline invocation anyway.
- Changing `api.py`'s existing `_build_triplets`/`/export` fixed-threshold logic — untouched; the
  native path's triplets keep coming from that existing, separately-designed mechanism.
- A fourth combined/ensemble embedding — the spec produces exactly 3 method-specific embeddings
  per language plus the pre-existing unsupervised one; combining them is not requested.
