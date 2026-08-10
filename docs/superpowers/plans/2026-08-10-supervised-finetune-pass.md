# Per-Method Supervised Fine-Tuning Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce 3 distinct fine-tuned embedding files per language (`svd`, `backtranslation`, `native`), by wiring the existing (never-invoked) `train_supervised.py` into a new standalone script, with a new percentile-based converter turning each auto method's scored pairs into training triplets.

**Architecture:** A new pure function `build_triplets_from_pairs` (percentile-based positive/negative split, zipped into triplets) plus a new orchestration script `scripts/supervised_finetune_pass.py --lang <lang> --label-method {svd,backtranslation,native}` that reuses the already-trained SimCSE model and corpus — no re-running corpus/tokenizer/pretrain/SimCSE.

**Tech Stack:** `numpy.percentile` (already a transitive dependency via `scipy`/`sentence-transformers`, already imported at module level in `src/langembed/eval/evaluate.py`) for the percentile split; existing `train_supervised()`, `build_svd_sts_pairs()`, `build_auto_sts_pairs()` reused unchanged.

## Global Constraints

- `build_triplets_from_pairs` uses **percentile-based** thresholds (`positive_percentile=0.7`, `negative_percentile=0.3` defaults), not fixed absolute thresholds — required so it works for SVD's continuous, uniformly-random-by-design scores as well as back-translation's discrete tiers.
- The native-speaker method does **not** generate anything — it locates the pre-existing `data/native_triplets_<lang>.jsonl` (written by the already-existing, unmodified annotation service `/export` endpoint) and raises `FileNotFoundError` with a clear message if it's absent.
- `src/langembed/annotation/api.py`'s existing `_build_triplets`/`/export` stays completely untouched — no changes to that file in this plan.
- `run_pipeline.py`'s main flow stays completely untouched — this is a separate, standalone script, not a new flag on the existing pipeline.
- Heavy/optional ML imports stay function-local, matching every other module in this codebase.
- `embed_corpus.py` lives in `scripts/`, not the `src/langembed` package, so it cannot be imported directly — invoke it via `subprocess.run`, mirroring exactly how `run_pipeline.py` already invokes it.
- `ruff format`, `ruff check`, and `mypy` must stay clean; line length 100.

---

### Task 1: `triplets.py` — percentile-based pair-to-triplet converter

**Files:**
- Create: `src/langembed/annotation/triplets.py`
- Test: `tests/test_triplets.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `build_triplets_from_pairs(pairs: list[tuple[str, str, float]], positive_percentile: float = 0.7, negative_percentile: float = 0.3, seed: int = 42) -> list[tuple[str, str, str]]`, used by Task 2's `get_triplets`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_triplets.py`:

```python
from langembed.annotation.triplets import build_triplets_from_pairs


def test_build_triplets_from_pairs_back_translation_tiers_split_correctly():
    """With 3 discrete tiers evenly split (paraphrase=4.8, adjacent=2.0,
    random=0.3, 20 each), the default 70th/30th percentile split recovers
    exactly the paraphrase tier as positive and the random tier as negative."""
    pairs = (
        [(f"para{i}", f"parab{i}", 4.8) for i in range(20)]
        + [(f"adj{i}", f"adjb{i}", 2.0) for i in range(20)]
        + [(f"rand{i}", f"randb{i}", 0.3) for i in range(20)]
    )

    triplets = build_triplets_from_pairs(pairs, seed=1)

    assert len(triplets) == 20
    for anchor, positive, negative in triplets:
        assert anchor.startswith("para")
        assert positive.startswith("parab")
        assert negative.startswith("randb")


def test_build_triplets_from_pairs_returns_min_of_bucket_sizes():
    pairs = [
        ("a1", "b1", 4.8),
        ("a2", "b2", 4.8),
        ("a3", "b3", 4.8),
        ("a4", "b4", 2.0),
        ("a5", "b5", 2.0),
        ("a6", "b6", 0.3),
    ]

    triplets = build_triplets_from_pairs(
        pairs, positive_percentile=0.7, negative_percentile=0.3, seed=1
    )

    assert len(triplets) >= 1
    for anchor, positive, negative in triplets:
        assert anchor.startswith("a")
        assert positive.startswith("b")
        assert negative.startswith("b")


def test_build_triplets_from_pairs_deterministic():
    pairs = [(f"a{i}", f"b{i}", float(i % 5)) for i in range(20)]

    t1 = build_triplets_from_pairs(pairs, seed=7)
    t2 = build_triplets_from_pairs(pairs, seed=7)

    assert t1 == t2


def test_build_triplets_from_pairs_empty_input_returns_empty():
    assert build_triplets_from_pairs([]) == []


def test_build_triplets_from_pairs_all_same_score_does_not_crash():
    pairs = [(f"a{i}", f"b{i}", 3.0) for i in range(10)]

    triplets = build_triplets_from_pairs(pairs, seed=1)

    assert isinstance(triplets, list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_triplets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'langembed.annotation.triplets'`

- [ ] **Step 3: Write the implementation**

Create `src/langembed/annotation/triplets.py`:

```python
"""Convert scored sentence pairs into (anchor, positive, negative) training
triplets -- see docs/superpowers/specs/2026-08-10-supervised-finetune-pass-design.md."""

from __future__ import annotations

import random

import numpy as np


def build_triplets_from_pairs(
    pairs: list[tuple[str, str, float]],
    positive_percentile: float = 0.7,
    negative_percentile: float = 0.3,
    seed: int = 42,
) -> list[tuple[str, str, str]]:
    """Convert scored (sentence_a, sentence_b, score) pairs into (anchor, positive,
    negative) triplets. Pairs scoring at or above the `positive_percentile` of this
    batch's own score distribution become positive candidates; pairs at or below
    `negative_percentile` become negative candidates. Positive and negative
    candidates are shuffled independently (seeded) and zipped, so the returned
    triplet count is min(len(positive_candidates), len(negative_candidates)).

    Percentile-based rather than a fixed absolute threshold (unlike
    langembed.annotation.api's _build_triplets) because some label methods produce
    continuous, uniformly-distributed scores where a fixed high threshold could
    starve the positive bucket -- a percentile split adapts to whatever
    distribution a given method actually produces.
    """
    if not pairs:
        return []

    scores = [score for _, _, score in pairs]
    pos_cutoff = float(np.percentile(scores, positive_percentile * 100))
    neg_cutoff = float(np.percentile(scores, negative_percentile * 100))

    positive = [(a, b) for a, b, score in pairs if score >= pos_cutoff]
    negative = [(a, b) for a, b, score in pairs if score <= neg_cutoff]

    rng = random.Random(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)

    triplets: list[tuple[str, str, str]] = []
    for (anchor, pos_sentence), (_, neg_sentence) in zip(positive, negative, strict=False):
        triplets.append((anchor, pos_sentence, neg_sentence))
    return triplets
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_triplets.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Lint and type-check**

Run:
```bash
ruff format src/langembed/annotation/triplets.py tests/test_triplets.py
ruff check src/langembed/annotation/triplets.py tests/test_triplets.py
mypy src/langembed/annotation/triplets.py
```
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/langembed/annotation/triplets.py tests/test_triplets.py
git commit -m "feat(annotation): add percentile-based pair-to-triplet converter"
```

---

### Task 2: `supervised_finetune_pass.py` — per-method fine-tuning orchestration

**Files:**
- Create: `scripts/supervised_finetune_pass.py`
- Test: `tests/test_supervised_finetune_pass.py`

**Interfaces:**
- Consumes: `build_triplets_from_pairs` from `langembed.annotation.triplets` (Task 1); `build_svd_sts_pairs` from `langembed.annotation.svd_label` (pre-existing); `build_auto_sts_pairs` from `langembed.annotation.auto_label` (pre-existing); `train_supervised` from `langembed.contrastive.train_supervised` (pre-existing); `scripts/embed_corpus.py` via subprocess (pre-existing, unchanged).
- Produces: `get_triplets(lang: str, label_method: str, n_labels: int, n_components: int) -> Path` and `run_supervised_finetune_pass(lang: str, label_method: str, n_labels: int = 60, n_components: int = 100) -> None`. Nothing later in this plan depends on this task (it's the last task).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_supervised_finetune_pass.py`:

```python
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "supervised_finetune_pass",
    Path(__file__).resolve().parent.parent / "scripts" / "supervised_finetune_pass.py",
)
assert _SPEC is not None and _SPEC.loader is not None
supervised_finetune_pass = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(supervised_finetune_pass)


def test_get_triplets_native_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(supervised_finetune_pass, "REPO_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="native_triplets_ru.jsonl"):
        supervised_finetune_pass.get_triplets("ru", "native", n_labels=60, n_components=100)


def test_get_triplets_native_returns_existing_path(monkeypatch, tmp_path):
    monkeypatch.setattr(supervised_finetune_pass, "REPO_ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    native_path = tmp_path / "data" / "native_triplets_ru.jsonl"
    native_path.write_text(
        '{"anchor": "a", "positive": "b", "negative": "c"}\n', encoding="utf-8"
    )

    result = supervised_finetune_pass.get_triplets(
        "ru", "native", n_labels=60, n_components=100
    )

    assert result == native_path


def test_get_triplets_svd_generates_and_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(supervised_finetune_pass, "REPO_ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "corpus_ru.txt").write_text(
        "\n".join(f"sentence {i}" for i in range(10)), encoding="utf-8"
    )

    from langembed.annotation import svd_label

    monkeypatch.setattr(
        svd_label,
        "build_svd_sts_pairs",
        lambda sentences, n, n_components: [("a", "b", 4.8)] * n,
    )

    result = supervised_finetune_pass.get_triplets("ru", "svd", n_labels=6, n_components=3)

    assert result == tmp_path / "data" / "triplets_ru_svd.jsonl"
    lines = result.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0
    row = json.loads(lines[0])
    assert set(row.keys()) == {"anchor", "positive", "negative"}


def test_run_supervised_finetune_pass_calls_train_supervised_and_embed_corpus(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(supervised_finetune_pass, "REPO_ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    native_path = tmp_path / "data" / "native_triplets_ru.jsonl"
    native_path.write_text(
        '{"anchor": "a", "positive": "b", "negative": "c"}\n', encoding="utf-8"
    )

    from langembed.contrastive import train_supervised as train_supervised_module

    seen_cfg = {}

    def fake_train_supervised(cfg):
        seen_cfg.update(cfg)

    monkeypatch.setattr(train_supervised_module, "train_supervised", fake_train_supervised)

    seen_subprocess_args = {}

    def fake_run(args, **kwargs):
        seen_subprocess_args["args"] = args
        seen_subprocess_args["kwargs"] = kwargs

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(supervised_finetune_pass.subprocess, "run", fake_run)

    supervised_finetune_pass.run_supervised_finetune_pass(
        "ru", "native", n_labels=60, n_components=100
    )

    assert seen_cfg["supervised"]["triplets_path"] == str(native_path)
    assert seen_cfg["supervised"]["in_dir"] == "artifacts/simcse_ru"
    assert seen_cfg["supervised"]["out_dir"] == "artifacts/embed_ru_native"
    assert any("scripts/embed_corpus.py" in str(a) for a in seen_subprocess_args["args"])
    assert "--out" in seen_subprocess_args["args"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_supervised_finetune_pass.py -v`
Expected: FAIL with `FileNotFoundError` (the script file doesn't exist yet, so `importlib.util.spec_from_file_location` produces a spec whose loader fails) or a collection error.

- [ ] **Step 3: Write the implementation**

Create `scripts/supervised_finetune_pass.py`:

```python
"""Per-method supervised fine-tuning pass: derives triplets for one label
method, fine-tunes the shared unsupervised SimCSE model on them, and produces
a method-specific final embeddings file.

See docs/superpowers/specs/2026-08-10-supervised-finetune-pass-design.md.

Usage:
    python scripts/supervised_finetune_pass.py --lang ru --label-method svd
    python scripts/supervised_finetune_pass.py --lang ru --label-method backtranslation
    python scripts/supervised_finetune_pass.py --lang ru --label-method native
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

LABEL_METHODS = ("svd", "backtranslation", "native")


def get_triplets(lang: str, label_method: str, n_labels: int, n_components: int) -> Path:
    """Return the path to a triplets JSONL file for `label_method`. Generates it
    first for svd/backtranslation; for native, locates the pre-existing
    data/native_triplets_<lang>.jsonl, raising FileNotFoundError if it doesn't
    exist yet (mirrors train_supervised.py's own "run Phase 5 and POST /export
    first" error).
    """
    if label_method == "native":
        native_path = REPO_ROOT / f"data/native_triplets_{lang}.jsonl"
        if not native_path.is_file() or native_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"No native triplets at {native_path}. Deploy the annotation service, "
                "have annotators label pairs, then POST /export first."
            )
        return native_path

    from langembed.annotation.triplets import build_triplets_from_pairs

    corpus_path = REPO_ROOT / f"data/corpus_{lang}.txt"
    with corpus_path.open(encoding="utf-8") as f:
        sentences = [ln.strip() for ln in f if ln.strip()]

    if label_method == "svd":
        from langembed.annotation.svd_label import build_svd_sts_pairs

        pairs = build_svd_sts_pairs(sentences, n=n_labels, n_components=n_components)
    elif label_method == "backtranslation":
        from langembed.annotation.auto_label import build_auto_sts_pairs

        cache_path = REPO_ROOT / f"data/backtranslation_cache_{lang}.jsonl"
        pairs = build_auto_sts_pairs(
            sentences,
            n=n_labels,
            providers=["google", "mymemory"],
            pivot_lang="en",
            source_lang=lang,
            cache_path=cache_path,
        )
    else:
        raise ValueError(f"unknown label_method: {label_method!r}")

    triplets = build_triplets_from_pairs(pairs)
    triplets_path = REPO_ROOT / f"data/triplets_{lang}_{label_method}.jsonl"
    triplets_path.parent.mkdir(parents=True, exist_ok=True)
    with triplets_path.open("w", encoding="utf-8") as f:
        for anchor, positive, negative in triplets:
            f.write(
                json.dumps(
                    {"anchor": anchor, "positive": positive, "negative": negative},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return triplets_path


def run_supervised_finetune_pass(
    lang: str, label_method: str, n_labels: int = 60, n_components: int = 100
) -> None:
    from langembed.contrastive.train_supervised import train_supervised

    print(f"=== [{lang}] supervised fine-tune ({label_method}) ===")
    triplets_path = get_triplets(lang, label_method, n_labels, n_components)
    print(f"  triplets: {triplets_path}")

    supervised_cfg: dict[str, Any] = {
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
    supervised_path = REPO_ROOT / "configs" / lang / f"supervised_{label_method}.yaml"
    supervised_path.parent.mkdir(parents=True, exist_ok=True)
    with supervised_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(supervised_cfg, f, allow_unicode=True, sort_keys=False)

    train_supervised(supervised_cfg)
    print(f"  fine-tuned model -> artifacts/embed_{lang}_{label_method}")

    embed_cfg = {
        "simcse": {
            "out_dir": f"artifacts/embed_{lang}_{label_method}",
            "sentences_path": f"data/corpus_{lang}.txt",
        }
    }
    embed_cfg_path = REPO_ROOT / "configs" / lang / f"embed_{label_method}.yaml"
    with embed_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(embed_cfg, f, allow_unicode=True, sort_keys=False)

    out_path = REPO_ROOT / f"output/{lang}/embeddings_{label_method}.jsonl"
    subprocess.run(
        [
            sys.executable,
            "scripts/embed_corpus.py",
            "--config",
            str(embed_cfg_path),
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    print(f"  wrote embeddings -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lang", required=True)
    ap.add_argument("--label-method", required=True, choices=LABEL_METHODS)
    ap.add_argument("--n-labels", type=int, default=60)
    ap.add_argument("--svd-components", type=int, default=100)
    args = ap.parse_args()
    run_supervised_finetune_pass(args.lang, args.label_method, args.n_labels, args.svd_components)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_supervised_finetune_pass.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -q --ignore=tests/e2e`
Expected: all pass, no regressions.

- [ ] **Step 6: Lint and type-check**

Run:
```bash
ruff format scripts/supervised_finetune_pass.py tests/test_supervised_finetune_pass.py
ruff check scripts/supervised_finetune_pass.py tests/test_supervised_finetune_pass.py
mypy scripts/supervised_finetune_pass.py
```
Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add scripts/supervised_finetune_pass.py tests/test_supervised_finetune_pass.py
git commit -m "feat(pipeline): add per-method supervised fine-tuning pass"
```
