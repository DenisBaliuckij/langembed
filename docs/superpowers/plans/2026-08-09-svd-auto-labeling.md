# SVD-Based Auto-Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--auto-label-method svd` to `run_pipeline.py` as a second, fully offline auto-labeling method alongside the existing back-translation method, plus bilingual README documentation for both.

**Architecture:** A new peer module `src/langembed/annotation/svd_label.py` fits TF-IDF + truncated SVD (LSA) over the corpus and scores random sentence pairs by cosine similarity. `run_pipeline.py`'s step 5 branching becomes three-way (svd / backtranslation / manual) instead of two-way, gated by a new `--auto-label-method` flag that defaults to `backtranslation` so every existing invocation is unaffected.

**Tech Stack:** scikit-learn (`TfidfVectorizer`, `TruncatedSVD`, `cosine_similarity`) — new dependency in the `ml` extras group. Python stdlib `random` for sampling, matching `auto_label.py` and `dedup.py`'s existing conventions.

## Global Constraints

- New dependency: `scikit-learn>=1.4`, added to the `ml` group in `pyproject.toml` (not a new group — this is a core numerical technique, unlike `translate`'s standalone `deep-translator`).
- `--auto-label-method` defaults to `"backtranslation"` — every existing `--auto-label` command must behave byte-for-byte identically to before this change.
- Follow the project's "no magic numbers in training/processing code" convention: the fit-size bound is a named module constant (`MAX_FIT_SENTENCES = 200_000`), not a bare literal, mirroring `dedup.py`'s `DEFAULT_BATCH_SIZE` and `train_simcse.py`'s `MAX_TRAIN_EXAMPLES`.
- Heavy/optional imports (`sklearn.*`) stay function-local, matching every other ML-dependent module in this codebase (`CLAUDE.md`: "ML-импорты... делаются внутри функций").
- Scores must land in `[0, score_scale]` (`score_scale: 5.0` in `configs/<lang>/eval.yaml`), matching back-translation's existing `PARAPHRASE_SCORE`/`ADJACENT_SCORE`/`RANDOM_SCORE` range.
- `ruff format`, `ruff check`, and `mypy` must stay clean; line length 100.

---

### Task 1: `svd_label.py` — core pair-building module

**Files:**
- Create: `src/langembed/annotation/svd_label.py`
- Test: `tests/test_svd_label.py`
- Modify: `pyproject.toml` (add `scikit-learn>=1.4` to the `ml` extras list)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `MAX_FIT_SENTENCES: int` (module constant) and
  `build_svd_sts_pairs(sentences: Sequence[str], n: int, n_components: int = 100, seed: int = 42, max_fit_sentences: int = MAX_FIT_SENTENCES) -> list[tuple[str, str, float]]`,
  used by Task 2's `generate_svd_sts`. Also relies on `write_sts_pairs` from
  `langembed.annotation.auto_label` (already exists, unchanged) — Task 2 wires that call, not
  this task.

- [ ] **Step 1: Add the scikit-learn dependency**

In `pyproject.toml`, find the `ml = [...]` list (currently ends with `"scipy>=1.11",`) and add one
line so the block reads:

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
  "scikit-learn>=1.4",
]
```

Install it into the active environment: `pip install scikit-learn>=1.4` (or
`pip install -e ".[ml]"` if that's how the project's other ML deps got installed locally).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_svd_label.py`:

```python
from langembed.annotation import svd_label


def _sentences(n: int) -> list[str]:
    return [f"the quick brown fox sentence number {i} about topic {i % 4}" for i in range(n)]


def test_build_svd_sts_pairs_returns_n_pairs():
    pairs = svd_label.build_svd_sts_pairs(_sentences(20), n=9, n_components=3, seed=1)
    assert len(pairs) == 9


def test_build_svd_sts_pairs_scores_in_range():
    pairs = svd_label.build_svd_sts_pairs(_sentences(20), n=15, n_components=3, seed=1)
    for _, _, score in pairs:
        assert 0.0 <= score <= 5.0


def test_build_svd_sts_pairs_pairs_come_from_input_sentences():
    sentences = _sentences(20)
    pairs = svd_label.build_svd_sts_pairs(sentences, n=9, n_components=3, seed=1)
    for a, b, _ in pairs:
        assert a in sentences
        assert b in sentences


def test_build_svd_sts_pairs_subsamples_above_max_fit_sentences():
    pairs = svd_label.build_svd_sts_pairs(
        _sentences(20), n=9, n_components=3, seed=1, max_fit_sentences=5
    )
    used = {a for a, _, _ in pairs} | {b for _, b, _ in pairs}
    assert len(used) <= 5


def test_build_svd_sts_pairs_no_subsampling_below_threshold():
    """When the corpus is at or below max_fit_sentences, the whole corpus is used
    unmodified -- two calls with different (but both non-restrictive) max_fit_sentences
    values must produce identical output, since neither actually subsamples."""
    sentences = _sentences(20)
    a = svd_label.build_svd_sts_pairs(sentences, n=9, n_components=3, seed=7, max_fit_sentences=20)
    b = svd_label.build_svd_sts_pairs(
        sentences, n=9, n_components=3, seed=7, max_fit_sentences=1000
    )
    assert a == b


def test_build_svd_sts_pairs_deterministic():
    kwargs = dict(sentences=_sentences(20), n=9, n_components=3, seed=7)
    assert svd_label.build_svd_sts_pairs(**kwargs) == svd_label.build_svd_sts_pairs(**kwargs)


def test_build_svd_sts_pairs_too_few_sentences_returns_empty():
    assert svd_label.build_svd_sts_pairs(["only one sentence"], n=9) == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_svd_label.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'langembed.annotation.svd_label'`

- [ ] **Step 4: Write the implementation**

Create `src/langembed/annotation/svd_label.py`:

```python
"""Silver-standard STS pair generation via TF-IDF + truncated SVD (LSA) -- fully offline,
no network calls, no human labeler needed. A second --auto-label-method alongside
back-translation (see auto_label.py); not the default."""

from __future__ import annotations

import random
from collections.abc import Sequence

# Bounds TF-IDF+SVD fit cost regardless of corpus size -- gu's ~13.8M-sentence corpus is
# why train_simcse and dedup both needed the same kind of bound this session; LSA needs far
# fewer documents than neural training to capture corpus-level semantic structure, so this
# constant is an order of magnitude smaller than train_simcse.MAX_TRAIN_EXAMPLES.
MAX_FIT_SENTENCES = 200_000


def build_svd_sts_pairs(
    sentences: Sequence[str],
    n: int,
    n_components: int = 100,
    seed: int = 42,
    max_fit_sentences: int = MAX_FIT_SENTENCES,
) -> list[tuple[str, str, float]]:
    """Silver STS pairs scored by real cosine similarity in TF-IDF+SVD (LSA) space --
    unlike back-translation's three fixed-score tiers, every pair gets a computed score.

    If `sentences` is larger than `max_fit_sentences`, a uniform random subsample is fit
    instead of the full corpus (bounded memory/time regardless of corpus size); pairs are
    then drawn only from that fit set. Returns [] if fewer than 2 sentences are available
    to pair.
    """
    if len(sentences) < 2:
        return []

    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    rng = random.Random(seed)
    fit_sentences = (
        rng.sample(list(sentences), max_fit_sentences)
        if len(sentences) > max_fit_sentences
        else list(sentences)
    )

    tfidf = TfidfVectorizer().fit_transform(fit_sentences)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    vectors = svd.fit_transform(tfidf)

    pairs: list[tuple[str, str, float]] = []
    for _ in range(n):
        i, j = rng.sample(range(len(fit_sentences)), 2)
        similarity = cosine_similarity(vectors[i : i + 1], vectors[j : j + 1])[0, 0]
        score = max(0.0, min(1.0, float(similarity))) * 5.0
        pairs.append((fit_sentences[i], fit_sentences[j], score))
    return pairs
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_svd_label.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 6: Lint and type-check**

Run:
```bash
ruff format src/langembed/annotation/svd_label.py tests/test_svd_label.py
ruff check src/langembed/annotation/svd_label.py tests/test_svd_label.py
mypy src/langembed/annotation/svd_label.py
```
Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/langembed/annotation/svd_label.py tests/test_svd_label.py
git commit -m "feat(annotation): add TF-IDF+SVD auto-labeling as a second --auto-label-method"
```

---

### Task 2: `run_pipeline.py` — wire up `--auto-label-method`

**Files:**
- Modify: `scripts/run_pipeline.py`
- Test: `tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: `build_svd_sts_pairs` and `MAX_FIT_SENTENCES` from
  `langembed.annotation.svd_label` (Task 1); `write_sts_pairs` from
  `langembed.annotation.auto_label` (pre-existing).
- Produces: `generate_svd_sts(corpus_path: str, sts_test_path: str, n_components: int, n_labels: int) -> int`,
  a new `--auto-label-method` / `--svd-components` CLI surface, and an `eval_cfg["label_method"]`
  field. Nothing later depends on this task within this plan (it's the last code task before
  docs).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_run_pipeline.py` (after the existing `test_generate_auto_sts_*` tests, before
`test_auto_label_help_text_discloses_external_services`):

```python
def test_generate_svd_sts_writes_pairs(tmp_path):
    corpus = tmp_path / "corpus_ru.txt"
    corpus.write_text("\n".join(f"sentence about topic {i % 4} number {i}" for i in range(20)), encoding="utf-8")
    sts_out = tmp_path / "sts_test_ru.jsonl"

    n = run_pipeline.generate_svd_sts(str(corpus), str(sts_out), 3, 9)

    assert n == 9
    lines = sts_out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 9
    row = json.loads(lines[0])
    assert set(row.keys()) == {"sentence_a", "sentence_b", "score"}


def test_generate_svd_sts_resolves_relative_paths_against_repo_root(monkeypatch, tmp_path):
    fake_repo_root = tmp_path / "fake_repo"
    (fake_repo_root / "data").mkdir(parents=True)
    (fake_repo_root / "data" / "corpus_de.txt").write_text(
        "\n".join(f"sentence about topic {i % 4} number {i}" for i in range(20)), encoding="utf-8"
    )
    monkeypatch.setattr(run_pipeline, "REPO_ROOT", fake_repo_root)

    other_cwd = tmp_path / "somewhere_else"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    n = run_pipeline.generate_svd_sts("data/corpus_de.txt", "data/sts_test_de.jsonl", 3, 9)

    assert n == 9
    assert (fake_repo_root / "data" / "sts_test_de.jsonl").exists()


def test_svd_label_cli_flag_defaults():
    ap = run_pipeline.build_arg_parser()
    args = ap.parse_args(["--lang", "ru", "--input", "book.pdf"])

    assert args.auto_label_method == "backtranslation"
    assert args.svd_components == 100


def test_svd_label_cli_flag_set():
    ap = run_pipeline.build_arg_parser()
    args = ap.parse_args(
        [
            "--lang", "ru", "--input", "book.pdf",
            "--auto-label", "--auto-label-method", "svd", "--svd-components", "50",
        ]
    )

    assert args.auto_label_method == "svd"
    assert args.svd_components == 50


def test_auto_label_method_rejects_unknown_value():
    ap = run_pipeline.build_arg_parser()
    import pytest

    with pytest.raises(SystemExit):
        ap.parse_args(["--lang", "ru", "--input", "book.pdf", "--auto-label-method", "bogus"])


def test_eval_cfg_records_label_method():
    """Same source-text-check approach as test_eval_cfg_records_label_source (main() runs
    a long unmocked subprocess pipeline with no test coverage by design)."""
    source = (Path(__file__).resolve().parent.parent / "scripts" / "run_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert '"label_method": args.auto_label_method if args.auto_label else None' in source


def test_main_branches_to_svd_when_method_is_svd():
    """Same source-text-check approach as test_eval_cfg_records_label_source."""
    source = (Path(__file__).resolve().parent.parent / "scripts" / "run_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert 'args.auto_label and args.auto_label_method == "svd"' in source
    assert "generate_svd_sts(" in source
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_run_pipeline.py -v -k "svd or label_method"`
Expected: FAIL — `generate_svd_sts` doesn't exist, `--auto-label-method`/`--svd-components` are
unrecognized arguments, and the two source-text assertions don't find their strings yet.

- [ ] **Step 3: Add the CLI flags**

In `scripts/run_pipeline.py`, inside `build_arg_parser()`, immediately after the existing
`--translate-rpm` argument block (right before `return ap`), add:

```python
    ap.add_argument(
        "--auto-label-method",
        choices=["backtranslation", "svd"],
        default="backtranslation",
        help=(
            "which automated method to use when --auto-label is set: 'backtranslation' "
            "(default, needs network access to free MT services) or 'svd' (fully offline "
            "TF-IDF + truncated SVD cosine similarity)"
        ),
    )
    ap.add_argument(
        "--svd-components",
        type=int,
        default=100,
        help="SVD dimensionality for --auto-label-method svd (ignored otherwise)",
    )
```

- [ ] **Step 4: Add `generate_svd_sts`**

Immediately after the existing `generate_auto_sts` function (before `def main() -> None:`), add:

```python
def generate_svd_sts(
    corpus_path: str,
    sts_test_path: str,
    n_components: int,
    n_labels: int,
) -> int:
    """Auto-label branch of pipeline step 5, SVD variant: build silver STS pairs via
    TF-IDF+SVD cosine similarity and write them to `sts_test_path`. Returns the number of
    pairs written. Fully offline -- no network calls, no docker/server/human dependency.

    `corpus_path` and `sts_test_path` are resolved against REPO_ROOT, like every other
    path in this file, so the pipeline behaves the same regardless of the process's CWD.
    """
    from langembed.annotation.auto_label import write_sts_pairs
    from langembed.annotation.svd_label import build_svd_sts_pairs

    print("  (fully offline: TF-IDF + truncated SVD, no external services)")

    corpus_abs = _resolve_repo_path(corpus_path)
    sts_test_abs = _resolve_repo_path(sts_test_path)

    with corpus_abs.open(encoding="utf-8") as f:
        sentences = [ln.strip() for ln in f if ln.strip()]
    pairs = build_svd_sts_pairs(sentences, n=n_labels, n_components=n_components)
    print(f"  {len(pairs)} pairs -> {sts_test_abs}")
    return write_sts_pairs(pairs, sts_test_abs)
```

- [ ] **Step 5: Make step 5 branching three-way**

In `main()`, replace:

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
```

with:

```python
    if not args.skip_eval:
        if args.auto_label and args.auto_label_method == "svd":
            print(f"=== [{lang}] 5/6 auto-label STS pairs (SVD, no human) ===")
            n_written = generate_svd_sts(
                corpus_path, sts_test_path, args.svd_components, args.n_labels
            )
            print(f"  wrote {n_written} auto-labeled STS pairs -> {sts_test_path}")
        elif args.auto_label:
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
```

(The `else:` branch and everything below it — the manual-labeling flow — is untouched.)

- [ ] **Step 6: Record `label_method` in `eval_cfg`**

In `main()`, inside the `eval_cfg = {...}` dict, immediately after the existing line:

```python
            "label_source": "auto" if args.auto_label else "manual",
```

add:

```python
            "label_method": args.auto_label_method if args.auto_label else None,
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_run_pipeline.py -v`
Expected: PASS (all tests, including every pre-existing one — confirms the default path is
unchanged)

- [ ] **Step 8: Lint and type-check**

Run:
```bash
ruff format scripts/run_pipeline.py tests/test_run_pipeline.py
ruff check scripts/run_pipeline.py tests/test_run_pipeline.py
mypy scripts/run_pipeline.py
```
Expected: all clean.

- [ ] **Step 9: Run the full test suite**

Run: `pytest tests/ -q --ignore=tests/e2e`
Expected: all pass, no regressions.

- [ ] **Step 10: Commit**

```bash
git add scripts/run_pipeline.py tests/test_run_pipeline.py
git commit -m "feat(pipeline): wire --auto-label-method svd into run_pipeline.py"
```

---

### Task 3: Documentation — `README.md` and `docs/ru/README_RU.md`

**Files:**
- Modify: `README.md`
- Modify: `docs/ru/README_RU.md`

**Interfaces:**
- Consumes: the final flag names/behavior from Task 2 (`--auto-label`, `--auto-label-method`,
  `--svd-components`, `--translate-providers`, `--pivot-lang`, `--translate-rpm`,
  `label_source`/`label_method` in `eval.yaml`). No other task depends on this one.

- [ ] **Step 1: Add the English section**

In `README.md`, the `## Annotation service and active learning` section ends right before
`## Evaluation` (look for the line `## Evaluation` — the new section goes immediately above it,
after the `---` separator that precedes `## Evaluation`). Insert:

```markdown
## Automated labeling (no human annotator)

Phase 5 (`=== [{lang}] 5/6 ... ===`) normally blocks on a human opening `http://localhost:PORT/label`
and rating STS pairs 1–5 (see [Annotation service and active learning](#annotation-service-and-active-learning)).
Passing `--auto-label` to `scripts/run_pipeline.py` skips that step entirely and generates
silver-standard STS pairs automatically instead, so the whole pipeline can run unattended
start-to-finish. Manual labeling stays the default — `--auto-label` is opt-in.

Two methods are available via `--auto-label-method`:

| Method | Flag value | How it works | Network? |
|---|---|---|---|
| Back-translation (default) | `backtranslation` | Round-trips corpus sentences through a free MT service and back; produces one measured "paraphrase" tier plus two positional heuristic tiers (adjacent / random sentence pairs) with fixed scores | Yes — calls Google/MyMemory via `deep-translator` |
| SVD (LSA) | `svd` | Fits TF-IDF + truncated SVD over the corpus and scores random sentence pairs by real cosine similarity in that space — every pair gets a computed score, not a fixed tier constant | No — fully offline |

```bash
# back-translation (default method once --auto-label is set)
python scripts/run_pipeline.py --lang gu --raw-input data/raw/gu_nllb.txt --auto-label

# back-translation with explicit tuning
python scripts/run_pipeline.py --lang gu --raw-input data/raw/gu_nllb.txt \
    --auto-label --translate-providers google mymemory --pivot-lang en --translate-rpm 20

# SVD (fully offline, no external services)
python scripts/run_pipeline.py --lang gu --raw-input data/raw/gu_nllb.txt \
    --auto-label --auto-label-method svd --svd-components 100
```

The generated `configs/<lang>/eval.yaml` records which mode produced the STS test set via
`label_source` (`"auto"` / `"manual"`) and, for automated runs, `label_method`
(`"backtranslation"` / `"svd"`) — back-translation's 3 fixed score values and SVD's continuous
scores have very different distributions, so this keeps eval runs from the two methods from being
compared blind.

---

```

Then update the table of contents. Replace this exact block (lines 11–29 of `README.md`):

```markdown
1. [What this is](#what-this-is)
2. [Architecture overview](#architecture-overview)
3. [Repository structure](#repository-structure)
4. [Installation](#installation)
5. [Quick start — smoke pipeline](#quick-start--smoke-pipeline)
6. [Production pipeline (full data)](#production-pipeline-full-data)
7. [DVC in depth](#dvc-in-depth)
8. [Configuration reference](#configuration-reference)
9. [Pipeline phases in detail](#pipeline-phases-in-detail)
10. [Creating and using embeddings](#creating-and-using-embeddings)
11. [Serving — /embed endpoint](#serving--embed-endpoint)
12. [Annotation service and active learning](#annotation-service-and-active-learning)
13. [Evaluation](#evaluation)
14. [MLflow experiment tracking](#mlflow-experiment-tracking)
15. [Testing](#testing)
16. [Docker and docker-compose](#docker-and-docker-compose)
17. [Adapting to another language](#adapting-to-another-language)
18. [Makefile reference](#makefile-reference)
19. [Troubleshooting](#troubleshooting)
```

with:

```markdown
1. [What this is](#what-this-is)
2. [Architecture overview](#architecture-overview)
3. [Repository structure](#repository-structure)
4. [Installation](#installation)
5. [Quick start — smoke pipeline](#quick-start--smoke-pipeline)
6. [Production pipeline (full data)](#production-pipeline-full-data)
7. [DVC in depth](#dvc-in-depth)
8. [Configuration reference](#configuration-reference)
9. [Pipeline phases in detail](#pipeline-phases-in-detail)
10. [Creating and using embeddings](#creating-and-using-embeddings)
11. [Serving — /embed endpoint](#serving--embed-endpoint)
12. [Annotation service and active learning](#annotation-service-and-active-learning)
13. [Automated labeling (no human annotator)](#automated-labeling-no-human-annotator)
14. [Evaluation](#evaluation)
15. [MLflow experiment tracking](#mlflow-experiment-tracking)
16. [Testing](#testing)
17. [Docker and docker-compose](#docker-and-docker-compose)
18. [Adapting to another language](#adapting-to-another-language)
19. [Makefile reference](#makefile-reference)
20. [Troubleshooting](#troubleshooting)
```

- [ ] **Step 2: Add the Russian section**

In `docs/ru/README_RU.md`, `## Сервис разметки и active learning` ends right before
`## Оценка качества`. Insert immediately above `## Оценка качества`:

```markdown
## Автоматическая разметка (без участия человека)

Фаза 5 (`=== [{lang}] 5/6 ... ===`) обычно блокируется на входе человека на
`http://localhost:PORT/label` для оценки пар STS по шкале 1–5 (см.
[Сервис разметки и active learning](#сервис-разметки-и-active-learning)). Флаг `--auto-label`
у `scripts/run_pipeline.py` полностью пропускает этот шаг и вместо этого автоматически
генерирует silver-эталонные пары STS, так что весь пайплайн может отработать без
присутствия человека от начала до конца. Ручная разметка остаётся режимом по умолчанию —
`--auto-label` включается явно.

Доступны два метода через `--auto-label-method`:

| Метод | Значение флага | Как работает | Нужна сеть? |
|---|---|---|---|
| Обратный перевод (по умолчанию) | `backtranslation` | Прогоняет предложения корпуса через бесплатный MT-сервис туда и обратно; даёт один измеренный «парафразный» уровень плюс два позиционных эвристических уровня (соседние / случайные пары предложений) с фиксированными оценками | Да — обращается к Google/MyMemory через `deep-translator` |
| SVD (LSA) | `svd` | Строит TF-IDF + усечённое SVD-разложение по корпусу и оценивает случайные пары предложений по реальному косинусному сходству в этом пространстве — каждая пара получает вычисленную оценку, а не фиксированную константу уровня | Нет — полностью офлайн |

```bash
# обратный перевод (метод по умолчанию при включённом --auto-label)
python scripts/run_pipeline.py --lang gu --raw-input data/raw/gu_nllb.txt --auto-label

# обратный перевод с явной настройкой
python scripts/run_pipeline.py --lang gu --raw-input data/raw/gu_nllb.txt \
    --auto-label --translate-providers google mymemory --pivot-lang en --translate-rpm 20

# SVD (полностью офлайн, без внешних сервисов)
python scripts/run_pipeline.py --lang gu --raw-input data/raw/gu_nllb.txt \
    --auto-label --auto-label-method svd --svd-components 100
```

Сгенерированный `configs/<lang>/eval.yaml` фиксирует, какой режим создал тестовый набор STS,
через поле `label_source` (`"auto"` / `"manual"`) и, для автоматических запусков, поле
`label_method` (`"backtranslation"` / `"svd"`) — три фиксированных значения оценки у обратного
перевода и непрерывные оценки SVD имеют очень разные распределения, поэтому это не даёт вслепую
сравнивать метрики двух методов.

---

```

Then update the table of contents. Replace this exact block (lines 11–29 of
`docs/ru/README_RU.md`):

```markdown
1. [Что это такое](#что-это-такое)
2. [Обзор архитектуры](#обзор-архитектуры)
3. [Структура репозитория](#структура-репозитория)
4. [Установка](#установка)
5. [Быстрый старт — smoke-пайплайн](#быстрый-старт--smoke-пайплайн)
6. [Продакшн-пайплайн (полные данные)](#продакшн-пайплайн-полные-данные)
7. [DVC: подробное руководство](#dvc-подробное-руководство)
8. [Справочник по конфигурации](#справочник-по-конфигурации)
9. [Фазы пайплайна в деталях](#фазы-пайплайна-в-деталях)
10. [Создание и использование эмбеддингов](#создание-и-использование-эмбеддингов)
11. [Сервинг — эндпоинт /embed](#сервинг--эндпоинт-embed)
12. [Сервис разметки и active learning](#сервис-разметки-и-active-learning)
13. [Оценка качества](#оценка-качества)
14. [Трекинг экспериментов в MLflow](#трекинг-экспериментов-в-mlflow)
15. [Тестирование](#тестирование)
16. [Docker и docker-compose](#docker-и-docker-compose)
17. [Адаптация под другой язык](#адаптация-под-другой-язык)
18. [Справочник по Makefile](#справочник-по-makefile)
19. [Решение проблем](#решение-проблем)
```

with:

```markdown
1. [Что это такое](#что-это-такое)
2. [Обзор архитектуры](#обзор-архитектуры)
3. [Структура репозитория](#структура-репозитория)
4. [Установка](#установка)
5. [Быстрый старт — smoke-пайплайн](#быстрый-старт--smoke-пайплайн)
6. [Продакшн-пайплайн (полные данные)](#продакшн-пайплайн-полные-данные)
7. [DVC: подробное руководство](#dvc-подробное-руководство)
8. [Справочник по конфигурации](#справочник-по-конфигурации)
9. [Фазы пайплайна в деталях](#фазы-пайплайна-в-деталях)
10. [Создание и использование эмбеддингов](#создание-и-использование-эмбеддингов)
11. [Сервинг — эндпоинт /embed](#сервинг--эндпоинт-embed)
12. [Сервис разметки и active learning](#сервис-разметки-и-active-learning)
13. [Автоматическая разметка (без участия человека)](#автоматическая-разметка-без-участия-человека)
14. [Оценка качества](#оценка-качества)
15. [Трекинг экспериментов в MLflow](#трекинг-экспериментов-в-mlflow)
16. [Тестирование](#тестирование)
17. [Docker и docker-compose](#docker-и-docker-compose)
18. [Адаптация под другой язык](#адаптация-под-другой-язык)
19. [Справочник по Makefile](#справочник-по-makefile)
20. [Решение проблем](#решение-проблем)
```

- [ ] **Step 3: Verify internal links resolve**

The anchor slug only appears literally in the TOC link — GitHub generates it from the heading
text at render time, so the heading line itself won't contain the slug string. Check each part
separately.

Run: `grep -n "automated-labeling-no-human-annotator" README.md`
Expected: exactly one match, the TOC line `13. [Automated labeling (no human annotator)](#automated-labeling-no-human-annotator)`.

Run: `grep -n "^## Automated labeling (no human annotator)$" README.md`
Expected: exactly one match, the heading itself. GitHub slugifies it to
`#automated-labeling-no-human-annotator` (lowercase, spaces to hyphens, parentheses dropped),
matching the TOC link above.

Run: `grep -n "автоматическая-разметка-без-участия-человека" docs/ru/README_RU.md`
Expected: exactly one match, the TOC line.

Run: `grep -n "^## Автоматическая разметка (без участия человека)$" docs/ru/README_RU.md`
Expected: exactly one match, the heading itself, slugifying the same way applied to Cyrillic text.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/ru/README_RU.md
git commit -m "docs: document --auto-label and --auto-label-method (backtranslation, svd) in EN/RU README"
```
