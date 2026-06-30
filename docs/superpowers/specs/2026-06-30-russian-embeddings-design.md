# Russian Embeddings from voina-i-mir.pdf — Design

**Date:** 2026-06-30
**Status:** Approved

## Problem

We have a from-scratch sentence-embedding pipeline built and tested for Gujarati
(`configs/*.yaml`, `data/raw/wiki_gu.txt`) plus an English fixture smoke path
(`configs/smoke/`). The user wants real sentence embeddings produced from
`data/raw/voina-i-mir.pdf` (Russian, "War and Peace"). The machine has no CUDA
(`torch.cuda.is_available() == False`), so full production-scale settings
(200k MLM steps, hidden 512/6 layers) are impractical. There is also no
Russian STS test set, and Phase 4 supervised / Phase 5 native triplets require
human-labeled data that doesn't exist yet.

## Solution

Add a new language track `ru` that reuses every existing pipeline module
unchanged (`preprocess.normalize`, `dedup`, `build_corpus`, `train_tokenizer`,
`train_mlm`, `train_simcse`, `evaluate`, `serve`) — only new configs, one new
data-prep module, two small operational scripts, and an additive HTML
labeling UI on the existing annotation API. Supervised contrastive (Phase 4
supervised) and branches B/C are **out of scope**: no native triplets exist
and none were requested. The model track stops at `artifacts/simcse_ru`
(Branch A only). The user will personally label ~30 Russian STS pairs through
a small browser UI hosted via the existing `docker-compose.yml` services, to
get a real (if small) Phase 6 eval and to allow `make eval` to run.

## Components

### 1. PDF extraction (`src/langembed/data/extract_text.py`)

```python
def extract_pdf_text(pdf_path: str | Path) -> str: ...
    # concatenate page text via pypdf

def split_sentences(text: str) -> list[str]: ...
    # regex split on .!?… followed by whitespace + capital/quote;
    # filters page numbers, footnote markers, very short fragments
```

CLI: `python -m langembed.data.extract_text --pdf data/raw/voina-i-mir.pdf --out data/raw/voina_i_mir_ru.txt`
writes one sentence per line, UTF-8 — the exact format `build_corpus.py`
already expects from `raw_paths` entries (it reads each raw file line-by-line
as a "doc"). No changes to `build_corpus.py`, `dedup.py`, or `preprocess.py`.

New dependency: `pypdf>=4` added to the base `dependencies` list in
`pyproject.toml` (lightweight parsing lib, not gated behind the `ml` extra).

### 2. Configs (`configs/ru/`)

Mirrors `configs/smoke/`'s structure, one file per phase, paths suffixed `_ru`.

`configs/ru/tokenizer.yaml`:
```yaml
language: ru
data:
  raw_paths:
    - data/raw/voina_i_mir_ru.txt
  out_path: data/corpus_ru.txt
  test_path: data/sts_test_ru.jsonl   # absent until Phase 5 labeling — guard is a no-op until then
tokenizer:
  vocab_size: 16000
  min_frequency: 2
  unk_rate_max: 0.01
  out_dir: artifacts/tokenizer_ru
```

`configs/ru/pretrain.yaml` — smaller-than-gu architecture for CPU feasibility:
```yaml
seed: 42
tokenizer_dir: artifacts/tokenizer_ru
corpus_path: data/corpus_ru.txt
out_dir: artifacts/encoder_ru
model:
  hidden_size: 256
  num_hidden_layers: 4
  num_attention_heads: 4
  intermediate_size: 1024
  max_position_embeddings: 130
  max_seq_length: 128
training:
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 1
  learning_rate: 0.0005
  weight_decay: 0.01
  warmup_steps: 200
  max_steps: <calibrated>   # set after a short timed run, target ~20-30 min wall-clock
  fp16: false
  save_steps: 500
  logging_steps: 50
  mlm_probability: 0.15
smoke:
  max_steps: 50   # reused as the calibration run, not just for --smoke
```

`configs/ru/contrastive.yaml`:
```yaml
seed: 42
encoder_dir: artifacts/encoder_ru
simcse:
  sentences_path: data/corpus_ru.txt
  out_dir: artifacts/simcse_ru
  batch_size: 32
  epochs: 1
  warmup_steps: 100
  max_seq_length: 128
supervised:
  triplets_path: data/native_triplets_ru.jsonl   # intentionally never produced
  in_dir: artifacts/simcse_ru
  out_dir: artifacts/embed_ru_v1
  batch_size: 32
  epochs: 3
  warmup_steps: 100
```
(`make supervised` for ru is never run; `train_supervised.py` already exits
cleanly with a guidance message if the triplets file is missing — no code
change needed.)

`configs/ru/eval.yaml`:
```yaml
test_path: data/sts_test_ru.jsonl
score_scale: 5.0
retrieval_k: 5
branches:
  A: artifacts/simcse_ru
train_paths:
  - data/corpus_ru.txt
metrics_path: metrics/eval_ru.json
```

### 3. Makefile targets

Add explicit `-ru` targets (no DVC pipeline for this track — out of scope,
plain `python -m` invocations like the existing root targets):

```makefile
extract-pdf-ru:
	$(PY) -m langembed.data.extract_text --pdf data/raw/voina-i-mir.pdf --out data/raw/voina_i_mir_ru.txt
corpus-ru:
	$(PY) -m langembed.data.build_corpus --config configs/ru/tokenizer.yaml
tokenizer-ru:
	$(PY) -m langembed.tokenizer.train_tokenizer --config configs/ru/tokenizer.yaml
pretrain-ru:
	$(PY) -m langembed.pretrain.train_mlm --config configs/ru/pretrain.yaml
simcse-ru:
	$(PY) -m langembed.contrastive.train_simcse --config configs/ru/contrastive.yaml
eval-ru:
	$(PY) -m langembed.eval.evaluate --config configs/ru/eval.yaml
seed-sts-ru:
	$(PY) scripts/seed_sts_pairs.py --config configs/ru/contrastive.yaml --n 60
embed-ru:
	$(PY) scripts/embed_corpus.py --config configs/ru/contrastive.yaml
```

### 4. STS-pair labeling UI (Phase 5, scoped down)

`scripts/seed_sts_pairs.py`: builds a candidate pool from `data/corpus_ru.txt`
(adjacent-sentence pairs for likely-related context + random distant pairs
for likely-unrelated context), scores all candidates with the existing
`active_learning.uncertainty()` against `artifacts/simcse_ru`, caps each
sentence's reuse across pairs, and inserts the ~60 most informative
(cosine nearest 0.5) as `Item` rows (`status="pending"`) via the existing
`db.py`/`models.py` — no schema changes.

New endpoints on `src/langembed/annotation/api.py` (additive; existing
`/queue`, `/annotate`, `/export` are untouched):

```python
@app.get("/label")          # HTMLResponse: next pending Item + 1-5 score form
@app.post("/label")         # form fields item_id, score; annotator_id fixed at 1;
                             # records Annotation, sets Item.status = "labeled", redirects to /label
@app.get("/export-sts")     # writes {sentence_a, sentence_b, score} JSONL for every
                             # labeled item to data/sts_test_ru.jsonl (default out_path)
```

Plain inline HTML via `fastapi.responses.HTMLResponse` — no template engine
dependency. Single fixed annotator (id=1, the user) means
`quality.aggregate()` on one label trivially returns that label, so no new
aggregation logic is needed.

Run via the existing `docker-compose.yml` services, unmodified:
```bash
docker compose up -d postgres annotation
# label at http://localhost:8001/label
curl -X POST "localhost:8001/export-sts"
```

### 5. Final embeddings deliverable

`scripts/embed_corpus.py`: loads `artifacts/simcse_ru`, encodes every line of
`data/corpus_ru.txt`, writes `artifacts/embeddings_ru/embeddings.jsonl` as
`{"text": ..., "embedding": [...]}` per line — this is the "embeddings from
the file" deliverable.

Verification (Phase 7 train/serve-skew check, per the project's existing
acceptance criterion): start `serve.py` with
`LANGEMBED_MODEL_DIR=artifacts/simcse_ru`, `POST /embed` a sample of book
sentences, and diff returned vectors against the batch script's output for
the same sentences — must match exactly (both paths call
`preprocess.normalize` + the same model).

## Data Flow

```
data/raw/voina-i-mir.pdf
  --(extract_text)-->        data/raw/voina_i_mir_ru.txt   (one sentence/line)
  --(build_corpus)-->        data/corpus_ru.txt            (normalized, deduped, leak-guarded)
  --(train_tokenizer)-->     artifacts/tokenizer_ru/
  --(train_mlm, bounded)-->  artifacts/encoder_ru/
  --(train_simcse)-->        artifacts/simcse_ru/
  --(seed_sts_pairs)-->      Postgres Item rows (uncertainty-ranked candidates)
  --(user labels via /label, 8001)--> Postgres Annotation rows
  --(/export-sts)-->         data/sts_test_ru.jsonl
  --(evaluate, configs/ru/eval.yaml)--> metrics/eval_ru.json   (Spearman + Recall@5/MRR@5 for branch A)
  --(embed_corpus.py)-->     artifacts/embeddings_ru/embeddings.jsonl
  --(serve.py /embed, cross-check)--> skew verification
```

## Error Handling

- `extract_pdf_text` / `split_sentences`: pure functions, no special error
  handling beyond what `pypdf` raises natively for a corrupt/missing PDF.
- `build_corpus`'s leakage guard is a no-op until `data/sts_test_ru.jsonl`
  exists (matches existing `load_test_hashes` behavior on a missing file —
  no smoke-style placeholder hack needed, unlike the English smoke pipeline).
- `train_supervised.py`'s existing guard (`SystemExit` with a guidance
  message) is left untouched and simply never exercised for `ru`.
- `evaluate.py`'s `assert_no_leakage` runs for real against
  `data/corpus_ru.txt` once `sts_test_ru.jsonl` exists.

## Testing

- Unit tests for `split_sentences` in `tests/test_extract_text.py`:
  deterministic edge cases (abbreviation handling, quotation marks, multiple
  terminators, junk-line filtering).
- Extend `tests/test_annotation_api.py` with contract tests for `/label`
  (GET renders a pending item, POST persists an Annotation and advances the
  queue) and `/export-sts` (writes correct JSONL schema, skips unlabeled
  items), following the existing `db_session`/`client` fixture pattern in
  `tests/conftest.py`.
- All new code follows existing conventions: heavy ML imports
  (`torch`/`transformers`/`sentence_transformers`/`pypdf`'s page-parsing path
  is fine at module level since it's not part of the ML stack) stay inside
  functions where the project pattern requires it; `ruff format`/`ruff
  check`/`mypy` clean; line length 100.

## Commits (one per phase, per CLAUDE.md)

1. `feat(ru): PDF extraction + ru corpus/tokenizer configs` — extract_text.py
   + tests + configs/ru/tokenizer.yaml + Makefile targets + corpus/tokenizer
   acceptance run.
2. `feat(ru): bounded MLM pretrain` — configs/ru/pretrain.yaml (calibrated
   max_steps) + Makefile target + pretrain acceptance run.
3. `feat(ru): unsupervised SimCSE` — configs/ru/contrastive.yaml + Makefile
   target + simcse acceptance run.
4. `feat(ru): STS labeling UI` — annotation/api.py additions + seed_sts_pairs.py
   + tests + docker compose labeling round (data not committed).
5. `feat(ru): eval` — configs/ru/eval.yaml + eval acceptance run
   (metrics/eval_ru.json committed, per the gu convention of committing
   metrics files).
6. `feat(ru): embeddings deliverable + serve skew verification` —
   embed_corpus.py + Makefile target + verification script/notes.

No `data/` or `artifacts/` content is ever committed (already gitignored,
matching current repo state — confirmed nothing under `data/` is tracked).

## Constraints

- No GPU; all training configs are CPU-bound and intentionally smaller than
  the gu production configs.
- No Russian STS benchmark exists — `data/sts_test_ru.jsonl` is built from
  scratch via the new labeling UI, with the explicit caveat that ~30
  user-labeled pairs is a sanity check, not a rigorous benchmark.
- Phase 4 supervised, Phase 6 branches B/C, and Phase 4C (LLM2Vec) are out of
  scope for this work.
- Existing `dvc.yaml`, `smoke/dvc.yaml`, and all gu/English-fixture
  configs/tests are untouched.
