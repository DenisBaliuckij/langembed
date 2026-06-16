# Design: langembed prototype completion

**Date:** 2026-06-16  
**Status:** Approved

---

## Context

The `langembed` skeleton has all pipeline source modules implemented (phases 0–7 + 4C): preprocessing, dedup, corpus builder, tokenizer, MLM pretrain, SimCSE, supervised contrastive, LLM branch (LoRA), annotation service (FastAPI + SQLAlchemy + postgres), evaluator, serving API. Configs, Makefile, DVC DAG, CI, and docker-compose (postgres + redis) are in place.

**What is missing:**
1. `tests/test_tokenizer.py` (specified in plan, absent)
2. API tests for annotation service and `/embed` endpoint
3. Retrieval@k metric in `evaluate.py` (config has `retrieval_k` but code ignores it)
4. Docker application containers (only postgres + redis exist)
5. End-to-end English smoke test proving the full pipeline cycle
6. Russian-language documentation

---

## Approach

All gaps closed in one implementation session (no phased delivery). The skeleton is solid and the gaps are independent of each other.

---

## Section 1: Completing the prototype

### 1.1 Retrieval@k in `evaluate.py`

Add `InformationRetrievalEvaluator` from sentence-transformers alongside the existing `EmbeddingSimilarityEvaluator`. Uses the STS pairs as a retrieval benchmark: each `sentence_a` is a query, all `sentence_b` are the corpus; evaluates Recall@k and MRR@k. Results written to `metrics/eval.json` under keys `retrieval_recall@k_<branch>` and `retrieval_mrr@k_<branch>`. The `score_scale` normalisation applied to Spearman does not affect retrieval (labels not used there).

### 1.2 `tests/test_tokenizer.py`

Tests using a 20-sentence English fixture string (no file I/O needed):
- **Round-trip:** `decode(encode(normalize(x))) == normalize(x)` for several sentences
- **unk_rate:** train a tokenizer on fixture corpus, assert unk_rate < configured threshold
- Uses real `train_tokenizer()` + `diagnose()` functions; fast (sub-second on CPU)

### 1.3 `tests/conftest.py`

Pytest fixtures shared across API tests:
- `db_session`: SQLite in-memory engine, creates all tables, yields a `Session`, rolls back after test
- `client`: `TestClient` wrapping the annotation FastAPI app with `get_db` overridden to the test session

### 1.4 `tests/test_annotation_api.py`

Tests via FastAPI `TestClient` (no real postgres needed):
- `GET /queue` returns items ordered by `uncertainty DESC`, always includes gold items
- `POST /annotate` persists an `Annotation` row; subsequent `/queue` shows updated state
- `POST /export` builds correct triplets from labeled pairs (positives score ≥ 4.0, negatives ≤ 1.0)
- Edge case: `/export` on empty DB returns `{"written": 0}`

### 1.5 `tests/test_serve.py`

Tests the `/embed` endpoint via `TestClient`:
- Monkeypatches `_get_model` to return a deterministic fake encoder (returns `np.ones((n, 4))` normalised)
- Asserts: `normalize()` is applied before encoding (verified by capturing call args)
- Asserts: response shape `{"embeddings": [[...]], "dim": 4}` is correct
- Asserts: same text input → identical embedding (determinism)

---

## Section 2: End-to-end English smoke test

### Fixture files (committed to repo)

- `tests/fixtures/en_corpus.txt`: 200 English sentences across 5 semantic clusters (animals, food, technology, sport, weather). Handcrafted so similarity relationships are unambiguous.
- `tests/fixtures/en_sts_test.jsonl`: 20 sentence pairs with scores 0–5. Includes 5 obviously-similar pairs (score 5), 5 obviously-dissimilar pairs (score 0), and 10 intermediate pairs.

### `tests/e2e/test_pipeline_english.py`

Marked `@pytest.mark.e2e`; excluded from `make test`; run via `make test-e2e`.

Steps (all in `tmp_path`):

1. `build_corpus([fixture_corpus], out, test_hashes)` → assert file written, line count > 0
2. `train_tokenizer(corpus, out_dir, vocab_size=500, min_frequency=1)` → assert round-trip holds, unk_rate < 0.05
3. `train_mlm(cfg_override, smoke=True)` with tiny model (hidden=128, 2 layers, 4 heads, 50 steps, CPU, fp16=False) → assert `config.json` + weight files exist, final loss is finite
4. `train_simcse(cfg_override, smoke=True)` (first 20 sentences from fixture, 1 epoch) → assert model directory written
5. Load the saved SimCSE model, encode 4 sentence pairs: assert 2 similar pairs have cosine > 0.0, embeddings are L2-normalised (norm ≈ 1.0 ± 0.01)
6. `evaluate()` on `en_sts_test.jsonl` → assert pipeline completes, `metrics/eval.json` written, embedding dim = 128

**What this proves:** The full code path from raw text to Spearman output works without errors on CPU with no GPU and no real data. It does **not** prove model quality — Spearman after 50 MLM steps is expected to be low. The Russian docs document this explicitly.

### `pytest.ini` / `pyproject.toml` change

Add `e2e` to `markers` so pytest doesn't warn; add `make test-e2e` target to Makefile running `pytest -m e2e tests/e2e/`.

---

## Section 3: Docker full stack

### `Dockerfile` (multi-stage, at `langembed/Dockerfile`)

```
Stage base  : python:3.11-slim + .[serve]   → annotation service image (~400 MB)
Stage ml    : base + .[ml]                  → serving + training image (~4 GB with torch)
```

No model weights baked in — artifacts mounted as volumes.

### Updated `docker-compose.yml`

Three new services added alongside existing `postgres` and `redis`:

| Service | Dockerfile stage | Port | Purpose |
|---|---|---|---|
| `annotation` | `base` | 8001 | Native-speaker annotation API |
| `serve` | `ml` | 8000 | Embedding inference API |
| `train` | `ml` | — | One-shot training runner (`docker compose run train dvc repro`) |

Volume mounts:
- `annotation`: `./data:/app/data`
- `serve`: `./artifacts:/app/artifacts`, `./data:/app/data`
- `train`: `./data:/app/data`, `./artifacts:/app/artifacts`, `./configs:/app/configs`

All three depend on `postgres` and `redis` (via `depends_on`).

### `serve.py` change

Replace hardcoded `"artifacts/embed_gu_v1"` with `os.environ.get("LANGEMBED_MODEL_DIR", "artifacts/embed_gu_v1")` so the container can be pointed at different model artifacts without rebuilding.

### `.env.example` additions

```
LANGEMBED_MODEL_DIR=artifacts/embed_gu_v1
```

---

## Section 4: Russian documentation

**File:** `docs/ru/README_RU.md`

Sections:
1. Что это такое — задача, три ветки (A/B/C), зачем носители
2. Архитектура — фазы 0→7, инварианты (единая нормализация, запрет утечки теста, конфиги через YAML), взаимосвязь компонентов
3. Структура репозитория — каждый модуль и его назначение
4. Установка — локально (pip) и через Docker
5. Запуск пайплайна — пошагово: данные → `docker compose up` → `dvc repro` → `make eval`, примеры команд
6. Фазы 0–7 + 4C — что делает каждая, входы/выходы, ключевые параметры YAML
7. Сервис разметки и active learning — контур носителей, очередь, экспорт триплетов
8. Тестирование — юнит / API / E2E, как запустить, ожидаемые результаты E2E smoke-теста на английском
9. DVC и воспроизводимость — DAG, `dvc repro`, `dvc metrics show`
10. MLflow — просмотр метрик, сравнение A/B/C
11. Адаптация под другой язык — что менять в конфигах

---

## Definition of Done

- `make lint && make test` green (all existing + new unit/API tests pass without GPU)
- `make test-e2e` runs full English pipeline smoke on CPU and exits 0
- `docker compose up` starts all 5 services healthy
- `docker compose run train dvc repro` (given data files) runs the full pipeline
- `docs/ru/README_RU.md` covers all 11 sections
