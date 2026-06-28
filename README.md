# langembed

> **Русская документация:** [docs/ru/README_RU.md](docs/ru/README_RU.md)

A from-scratch sentence-embedding pipeline for low-resource languages, with a native-speaker active learning loop. Three model branches are trained on the same data and evaluated with the same metric, enabling a clean A/B/C architecture comparison.

---

## Table of contents

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

---

## What this is

The research goal is to train high-quality sentence embeddings for a language with limited resources (Gujarati is used as the example), then measure how much each architectural choice contributes to quality.

**Three branches trained on identical native-speaker data:**

| Branch | Approach | Key components |
|--------|----------|----------------|
| **A** | From scratch | Custom BPE tokenizer → RoBERTa-style encoder (MLM pre-train) → SimCSE contrastive fine-tune |
| **B** | Multilingual transfer | mBERT / XLM-R fine-tuned on the same native-speaker supervision |
| **C** | LLM as embedder | Decoder LLM (LLaMA-style) + LoRA adapters, mean-pool over last hidden states (llm2vec approach) |

**Native speakers are central:** they supply annotation signal through an active learning loop. The system surfaces the most *uncertain* sentence pairs to annotators, maximising the information value of each labelled example.

**Evaluation metric:** Spearman correlation on an isolated STS (Semantic Textual Similarity) test set. The test set never enters any training stage — enforced as an architectural invariant.

---

## Architecture overview

The pipeline runs as a sequence of phases (0–7) plus branch-C phase 4C:

| Phase | Name | Description |
|-------|------|-------------|
| 0 | Normalisation | NFKC + IndicNLP (Gujarati) + whitespace collapse via `preprocess.normalize` |
| 1 | Corpus | Aggregate `data/raw/*.txt`, MinHash-dedup, write `data/corpus.txt` |
| 2 | Tokenizer | BPE with Whitespace pre-tokenizer, NFKC normaliser, vocab_size=8000 |
| 3 | MLM pre-train | RoBERTa-style encoder, HuggingFace Trainer, logs to MLflow |
| 4 | SimCSE | Contrastive fine-tune (dropout-augmented positive pairs) via SentenceTransformers |
| 4C | LLM LoRA | LoRA-adapt a decoder LLM for embeddings; base weights frozen |
| 5 | Annotation | FastAPI service: active-learning queue, label collection, triplet export |
| 6 | Evaluation | Spearman, Recall@k, MRR@k on the held-out STS test |
| 7 | Serving | FastAPI `/embed` endpoint; L2-normalised vectors over HTTP |

**Three invariants — never violate:**

1. **Single normalisation source** — all text (train and serve) passes through `langembed.preprocess.normalize`. No duplicated logic.
2. **No test leakage** — `data/sts_test_*` files never enter any training stage. Guarded in `build_corpus.py` and `evaluate.py`.
3. **Config-driven hyperparameters** — all hyperparameters come from `configs/*.yaml`. No magic numbers in training code.

**Data flow:**

```
raw text files
    │
    ▼
build_corpus  ──(MinHash dedup)──► data/corpus.txt
    │
    ▼
train_tokenizer ──────────────────► artifacts/tokenizer/
    │
    ▼
train_mlm  ───(MLM pre-train)─────► artifacts/encoder/
    │
    ▼
train_simcse  ─(SimCSE fine-tune)─► artifacts/simcse/   ← Branch A
    │
    ├── annotation service (active learning loop)
    │       │
    │       ▼
    │   native_triplets.jsonl
    │       │
    │       ▼
    ├── train_supervised ──────────► artifacts/supervised/
    │
    ▼
evaluate ─────────────────────────► metrics/eval.json → MLflow
    │
    ▼
serve ────────────────────────────► POST /embed
```

---

## Repository structure

```
src/langembed/
  config.py              load_config() — single entry point for YAML configs
  preprocess.py          normalize() — the one text normalisation function
  data/
    build_corpus.py      aggregate + MinHash-dedup raw text files
    dedup.py             MinHash/LSH deduplication utilities
  tokenizer/
    train_tokenizer.py   BPE tokenizer (HuggingFace tokenizers library)
  pretrain/
    train_mlm.py         RoBERTa-style MLM pre-training (HuggingFace Trainer)
  contrastive/
    train_simcse.py      SimCSE contrastive fine-tune (SentenceTransformers)
    train_supervised.py  Supervised contrastive fine-tune on exported triplets
  llm_embed/
    train_lora.py        LoRA fine-tune of a decoder LLM for embeddings
    mntp.py              Masked next-token prediction pre-training (branch C)
    model.py             LLM embedding model wrapper
  annotation/
    api.py               FastAPI: /queue, /annotate, /export
    models.py            SQLAlchemy ORM: Annotator, Item, Annotation
    db.py                Database session / get_db dependency
    active_learning.py   Uncertainty scoring, queue selection
    quality.py           Weighted kappa, reliability-weighted aggregation
  eval/
    evaluate.py          Spearman + Recall@k + MRR@k
  serving/
    serve.py             FastAPI /embed endpoint

configs/
  tokenizer.yaml         Phase 1+2 (corpus + tokenizer)
  pretrain.yaml          Phase 3 (MLM pre-train)
  contrastive.yaml       Phase 4 (SimCSE / supervised)
  eval.yaml              Phase 6 (evaluation, branch paths)
  llm_embed.yaml         Phase 4C (LLM LoRA)
  smoke/                 Smoke pipeline configs (English fixtures, CPU-only)
    tokenizer.yaml
    pretrain.yaml
    contrastive.yaml
    eval.yaml

smoke/
  dvc.yaml               DVC smoke pipeline (5 stages, English fixture data)
  dvc.lock               DVC lock file (committed)

dvc.yaml                 Production DVC pipeline (full language data)
Makefile                 make lint | test | test-e2e | corpus | pretrain | …
Dockerfile               Multi-stage: base (serve extras) / ml (+ torch)
docker-compose.yml       postgres, redis, annotation :8001, serve :8000, train

tests/
  conftest.py            Shared fixtures (SQLite in-memory, TestClient)
  e2e/                   Full-pipeline English CPU smoke test
  fixtures/
    en_corpus.txt        English fixture sentences
    en_sts_test.jsonl    English STS pairs for E2E evaluation

data/                    DVC-tracked (not in git)
artifacts/               DVC-tracked model artifacts (not in git)
metrics/
  smoke_eval.json        Smoke pipeline metrics (committed, cache: false)
  eval.json              Production evaluation metrics (DVC-tracked)

docs/
  ru/README_RU.md        Russian documentation
  IMPLEMENTATION_PLAN.md Phase-by-phase implementation reference
```

---

## Installation

### Local (CPU or GPU)

```bash
git clone https://github.com/DenisBaliuckij/langembed
cd langembed
pip install -e ".[ml,serve,dev]"
cp .env.example .env        # fill in DATABASE_URL and REDIS_URL
```

**Dependencies by extra:**

| Extra | Installs | Use when |
|-------|----------|----------|
| `ml` | torch, transformers, sentence-transformers, datasets | Training phases |
| `serve` | fastapi, uvicorn, sqlalchemy, psycopg2 | Annotation + serving |
| `dev` | ruff, mypy, pytest | Development |

Install all three for local development:

```bash
pip install -e ".[ml,serve,dev]"
```

### Docker (full stack)

```bash
cp .env.example .env
# Edit .env: set POSTGRES_PASSWORD, REDIS_URL, DATABASE_URL

docker compose up -d postgres redis

# Base image (~400 MB): annotation service + serving, no torch
docker build --target base -t langembed-annotation .

# ML image (~4 GB): includes torch + all training deps
docker build --target ml -t langembed-ml .

docker compose up -d
```

Services after `docker compose up`:

| Service | Port | Purpose |
|---------|------|---------|
| `annotation` | 8001 | Active learning annotation API |
| `serve` | 8000 | `/embed` inference endpoint |
| `train` | — | One-shot training container |
| `postgres` | 5432 | Annotation storage |
| `redis` | 6379 | Task queue |

---

## Quick start — smoke pipeline

The smoke pipeline trains a tiny English model end-to-end in ~30 seconds on CPU, using fixture data from `tests/fixtures/`. It validates the entire code path without requiring a GPU or real language data.

```bash
# Run the full smoke DVC pipeline (5 stages)
make smoke-dvc

# Which is equivalent to:
python -m dvc repro smoke/dvc.yaml
```

**What happens:**

```
Stage 1: corpus    — dedup tests/fixtures/en_corpus.txt → data/smoke/corpus_en.txt
Stage 2: tokenizer — BPE vocab_size=500 → artifacts/smoke/tokenizer_en/
Stage 3: pretrain  — RoBERTa hidden=128, 50 steps → artifacts/smoke/encoder_en/
Stage 4: simcse    — SimCSE 1 epoch → artifacts/smoke/simcse_en/
Stage 5: evaluate  — Spearman on en_sts_test.jsonl → metrics/smoke_eval.json
```

**Expected output after first run:**

```
Running stage 'corpus':   ...
Running stage 'tokenizer': ...
Running stage 'pretrain':  ...
Running stage 'simcse':    ...
Running stage 'evaluate':  ...
branch en_smoke: Spearman = 0.6499
branch en_smoke: Recall@5=0.5000, MRR@5=0.3308
metrics written to metrics/smoke_eval.json
```

**Idempotency (second run — all cached):**

```bash
make smoke-dvc
# Stage 'corpus': cached
# Stage 'tokenizer': cached
# Stage 'pretrain': cached
# Stage 'simcse': cached
# Stage 'evaluate': cached
# All stages are up to date.
```

DVC tracks input hashes and only re-runs stages whose inputs changed.

**Inspect smoke metrics:**

```bash
dvc metrics show metrics/smoke_eval.json
# or simply:
cat metrics/smoke_eval.json
```

**Note on smoke metrics:** Spearman ≈ 0.65 on a 50-step model is expected to be lower than a fully trained model. The smoke pipeline is a code-correctness check, not a quality benchmark.

---

## Production pipeline (full data)

### 1. Prepare raw data

```bash
# Place raw text files (one sentence per line) in data/raw/
cp your_language_wiki.txt data/raw/wiki_gu.txt
cp your_other_corpus.txt  data/raw/news_gu.txt

# Prepare your STS test set (JSONL: sentence_a, sentence_b, score)
cp your_sts_test.jsonl data/sts_test_gu.jsonl
```

### 2. Start infrastructure

```bash
docker compose up -d postgres redis
```

### 3. Run the full DVC pipeline

```bash
dvc repro
```

The production `dvc.yaml` executes these stages in order:

```
corpus → tokenizer → pretrain → simcse → supervised → evaluate
```

### 4. Start annotation campaign (Phase 5)

```bash
make serve-annotation
# API docs: http://localhost:8001/docs
```

See [Annotation service and active learning](#annotation-service-and-active-learning).

### 5. Evaluate all branches

```bash
make eval
# Reads configs/eval.yaml (branches A, B, C)
# Writes metrics/eval.json + logs to MLflow
```

### 6. Serve embeddings

```bash
LANGEMBED_MODEL_DIR=artifacts/embed_gu_v1 make serve
# API docs: http://localhost:8000/docs
```

### 7. Inspect results

```bash
mlflow ui          # http://localhost:5000
dvc metrics show   # print metrics/eval.json
dvc dag            # visualise the pipeline DAG
```

---

## DVC in depth

### Core commands

```bash
# Run all stale pipeline stages (production)
dvc repro

# Run only the smoke pipeline
dvc repro smoke/dvc.yaml
# or:
make smoke-dvc

# Check which stages are stale without running them
dvc status

# Print current metrics
dvc metrics show

# Compare metrics across git commits / branches
dvc metrics diff HEAD~1

# Visualise the production DAG
dvc dag

# Visualise the smoke DAG
dvc dag smoke/dvc.yaml
```

### Remote storage (team collaboration)

```bash
# Configure an S3 remote (once per repo)
dvc remote add -d myremote s3://my-bucket/langembed
dvc remote modify myremote region eu-central-1

# Push all tracked data and model artifacts
dvc push

# Reproduce on another machine
git clone https://github.com/DenisBaliuckij/langembed
cd langembed
dvc pull       # download artifacts from remote
dvc repro      # re-run any stale stages
```

Any team member can reproduce the exact experiment with `dvc pull && dvc repro`.

### How DVC tracks files

- All outputs listed under `outs:` in `dvc.yaml` / `smoke/dvc.yaml` are content-hashed by DVC.
- Hashes are stored in `.dvc/cache` and in the lock files (`dvc.lock`, `smoke/dvc.lock`).
- Large files (model weights, datasets) are excluded from git via `.gitignore`.
- `metrics/smoke_eval.json` is committed to git directly (`cache: false`) so it appears in `git log` and `dvc metrics show` without needing a DVC remote.

### Smoke pipeline architecture note

The smoke pipeline lives in `smoke/dvc.yaml` (a subdirectory). Every stage has `wdir: ..` so all paths in configs (e.g. `data/smoke/corpus_en.txt`, `artifacts/smoke/tokenizer_en`) resolve relative to the repo root, not the `smoke/` subdirectory. This matches the conventions in `.gitignore` and config files.

---

## Configuration reference

All hyperparameters are in `configs/*.yaml`. Training code reads them via `langembed.config.load_config(path)`.

### `configs/tokenizer.yaml` (Phases 1–2)

```yaml
language: en               # language code for preprocess.normalize
data:
  raw_paths:               # list of raw text files to aggregate
    - data/raw/wiki_gu.txt
  out_path: data/corpus_gu.txt
  test_path: data/sts_test_gu.jsonl  # leakage guard: these sentences excluded from corpus
tokenizer:
  vocab_size: 8000
  min_frequency: 2
  unk_rate_max: 0.05       # fail if unknown-token rate exceeds this
  out_dir: artifacts/tokenizer_gu
```

### `configs/pretrain.yaml` (Phase 3)

```yaml
seed: 42
tokenizer_dir: artifacts/tokenizer_gu
corpus_path: data/corpus_gu.txt
report_to: [mlflow]        # experiment tracking backends
out_dir: artifacts/encoder_gu
model:
  hidden_size: 512
  num_hidden_layers: 6
  num_attention_heads: 8
  intermediate_size: 2048
  max_position_embeddings: 514
  max_seq_length: 512
training:
  per_device_train_batch_size: 64
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-4
  weight_decay: 0.01
  warmup_steps: 10000
  max_steps: 200000
  fp16: true               # set false on CPU
  save_steps: 10000
  logging_steps: 500
  mlm_probability: 0.15
smoke:
  max_steps: 50            # used only when --smoke flag is passed
```

### `configs/contrastive.yaml` (Phase 4)

```yaml
seed: 42
encoder_dir: artifacts/encoder_gu
simcse:
  sentences_path: data/corpus_gu.txt
  out_dir: artifacts/simcse_gu
  batch_size: 128
  epochs: 3
  warmup_steps: 100
  max_seq_length: 512
```

### `configs/eval.yaml` (Phase 6)

```yaml
test_path: data/sts_test_gu.jsonl
score_scale: 5.0           # divides raw scores to normalise to [0, 1]
retrieval_k: 10            # k for Recall@k and MRR@k
branches:                  # name → model directory
  A: artifacts/embed_gu_v1
  B: artifacts/embed_gu_mling
  C: artifacts/embed_gu_llm
train_paths:               # leakage guard: must not contain test sentences
  - data/corpus_gu.txt
  - data/native_triplets.jsonl
metrics_path: metrics/eval.json
```

### Smoke configs (`configs/smoke/`)

The smoke configs mirror production with smaller sizes:
- `vocab_size: 500` (vs 8000 in production)
- `hidden_size: 128`, `num_hidden_layers: 2` (vs 512 / 6)
- `max_steps: 50` via `--smoke` flag
- `batch_size: 8` (vs 128)
- `test_path: data/smoke/sts_test_placeholder.jsonl` — intentionally absent file; leakage guard returns empty set because fixture corpus sentences appear in the test set by design

---

## Pipeline phases in detail

### Phase 0 — Normalisation

`langembed.preprocess.normalize(text, lang="gu")` is the **single normalisation function** used everywhere — corpus building, training, and serving.

Steps applied:
1. NFKC Unicode normalisation
2. IndicNLP script normalisation (Gujarati by default; skipped automatically for non-Indic languages)
3. Whitespace collapse

No code change is needed for non-Indic languages — IndicNLP is skipped gracefully when `lang` is not an Indic code.

### Phase 1 — Corpus building

```bash
make corpus
# or: python -m langembed.data.build_corpus --config configs/tokenizer.yaml
```

Reads all files from `raw_paths`, normalises each sentence, removes duplicates using MinHash LSH (configurable Jaccard threshold), and writes `out_path`. The leakage guard excludes sentences that appear in the STS test set (`test_path`).

### Phase 2 — BPE tokenizer

```bash
make tokenizer
# or: python -m langembed.tokenizer.train_tokenizer --config configs/tokenizer.yaml
```

Trains a Byte-Pair Encoding tokenizer (HuggingFace `tokenizers` library) with Whitespace pre-tokenizer and NFKC normaliser. Artifact: `artifacts/tokenizer_gu/` (vocabulary + merges + special tokens).

### Phase 3 — MLM pre-training

```bash
# Full training (200k steps, GPU recommended)
make pretrain

# CPU smoke (50 steps, hidden=128)
make pretrain-smoke
```

Trains a RoBERTa-style encoder from scratch using HuggingFace Trainer. Masked Language Modelling with 15% masking probability. Loss and perplexity are logged to MLflow. Artifact: `artifacts/encoder_gu/`.

**Production settings:** 200,000 steps, batch_size=64, gradient_accumulation=4 (effective batch=256), hidden_size=512, 6 layers, 8 heads, fp16=true.

### Phase 4 — SimCSE contrastive fine-tuning

```bash
# Full training
make simcse

# CPU smoke (1 epoch, 256 sentences)
make simcse-smoke
```

Unsupervised SimCSE: each sentence is passed through the encoder twice with different dropout masks, creating two views as a positive pair. `MultipleNegativesRankingLoss` treats all other sentences in the batch as in-batch negatives. Artifact: `artifacts/simcse_gu/` (complete SentenceTransformer directory).

### Phase 4C — LLM LoRA (branch C)

```bash
make llm-mntp   # Masked Next-Token Prediction pre-training
make llm-lora   # LoRA adapter fine-tuning
```

Adapts a decoder LLM (LLaMA-style) for sentence embeddings. All base model weights are frozen; only LoRA adapters are trained. Embeddings are produced by mean-pooling over the last hidden states. Config: `configs/llm_embed.yaml`. Artifact: `artifacts/llm_lora/`.

### Phase 5 — Annotation service

See [Annotation service and active learning](#annotation-service-and-active-learning).

### Phase 6 — Evaluation

```bash
make eval
# or: python -m langembed.eval.evaluate --config configs/eval.yaml
```

Computes for each branch in `configs/eval.yaml`:
- **Spearman correlation** between predicted cosine similarities and human STS scores
- **Recall@k** — fraction of queries where the correct item is in the top-k results
- **MRR@k** — Mean Reciprocal Rank of the first correct result within top-k

Results written to `metrics/eval.json` and logged to MLflow. The leakage guard runs before any model loads.

### Phase 7 — Serving

See [Serving — /embed endpoint](#serving--embed-endpoint).

---

## Creating and using embeddings

### After running the pipeline

```python
from sentence_transformers import SentenceTransformer

# Load any trained model
model = SentenceTransformer("artifacts/smoke/simcse_en")  # smoke model (128-dim)
# model = SentenceTransformer("artifacts/embed_gu_v1")    # production branch A (512-dim)

sentences = [
    "The cat sat on the mat.",
    "A cat was resting on a rug.",
    "The weather is sunny today.",
]

# Encode with L2 normalisation
embeddings = model.encode(sentences, normalize_embeddings=True)
print(embeddings.shape)  # (3, 128) for smoke, (3, 512) for production

# Cosine similarity (L2-normalised → dot product = cosine)
import numpy as np
sims = embeddings @ embeddings.T
print(sims)
```

### Using the normalisation invariant

All text must go through `normalize` before encoding, mirroring the training pipeline:

```python
from langembed.preprocess import normalize
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("artifacts/embed_gu_v1")

texts = ["Hello world", "Привет мир"]
embeddings = model.encode([normalize(t) for t in texts], normalize_embeddings=True)
```

The `/embed` serving endpoint applies `normalize` automatically. For direct Python use, apply it yourself to ensure consistency with training.

### Batch encoding large datasets

```python
from sentence_transformers import SentenceTransformer
from langembed.preprocess import normalize

model = SentenceTransformer("artifacts/embed_gu_v1")

with open("data/corpus_gu.txt", encoding="utf-8") as f:
    sentences = [normalize(line.strip()) for line in f if line.strip()]

embeddings = model.encode(
    sentences,
    batch_size=256,
    normalize_embeddings=True,
    show_progress_bar=True,
)
print(f"Encoded {len(sentences)} sentences → shape {embeddings.shape}")
```

### Evaluating a model manually

```python
from langembed.config import load_config
from langembed.eval.evaluate import evaluate

cfg = load_config("configs/eval.yaml")
results = evaluate(cfg)
# {'spearman_A': 0.82, 'retrieval_recall@10_A': 0.75, 'retrieval_mrr@10_A': 0.61, ...}
```

---

## Serving — `/embed` endpoint

### Start the server

```bash
# Default: loads from artifacts/embed_gu_v1
make serve

# Custom model:
LANGEMBED_MODEL_DIR=artifacts/smoke/simcse_en make serve

# Production with multiple workers:
LANGEMBED_MODEL_DIR=artifacts/embed_gu_v1 \
  uvicorn langembed.serving.serve:app \
  --host 0.0.0.0 --port 8000 --workers 4
```

Server runs at `http://localhost:8000`. Interactive API docs at `http://localhost:8000/docs`.

### `POST /embed`

**Request:**

```json
{
  "texts": ["Hello world", "Another sentence"]
}
```

**Response:**

```json
{
  "embeddings": [[0.12, -0.34, 0.56, ...], [0.78, 0.23, -0.11, ...]],
  "dim": 512
}
```

**curl:**

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Test sentence"]}'
```

**Python:**

```python
import requests

response = requests.post(
    "http://localhost:8000/embed",
    json={"texts": ["Hello world", "Another sentence"]}
)
data = response.json()
embeddings = data["embeddings"]  # list[list[float]]
dim = data["dim"]                # int
```

**Notes:**
- Input texts are normalised via `preprocess.normalize` server-side before encoding.
- All returned vectors are L2-normalised (‖v‖₂ = 1).
- Model loads once at startup and is reused for all requests.
- Model directory is controlled by the `LANGEMBED_MODEL_DIR` environment variable.

---

## Annotation service and active learning

The annotation service runs on port 8001 and drives the active learning loop for Phase 5.

### Start the service

```bash
# Local (requires postgres + redis)
docker compose up -d postgres redis
make serve-annotation
# Interactive docs: http://localhost:8001/docs

# Docker
docker compose up -d annotation
```

### Active learning workflow

```
1. After SimCSE training, score all corpus sentence pairs:
       uncertainty(a, b) = 1 - |cos(embed(a), embed(b)) - 0.5| / 0.5
   Maximum uncertainty at cos=0.5 (model is undecided).
   Zero uncertainty at cos=0 or cos=1 (pair is obvious).

2. Load high-uncertainty pairs into the Item table.

3. Annotators fetch pairs:
       GET /queue?annotator_id=1&n=20
   → 20 uncertain pairs + 2 hidden gold calibration pairs

4. Each pair receives a score 0–5 (0=unrelated, 5=identical meaning):
       POST /annotate  {"item_id": 42, "annotator_id": 1, "label": 4.0}

5. Export aggregated labels as training triplets:
       POST /export  → writes data/native_triplets.jsonl

6. Train supervised contrastive model on triplets:
       make supervised

7. Re-score corpus with the new model → refill queue → repeat.
```

### Endpoints

#### `GET /queue?annotator_id={id}&n={n}`

```bash
curl "http://localhost:8001/queue?annotator_id=1&n=10"
```

```json
{
  "items": [
    {
      "id": 123,
      "sentence_a": "The cat sat on the mat.",
      "sentence_b": "A kitten rested on the rug.",
      "uncertainty": 0.98,
      "status": "pending"
    }
  ]
}
```

#### `POST /annotate`

```bash
curl -X POST http://localhost:8001/annotate \
  -H "Content-Type: application/json" \
  -d '{"item_id": 123, "annotator_id": 1, "label": 4.0}'
```

```json
{"ok": true}
```

#### `POST /export`

```bash
curl -X POST "http://localhost:8001/export?out_path=data/native_triplets.jsonl"
```

```json
{"written": 847, "path": "data/native_triplets.jsonl"}
```

Triplet format in `data/native_triplets.jsonl`:

```json
{"anchor": "sentence A", "positive": "similar sentence", "negative": "unrelated sentence"}
```

### Annotator quality

- Each annotator's reliability is tracked via **Cohen's weighted kappa**.
- Labels are aggregated using **reliability-weighted averaging** — annotators who systematically diverge from consensus receive lower weight over time.
- Gold calibration pairs (status `"gold"`) have known ground-truth labels; they are mixed into the queue transparently to detect unreliable annotators.

### Seeding gold questions

```bash
python scripts/seed_gold.py
```

Run once before starting an annotation campaign to populate the gold calibration table.

---

## Evaluation

### Run evaluation

```bash
make eval
# or: python -m langembed.eval.evaluate --config configs/eval.yaml

# Smoke model only:
python -m langembed.eval.evaluate --config configs/smoke/eval.yaml
```

### Metrics

For each branch listed in `configs/eval.yaml`:

| Metric | Meaning |
|--------|---------|
| `spearman_{branch}` | Spearman ρ between predicted cosines and human STS scores |
| `retrieval_recall@k_{branch}` | Fraction of queries where correct item is in top-k |
| `retrieval_mrr@k_{branch}` | Mean Reciprocal Rank of first correct result |

### Comparing branches

```bash
# Print current metrics
dvc metrics show

# Compare with previous commit
dvc metrics diff HEAD~1

# Visual comparison in MLflow
mlflow ui   # Experiments → select → check runs → Compare
```

### Leakage guard

Before any model loads, `evaluate.py` hashes all sentences in the test file, then scans every path listed in `train_paths`. If any training sentence appears in the test set, evaluation aborts:

```
RuntimeError: Test leakage detected via data/corpus_gu.txt
```

---

## MLflow experiment tracking

MLM pre-training logs loss and perplexity to MLflow at every eval step. The evaluation phase writes Spearman, Recall@k, and MRR@k to both `metrics/eval.json` and MLflow.

```bash
mlflow ui    # http://localhost:5000
```

To compare branches: **Experiments → select experiment → check all branch runs → Compare**.

The "Parallel Coordinates" chart in MLflow is especially useful for seeing which hyperparameter settings correlate with higher Spearman scores across many runs.

---

## Testing

### Unit tests (fast, no external deps)

```bash
make test
# or: pytest tests/
```

Covers: `normalize`, `dedup`, `uncertainty_from_cosine`, `weighted_kappa`, `aggregate`.

### API contract tests (FastAPI TestClient + SQLite in-memory)

Included in `make test`. No running Postgres or Redis required.

Covers: `/queue`, `/annotate`, `/export`, `/embed`.

### E2E smoke test (full CPU pipeline)

```bash
make test-e2e
# or: pytest -m e2e tests/e2e/ -v
```

Runs the complete pipeline on English fixture data: `build_corpus` → tokenizer → MLM (50 steps) → SimCSE (1 epoch) → evaluate. Takes 2–5 minutes on CPU.

### Linting and type checking

```bash
make lint
# runs:
#   ruff check src tests
#   ruff format --check src tests
#   mypy src
```

All three must pass. CI enforces `make lint && make test`.

---

## Docker and docker-compose

### Building images

```bash
# Annotation + serving (~400 MB, no torch)
docker build --target base -t langembed-annotation .

# Full ML image (~4 GB, includes torch)
docker build --target ml -t langembed-ml .
```

### Running services

```bash
# Infrastructure only
docker compose up -d postgres redis

# Full stack
docker compose up -d

# Run training inside the ML container
docker compose run --rm train make corpus
docker compose run --rm train make pretrain

# View logs
docker compose logs -f annotation
docker compose logs -f serve
```

### Environment variables (`.env`)

```bash
POSTGRES_USER=langembed
POSTGRES_PASSWORD=secret
POSTGRES_DB=langembed
DATABASE_URL=postgresql://langembed:secret@postgres:5432/langembed
REDIS_URL=redis://redis:6379/0
LANGEMBED_MODEL_DIR=artifacts/embed_gu_v1
```

---

## Adapting to another language

1. **`configs/tokenizer.yaml`** — set `language: <lang_code>` and update `raw_paths`.
2. **`configs/eval.yaml`** — update `test_path` to your STS test file.
3. **`preprocess.normalize`** — for non-Indic languages, IndicNLP is skipped automatically; only NFKC + whitespace collapse applies.
4. Run `dvc repro` — all stages re-run automatically when configs change.

No code changes needed for non-Indic languages.

**STS test file format** (JSONL, one pair per line):

```json
{"sentence_a": "First sentence", "sentence_b": "Second sentence", "score": 3.5}
```

`score` must be in `[0, score_scale]` where `score_scale` is defined in `configs/eval.yaml` (default: 5.0).

---

## Makefile reference

| Target | Description |
|--------|-------------|
| `make setup` | `pip install -e ".[ml,serve,dev]"` |
| `make lint` | `ruff check` + `ruff format --check` + `mypy` |
| `make test` | Unit + API contract tests |
| `make test-e2e` | Full English pipeline E2E smoke test |
| `make corpus` | Phase 1: build cleaned corpus |
| `make tokenizer` | Phase 2: train BPE tokenizer |
| `make pretrain` | Phase 3: MLM pre-training (full, 200k steps) |
| `make pretrain-smoke` | Phase 3: MLM pre-training (50 steps, CPU) |
| `make simcse` | Phase 4: SimCSE contrastive fine-tune |
| `make simcse-smoke` | Phase 4: SimCSE (1 epoch, 256 sentences, CPU) |
| `make supervised` | Phase 4: supervised triplet fine-tune |
| `make llm-mntp` | Phase 4C: LLM Masked Next-Token Prediction |
| `make llm-lora` | Phase 4C: LLM LoRA fine-tuning |
| `make serve-annotation` | Phase 5: annotation service on :8001 |
| `make eval` | Phase 6: evaluate all branches |
| `make serve` | Phase 7: `/embed` service on :8000 |
| `make smoke-dvc` | Run complete smoke DVC pipeline |

---

## Troubleshooting

### Stage re-runs unexpectedly after `dvc repro`

DVC re-runs a stage when any dependency hash changes. Run `dvc status` to see which deps changed. If you modified a source file tracked as a dep, all downstream stages will re-run.

### `RuntimeError: Test leakage detected`

A training corpus sentence appeared in the STS test set. Verify `test_path` in `configs/eval.yaml` points to the correct file. For the smoke pipeline, this is expected because fixture data overlaps — the smoke config uses `data/smoke/sts_test_placeholder.jsonl` (absent file) with `train_paths: []` to skip the guard intentionally.

### Out-of-memory during pre-training

- Reduce `per_device_train_batch_size` in `configs/pretrain.yaml`.
- Increase `gradient_accumulation_steps` to maintain effective batch size.
- Reduce `hidden_size` for a smaller model.
- Use `fp16: true` on GPU to halve memory usage.

### MLflow not logging

- Confirm the MLflow server is running: `mlflow ui`.
- Check `report_to: [mlflow]` in `configs/pretrain.yaml`.
- Set `MLFLOW_TRACKING_URI` if using a remote tracking server.

### `make test-e2e` is slow

Expected duration: 2–5 minutes on CPU. If it exceeds 15 minutes, reduce `smoke.max_steps` in `configs/pretrain.yaml` (default: 50).

### Windows + Python 3.14: `sentence_transformers` subprocess crash

Symptom: subprocess exits with code `3221225477` (access violation). Cause: `pyarrow` C extension DLL initialisation order conflict. Fix already applied in `train_simcse.py` and `evaluate.py`: `datasets`, `pandas`, `pyarrow`, and `torch` are pre-imported before `sentence_transformers` inside the function body. Apply the same pattern to any new code that uses `sentence_transformers` in a subprocess.
