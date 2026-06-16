# langembed

> **Русская документация:** [docs/ru/README_RU.md](docs/ru/README_RU.md)

A from-scratch sentence-embedding pipeline for low-resource languages, with a native-speaker active learning loop. The pipeline trains three parallel model branches on the same data and evaluates them with the same metric — enabling a clean A/B/C comparison.

---

## What this is

The research goal is to train high-quality sentence embeddings for a language with limited resources (Gujarati is used as the example), then measure how much each architectural choice contributes to quality.

**Three branches trained on identical native-speaker data:**

| Branch | Approach | Key components |
|--------|----------|----------------|
| **A** | From scratch | Custom BPE tokenizer → RoBERTa-style encoder (MLM pre-train) → SimCSE contrastive fine-tune |
| **B** | Multilingual transfer | mBERT / XLM-R fine-tuned on the same native-speaker supervision |
| **C** | LLM as embedder | Decoder LLM (LLaMA-style) + LoRA adapters, mean-pool over last hidden states (llm2vec approach) |

**Native speakers are central:** they supply annotation signal and quality control through an active learning loop. The system surfaces the most *uncertain* sentence pairs to annotators, maximising the information value of each labelled example.

**Evaluation metric:** Spearman correlation on an isolated STS (Semantic Textual Similarity) test set. The test set never enters any training stage — this is enforced as an architectural invariant.

---

## Architecture

The pipeline runs as a sequence of eight phases (0–7) plus branch-C phase 4C:

| Phase | Name | Description |
|-------|------|-------------|
| 0 | Normalisation | NFKC + IndicNLP (Gujarati) + whitespace collapse via `preprocess.normalize` |
| 1 | Corpus | Aggregate `data/raw/*.txt`, MinHash-dedup, write `data/corpus.txt` |
| 2 | Tokenizer | BPE with Whitespace pre-tokenizer, NFKC normaliser, vocab\_size=8000 |
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

**Data flow:** corpus → tokenizer → encoder → contrastive fine-tune → annotation loop → evaluation → serving

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

configs/                 YAML hyperparameters for every phase
tests/                   Unit, API contract, and E2E tests
  conftest.py            Shared fixtures (SQLite in-memory, TestClient)
  e2e/                   Full-pipeline English CPU smoke test
  fixtures/              English corpus + STS pairs for E2E test
docs/
  ru/README_RU.md        Russian documentation (this file's counterpart)
  IMPLEMENTATION_PLAN.md Phase-by-phase implementation reference
Makefile                 make lint | test | test-e2e | corpus | pretrain | …
Dockerfile               Multi-stage: base (serve extras) / ml (+ torch)
docker-compose.yml       postgres, redis, annotation :8001, serve :8000, train
dvc.yaml                 Reproducibility DAG
```

---

## Installation

**Local:**

```bash
git clone https://github.com/DenisBaliuckij/langembed
cd langembed
pip install -e ".[ml,serve,dev]"
cp .env.example .env        # fill in passwords
docker compose up -d postgres redis
```

**Docker (full stack):**

```bash
cp .env.example .env
docker compose up -d postgres redis
docker build --target base -t langembed-annotation .   # ~400 MB, no torch
docker build --target ml   -t langembed-ml .           # ~4 GB, includes torch
docker compose up -d
```

---

## Running the pipeline

```bash
# 1. Place raw text data
cp your_data.txt data/raw/wiki_gu.txt

# 2. Start infrastructure
docker compose up -d postgres redis

# 3. Run full DVC pipeline
dvc repro

# 4. Evaluate all branches
make eval

# 5. Inspect results
mlflow ui    # http://localhost:5000
```

Individual-phase smoke tests (fast CPU check without real data):

```bash
make pretrain-smoke   # MLM pre-train: 50 steps, tiny model
make simcse-smoke     # SimCSE: 1 epoch on a small sample
make test-e2e         # Full English pipeline smoke test (few minutes)
```

---

## Pipeline phases in detail

**Phase 0 — Normalisation.**
Applies NFKC Unicode normalisation, IndicNLP script normalisation (Gujarati by default), and whitespace collapse. `preprocess.normalize` is called automatically in every downstream phase. For non-Indic languages the IndicNLP step is skipped gracefully.

**Phase 1 — Corpus.**
Reads all files matching `data/raw/*.txt`, deduplicates sentences with MinHash LSH (configurable Jaccard threshold), and writes the clean corpus to `data/corpus.txt`. Config: `configs/corpus.yaml` (`raw_paths`, `out_path`, `threshold`, `num_perm`).

**Phase 2 — Tokenizer.**
Trains a BPE tokenizer with Whitespace pre-tokenizer and NFKC normaliser. Default `vocab_size=8000`. Artifact saved to `artifacts/tokenizer/`. Config: `configs/tokenizer.yaml`.

**Phase 3 — MLM pre-train.**
Trains a RoBERTa-style encoder (configurable `hidden_size`, `num_hidden_layers`, `num_attention_heads`, etc.) on masked language modelling using HuggingFace Trainer. Loss and perplexity logged to MLflow. Artifact: `artifacts/encoder/`. Config: `configs/pretrain.yaml`.

**Phase 4 — SimCSE.**
Contrastive fine-tuning using the encoder from Phase 3. Positive pairs are two forward passes of the same sentence with different dropout masks. Implemented via SentenceTransformers `MultipleNegativesRankingLoss`. Artifact: `artifacts/simcse/`. Config: `configs/simcse.yaml`.

**Phase 4C — LLM LoRA.**
LoRA-adapts a decoder LLM (LLaMA-style architecture) for sentence embeddings via mean-pooling over the last hidden layer. All base-model weights are frozen; only LoRA adapters are trained. Preceded by Masked Next-Token Prediction (MNTP) pre-training (`llm_embed/mntp.py`). Artifact: `artifacts/llm_lora/`. Config: `configs/llm_lora.yaml`.

**Phase 5 — Annotation service.**
FastAPI application (port 8001) for collecting sentence-pair labels from native speakers. PostgreSQL stores items and labels; Redis backs the Celery task queue. Active learning selects the most uncertain pairs: uncertainty = `1 - |cos - 0.5| / 0.5` (maximum at cos=0.5, zero at cos=0 or cos=1).

**Phase 6 — Evaluation.**
Computes Spearman correlation between predicted cosine similarities and human STS scores, plus Recall@k and MRR@k for retrieval. Results written to `metrics/eval.json` and logged to MLflow. Config: `configs/eval.yaml`.

**Phase 7 — Serving.**
FastAPI application (port 8000) with a single `/embed` endpoint. Accepts a JSON list of strings, returns L2-normalised embedding vectors. Model directory read from the `LANGEMBED_MODEL_DIR` environment variable (default: `artifacts/embed_gu_v1`).

---

## Annotation service and active learning

The annotation service (`annotation/api.py`) exposes three endpoints:

- **`GET /queue?annotator_id=N&n=10`** — returns the N most uncertain sentence pairs plus 2 gold calibration questions with known answers. Gold questions detect unreliable annotators.
- **`POST /annotate`** — saves an annotator's score (0–5) for a sentence pair.
- **`POST /export`** — aggregates labels from multiple annotators and writes training triplets (anchor, positive, negative).

**Uncertainty formula:** `1 - |cos - 0.5| / 0.5`

A pair is maximally uncertain when the model gives it a cosine of 0.5 — the annotator's label is most informative. Pairs near 0 or 1 are obvious and skipped.

**Annotator quality** is tracked via Cohen's weighted kappa. Labels are aggregated with weights proportional to each annotator's reliability score. Annotators who systematically disagree with consensus receive lower weight over time.

Label Studio configs for setting up annotation projects live in `annotation/label_studio/`. Use `scripts/seed_gold.py` to seed the gold calibration table before starting an annotation campaign.

---

## Testing

Three test layers:

**Unit tests** (pure functions, no external dependencies):
`normalize`, `dedup`, `uncertainty_from_cosine`, `weighted_kappa`, `aggregate`

**API contract tests** (FastAPI TestClient + SQLite in-memory via StaticPool):
Annotation service endpoints (`/queue`, `/annotate`, `/export`) and `/embed` — no running Postgres or Redis required.

**E2E smoke test** (full pipeline on English fixture data, CPU only):
`build_corpus` → tokenizer → MLM (50 steps, hidden=128) → SimCSE (1 epoch) → evaluate.
After 50 MLM steps, Spearman will be low — the test proves the code path works end-to-end, not that the model is good.

```bash
make lint          # ruff check + ruff format + mypy
make test          # unit + API contract tests
make test-e2e      # full English pipeline smoke (a few minutes on CPU)
```

---

## DVC and reproducibility

`dvc repro` runs the DAG defined in `dvc.yaml`. Stages execute in dependency order:

```
corpus → tokenizer → pretrain → simcse → supervised → llm_lora → evaluate
```

```bash
dvc repro            # run all stale stages
dvc metrics show     # print metrics/eval.json
dvc dag              # visualise the DAG
```

Model weights and datasets are tracked by DVC and excluded from git (`.gitignore`, `.dvcignore`). Configure remote storage for team use:

```bash
dvc remote add -d myremote s3://my-bucket/langembed
dvc push
```

Any team member can then reproduce the full experiment with `dvc pull && dvc repro`.

---

## MLflow

MLM pre-training logs loss and perplexity to MLflow at every eval step. The evaluation phase writes Spearman, Recall@k, and MRR@k to both `metrics/eval.json` and MLflow.

```bash
mlflow ui    # open http://localhost:5000
```

To compare branches: **Experiments → select experiment → check runs → Compare**. Compare branch A (from scratch), B (multilingual fine-tuned), and C (LLM LoRA) side by side to determine which architecture achieves the best Spearman at your data scale.

---

## Adapting to another language

To switch from Gujarati to any other language:

1. **`configs/tokenizer.yaml`** — set `language: <lang_code>` and update `raw_paths` to your corpus.
2. **`configs/eval.yaml`** — update `test_path` to your STS test file.
3. **`preprocess.normalize`** — the `lang` parameter in `_indic_normalizer` is config-driven (defaults to `"gu"`). For non-Indic languages the IndicNLP normaliser is skipped automatically; only NFC + whitespace collapse applies.
4. Run `dvc repro` — all stages are parameterised and will re-run automatically.

No code changes are needed for non-Indic languages.
