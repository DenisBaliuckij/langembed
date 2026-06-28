# DVC Smoke Pipeline Design

**Date:** 2026-06-28  
**Status:** Approved

## Problem

`dvc.yaml` defines a 7-stage Gujarati training DAG but has never been executed — no `dvc.lock` exists and no DVC remote is configured. The production configs target GPU-scale training (512-dim, 200k steps, fp16) on Gujarati data. The only available corpus (`data/raw/wiki_gu.txt`) is 10 lines — too small even for the tokenizer's `min_frequency: 2`. A smoke path is needed to verify the DVC pipeline mechanics work end-to-end without GPU or large data.

## Solution

Add a self-contained smoke DVC pipeline (`dvc-smoke.yaml`) backed by dedicated smoke configs (`configs/smoke/`). It reuses the English fixture already proven by the e2e pytest suite, runs entirely on CPU in ~30 seconds, and is completely isolated from the production `dvc.yaml`.

## Files

### `configs/smoke/tokenizer.yaml`

```yaml
language: en
data:
  raw_paths:
    - tests/fixtures/en_corpus.txt
  out_path: data/smoke/corpus_en.txt
  test_path: tests/fixtures/en_sts_test.jsonl
tokenizer:
  vocab_size: 500
  min_frequency: 1
  unk_rate_max: 0.05
  out_dir: artifacts/smoke/tokenizer_en
```

### `configs/smoke/pretrain.yaml`

```yaml
seed: 42
tokenizer_dir: artifacts/smoke/tokenizer_en
corpus_path: data/smoke/corpus_en.txt
report_to: []
out_dir: artifacts/smoke/encoder_en
model:
  hidden_size: 128
  num_hidden_layers: 2
  num_attention_heads: 4
  intermediate_size: 256
  max_position_embeddings: 514
  max_seq_length: 64
training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 1
  learning_rate: 5.0e-4
  weight_decay: 0.01
  warmup_steps: 5
  max_steps: 200
  fp16: false
  save_steps: 100
  logging_steps: 10
  mlm_probability: 0.15
smoke:
  max_steps: 50
```

### `configs/smoke/contrastive.yaml`

```yaml
seed: 42
encoder_dir: artifacts/smoke/encoder_en
simcse:
  sentences_path: data/smoke/corpus_en.txt
  out_dir: artifacts/smoke/simcse_en
  batch_size: 8
  epochs: 1
  warmup_steps: 2
  max_seq_length: 64
```

### `configs/smoke/eval.yaml`

```yaml
test_path: tests/fixtures/en_sts_test.jsonl
score_scale: 5.0
retrieval_k: 5
branches:
  en_smoke: artifacts/smoke/simcse_en
train_paths:
  - data/smoke/corpus_en.txt
metrics_path: metrics/smoke_eval.json
```

### `dvc-smoke.yaml`

Five stages: `corpus → tokenizer → pretrain → simcse → evaluate`.

- `supervised` and `llm_lora` are excluded: both require `native_triplets.jsonl` and `llm_lora` needs a multi-GB LLM download.
- `pretrain` and `simcse` pass `--smoke` to use the 50-step fast-path.
- Each stage lists the relevant `src/langembed/` source files as deps so DVC detects code changes.
- Outputs land under `data/smoke/` and `artifacts/smoke/` (already gitignored by existing `data/` and `artifacts/` rules).
- `metrics/smoke_eval.json` uses `cache: false` — tracked in `dvc-smoke.lock`, not cached in DVC store.

### Makefile

Add one target:

```makefile
smoke-dvc:
	dvc repro --file dvc-smoke.yaml
```

## Stage DAG

```
corpus → tokenizer → pretrain → simcse → evaluate
```

Inputs (all committed to git, no download needed):
- `tests/fixtures/en_corpus.txt` — 160-line English corpus
- `tests/fixtures/en_sts_test.jsonl` — 20 STS pairs

Outputs (DVC-tracked, gitignored):
- `data/smoke/corpus_en.txt`
- `artifacts/smoke/tokenizer_en/`
- `artifacts/smoke/encoder_en/`
- `artifacts/smoke/simcse_en/`

Metrics (committed, DVC `cache: false`):
- `metrics/smoke_eval.json` — Spearman + retrieval@5 for `en_smoke` branch

## Verification

```bash
make smoke-dvc                              # full repro from scratch (~30s on CPU)
dvc repro --file dvc-smoke.yaml            # idempotent: second run is a no-op
dvc metrics show metrics/smoke_eval.json   # prints Spearman + retrieval@5
```

## Constraints

- Production `dvc.yaml` is untouched.
- All smoke outputs are isolated under `data/smoke/` and `artifacts/smoke/`.
- No remote DVC storage is required (local cache only).
- The eval leak-guard runs against the fixture corpus — it passes because `en_sts_test.jsonl` pairs are distinct from `en_corpus.txt` sentences.
