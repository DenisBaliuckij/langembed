# DVC Smoke Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-contained `dvc-smoke.yaml` pipeline backed by `configs/smoke/` that runs the full corpus→tokenizer→pretrain→simcse→evaluate DAG on the English fixture corpus, on CPU, in ~30 seconds.

**Architecture:** Four smoke YAML configs (one per pipeline stage group) point at `tests/fixtures/en_corpus.txt` as input and write outputs under `data/smoke/` and `artifacts/smoke/`. A standalone `dvc-smoke.yaml` defines the five DVC stages using those configs. Production `dvc.yaml` is untouched.

**Tech Stack:** DVC 3.x, Python 3.11, existing `langembed` modules (no new Python code).

## Global Constraints

- Production `dvc.yaml` must not be modified.
- All smoke outputs land under `data/smoke/` or `artifacts/smoke/` (both already gitignored).
- `metrics/smoke_eval.json` uses DVC `cache: false` — it is committed to git, not cached in `.dvc/cache/`.
- `fp16: false` in pretrain config — CPU-only smoke run.
- `--smoke` flag passed to `pretrain` and `simcse` stages — caps MLM at 50 steps, SimCSE at 20 sentences.
- No DVC remote is configured or required — local cache only.

---

### Task 1: Create smoke config files

**Files:**
- Create: `configs/smoke/tokenizer.yaml`
- Create: `configs/smoke/pretrain.yaml`
- Create: `configs/smoke/contrastive.yaml`
- Create: `configs/smoke/eval.yaml`

**Interfaces:**
- Consumes: `tests/fixtures/en_corpus.txt` (160 lines), `tests/fixtures/en_sts_test.jsonl` (20 pairs)
- Produces: Config paths consumed by Task 2's `dvc-smoke.yaml` stage `cmd:` lines

- [ ] **Step 1: Create `configs/smoke/tokenizer.yaml`**

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

- [ ] **Step 2: Create `configs/smoke/pretrain.yaml`**

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

- [ ] **Step 3: Create `configs/smoke/contrastive.yaml`**

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

- [ ] **Step 4: Create `configs/smoke/eval.yaml`**

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

- [ ] **Step 5: Verify all four configs parse**

```bash
python -c "
from langembed.config import load_config
for name in ['tokenizer', 'pretrain', 'contrastive', 'eval']:
    cfg = load_config(f'configs/smoke/{name}.yaml')
    print(name, 'OK', list(cfg.keys()))
"
```

Expected output (keys will differ per file, no exceptions):
```
tokenizer OK ['language', 'data', 'tokenizer']
pretrain OK ['seed', 'tokenizer_dir', 'corpus_path', 'report_to', 'out_dir', 'model', 'training', 'smoke']
contrastive OK ['seed', 'encoder_dir', 'simcse']
eval OK ['test_path', 'score_scale', 'retrieval_k', 'branches', 'train_paths', 'metrics_path']
```

- [ ] **Step 6: Commit configs**

```bash
git add configs/smoke/
git commit -m "feat(smoke): add smoke configs for DVC pipeline (English fixture, CPU)"
```

---

### Task 2: Create `dvc-smoke.yaml` and update Makefile

**Files:**
- Create: `dvc-smoke.yaml`
- Modify: `Makefile` — add `smoke-dvc` to `.PHONY` and add target

**Interfaces:**
- Consumes: `configs/smoke/*.yaml` (Task 1), all `src/langembed/` source modules
- Produces: `dvc-smoke.yaml` (consumed by Task 3's `dvc repro` command)

- [ ] **Step 1: Create `dvc-smoke.yaml`**

```yaml
# Smoke pipeline: English fixture data, CPU-only, full DAG in ~30 s.
# Run:  dvc repro --file dvc-smoke.yaml
# Or:   make smoke-dvc
stages:
  corpus:
    cmd: python -m langembed.data.build_corpus --config configs/smoke/tokenizer.yaml
    deps:
      - configs/smoke/tokenizer.yaml
      - tests/fixtures/en_corpus.txt
      - src/langembed/preprocess.py
      - src/langembed/data/dedup.py
      - src/langembed/data/build_corpus.py
    outs:
      - data/smoke/corpus_en.txt

  tokenizer:
    cmd: python -m langembed.tokenizer.train_tokenizer --config configs/smoke/tokenizer.yaml
    deps:
      - configs/smoke/tokenizer.yaml
      - data/smoke/corpus_en.txt
      - src/langembed/tokenizer/train_tokenizer.py
    outs:
      - artifacts/smoke/tokenizer_en

  pretrain:
    cmd: python -m langembed.pretrain.train_mlm --config configs/smoke/pretrain.yaml --smoke
    deps:
      - configs/smoke/pretrain.yaml
      - data/smoke/corpus_en.txt
      - artifacts/smoke/tokenizer_en
      - src/langembed/pretrain/train_mlm.py
    outs:
      - artifacts/smoke/encoder_en

  simcse:
    cmd: python -m langembed.contrastive.train_simcse --config configs/smoke/contrastive.yaml --smoke
    deps:
      - configs/smoke/contrastive.yaml
      - data/smoke/corpus_en.txt
      - artifacts/smoke/encoder_en
      - src/langembed/contrastive/train_simcse.py
    outs:
      - artifacts/smoke/simcse_en

  evaluate:
    cmd: python -m langembed.eval.evaluate --config configs/smoke/eval.yaml
    deps:
      - configs/smoke/eval.yaml
      - tests/fixtures/en_sts_test.jsonl
      - artifacts/smoke/simcse_en
      - src/langembed/eval/evaluate.py
    metrics:
      - metrics/smoke_eval.json:
          cache: false
```

- [ ] **Step 2: Add `smoke-dvc` to Makefile**

In `Makefile`, change the first line from:
```makefile
.PHONY: setup lint test test-e2e corpus tokenizer pretrain pretrain-smoke simcse simcse-smoke supervised serve-annotation eval serve llm-mntp llm-lora
```
to:
```makefile
.PHONY: setup lint test test-e2e corpus tokenizer pretrain pretrain-smoke simcse simcse-smoke supervised serve-annotation eval serve llm-mntp llm-lora smoke-dvc
```

Then add the target before the catch-all `%:` rule at the bottom:
```makefile
smoke-dvc:
	dvc repro --file dvc-smoke.yaml
```

- [ ] **Step 3: Verify DVC recognises the pipeline**

```bash
python -m dvc dag --file dvc-smoke.yaml
```

Expected output — five-node linear DAG:
```
        +--------+
        | corpus |
    ****+--------+****
****         *        ****
...
+----------+
| evaluate |
+----------+
```

- [ ] **Step 4: Verify DVC status shows all stages as new/changed**

```bash
python -m dvc status --file dvc-smoke.yaml
```

Expected: every stage listed under `changed deps` or `new` (none should say "up to date" because nothing has run yet).

- [ ] **Step 5: Commit**

```bash
git add dvc-smoke.yaml Makefile
git commit -m "feat(smoke): add dvc-smoke.yaml pipeline and smoke-dvc Makefile target"
```

---

### Task 3: Run the pipeline and verify correctness and idempotency

**Files:**
- Generated: `data/smoke/corpus_en.txt`, `artifacts/smoke/tokenizer_en/`, `artifacts/smoke/encoder_en/`, `artifacts/smoke/simcse_en/`
- Generated: `metrics/smoke_eval.json`, `dvc-smoke.lock`

**Interfaces:**
- Consumes: `dvc-smoke.yaml` (Task 2), `configs/smoke/*.yaml` (Task 1)
- Produces: Committed `dvc-smoke.lock` + `metrics/smoke_eval.json`

- [ ] **Step 1: Run the full pipeline**

```bash
make smoke-dvc
```

Expected: five stages execute in order with no errors. Final lines should include:
```
branch en_smoke: Spearman = ...
branch en_smoke: Recall@5=..., MRR@5=...
metrics written to metrics/smoke_eval.json
```

- [ ] **Step 2: Verify every output exists**

```bash
python -c "
from pathlib import Path
outputs = [
    'data/smoke/corpus_en.txt',
    'artifacts/smoke/tokenizer_en/tokenizer.json',
    'artifacts/smoke/encoder_en/config.json',
    'artifacts/smoke/simcse_en',
    'metrics/smoke_eval.json',
    'dvc-smoke.lock',
]
for p in outputs:
    path = Path(p)
    exists = path.exists() or (path.is_dir() and any(path.iterdir()))
    status = 'OK' if exists else 'MISSING'
    print(status + ': ' + p)
"
```

Expected: all lines print `OK`.

- [ ] **Step 3: Verify metrics content**

```bash
python -m dvc metrics show metrics/smoke_eval.json
```

Expected: JSON with keys `spearman_en_smoke`, `retrieval_recall@5_en_smoke`, `retrieval_mrr@5_en_smoke`. Values will be low (50-step smoke model) but must be floats, not `null`.

- [ ] **Step 4: Verify idempotency — second run is a no-op**

```bash
python -m dvc repro --file dvc-smoke.yaml
```

Expected output:
```
Stage 'corpus' didn't change, skipping
Stage 'tokenizer' didn't change, skipping
Stage 'pretrain' didn't change, skipping
Stage 'simcse' didn't change, skipping
Stage 'evaluate' didn't change, skipping
```

All five stages must say "didn't change" (exact wording may vary by DVC version, but no stage should re-execute).

- [ ] **Step 5: Commit lock file and metrics**

```bash
git add dvc-smoke.lock metrics/smoke_eval.json
git commit -m "feat(smoke): run dvc-smoke pipeline — lock file + initial metrics"
```
