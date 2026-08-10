# GPT-Style LLM Training Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `--train-llm` stage to `run_pipeline.py` that trains a GPT-2-style causal decoder, warm-started from the already-trained encoder's token embedding table, on the same corpus/tokenizer built earlier in the pipeline.

**Architecture:** A new module `src/langembed/llm/train_gpt.py` builds a `GPT2Config`/`GPT2LMHeadModel`, copies the encoder's token embedding weights into it (`warm_start_embeddings`), then trains via HF `Trainer` for causal LM on `datasets.load_dataset("text", ...)` (Arrow-backed, matching `train_mlm.py`'s proven-safe-at-scale pattern). `run_pipeline.py` gains `--train-llm`/`--llm-minutes` flags and a new stage after the existing step 6/6, using the same smoke-run-then-calibrate pattern as pretrain.

**Tech Stack:** `transformers.GPT2Config`/`GPT2LMHeadModel`, HF `Trainer`, `datasets.load_dataset` — same library set already used by `train_mlm.py`, no new dependencies.

## Global Constraints

- `--train-llm` defaults to off — every existing `run_pipeline.py` invocation without it must behave byte-for-byte as today.
- The new stage's `model.hidden_size` is `256`, hardcoded to match `pretrain_cfg["model"]["hidden_size"]` (also `256`, also hardcoded) — required for `warm_start_embeddings`'s direct weight copy to succeed.
- Heavy/optional imports (`torch`, `transformers`, `datasets`) stay function-local, matching every other ML-dependent module in this codebase (`CLAUDE.md`: "ML-импорты... делаются внутри функций").
- `warm_start_embeddings` raises `ValueError` (naming both shapes) on a hidden_size/vocab_size mismatch rather than silently skipping the copy.
- No new corpus-building or tokenizer-training work — this stage reuses `data/corpus_<lang>.txt` and `artifacts/tokenizer_<lang>` exactly as already built by stages 1-2.
- `ruff format`, `ruff check`, and `mypy` must stay clean; line length 100.

---

### Task 1: `train_gpt.py` — core module (warm-start + training)

**Files:**
- Create: `src/langembed/llm/__init__.py` (empty, matches every other package's `__init__.py` in this codebase, e.g. `src/langembed/contrastive/__init__.py`)
- Create: `src/langembed/llm/train_gpt.py`
- Test: `tests/test_train_gpt.py`
- Modify: `tests/e2e/test_pipeline_english.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `warm_start_embeddings(gpt_model, encoder_model) -> None` and
  `train_gpt(cfg: dict[str, Any], smoke: bool = False) -> None`, used by Task 2's new
  `run_pipeline.py` stage via `python -m langembed.llm.train_gpt --config <path> [--smoke]`
  (same CLI shape as `langembed.pretrain.train_mlm`).

- [ ] **Step 1: Create the empty package `__init__.py`**

Create `src/langembed/llm/__init__.py` with no content (0 bytes), matching
`src/langembed/contrastive/__init__.py`.

- [ ] **Step 2: Write the failing unit test for `warm_start_embeddings`**

Create `tests/test_train_gpt.py`:

```python
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch  # noqa: E402

from langembed.llm.train_gpt import warm_start_embeddings  # noqa: E402


def _make_encoder(vocab_size: int, hidden_size: int):
    from transformers import RobertaConfig, RobertaForMaskedLM

    config = RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=hidden_size * 2,
        max_position_embeddings=32,
        type_vocab_size=1,
    )
    return RobertaForMaskedLM(config)


def _make_gpt(vocab_size: int, hidden_size: int):
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=32,
        n_embd=hidden_size,
        n_layer=1,
        n_head=1,
    )
    return GPT2LMHeadModel(config)


def test_warm_start_embeddings_copies_weights():
    encoder = _make_encoder(vocab_size=50, hidden_size=16)
    gpt = _make_gpt(vocab_size=50, hidden_size=16)

    warm_start_embeddings(gpt, encoder)

    encoder_weights = encoder.roberta.embeddings.word_embeddings.weight
    gpt_weights = gpt.transformer.wte.weight
    assert torch.equal(gpt_weights, encoder_weights)


def test_warm_start_embeddings_raises_on_shape_mismatch():
    encoder = _make_encoder(vocab_size=50, hidden_size=16)
    gpt = _make_gpt(vocab_size=50, hidden_size=32)

    with pytest.raises(ValueError, match="shape"):
        warm_start_embeddings(gpt, encoder)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_train_gpt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'langembed.llm.train_gpt'`

- [ ] **Step 4: Write `train_gpt.py`**

Create `src/langembed/llm/train_gpt.py`:

```python
"""Optional pipeline stage: GPT-style causal LM warm-started from the encoder's
embedding table (see docs/superpowers/specs/2026-08-10-gpt-llm-training-design.md)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from langembed.config import load_config


def warm_start_embeddings(gpt_model: Any, encoder_model: Any) -> None:
    """Copy the encoder's token embedding table into the GPT model's embedding table.
    Both models must share the same tokenizer (same vocab_size) and hidden_size, since
    this is a direct weight copy, not a projection -- raises ValueError naming both
    shapes if they don't match, rather than silently skipping (a model that looks
    warm-started but isn't is worse than a loud failure).
    """
    import torch

    encoder_embeddings = encoder_model.roberta.embeddings.word_embeddings.weight
    gpt_embeddings = gpt_model.transformer.wte.weight
    if tuple(encoder_embeddings.shape) != tuple(gpt_embeddings.shape):
        raise ValueError(
            f"encoder embedding shape {tuple(encoder_embeddings.shape)} != "
            f"GPT embedding shape {tuple(gpt_embeddings.shape)} -- both models must "
            "share the same tokenizer/vocab_size and hidden_size for a direct copy"
        )
    with torch.no_grad():
        gpt_model.transformer.wte.weight.copy_(encoder_embeddings)


def train_gpt(cfg: dict[str, Any], smoke: bool = False) -> None:
    import torch
    from datasets import load_dataset
    from transformers import (
        DataCollatorForLanguageModeling,
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
        RobertaForMaskedLM,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(cfg.get("seed", 42))
    tok = PreTrainedTokenizerFast.from_pretrained(cfg["tokenizer_dir"])
    m = cfg["model"]
    config = GPT2Config(
        vocab_size=tok.vocab_size,
        n_positions=m["max_position_embeddings"],
        n_embd=m["hidden_size"],
        n_layer=m["num_hidden_layers"],
        n_head=m["num_attention_heads"],
        n_inner=m["intermediate_size"],
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    model = GPT2LMHeadModel(config)  # random init except for the embedding warm-start below

    encoder = RobertaForMaskedLM.from_pretrained(cfg["encoder_dir"])
    warm_start_embeddings(model, encoder)
    print("parameters:", model.num_parameters())

    ds = load_dataset("text", data_files={"train": cfg["corpus_path"]})["train"]
    ds = ds.map(
        lambda b: tok(b["text"], truncation=True, max_length=m["max_seq_length"]),
        batched=True,
        remove_columns=["text"],
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    t = cfg["training"]
    max_steps = cfg["smoke"]["max_steps"] if smoke else t["max_steps"]
    args = TrainingArguments(
        output_dir=str(Path(cfg["out_dir"]) / "ckpt"),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        weight_decay=t["weight_decay"],
        warmup_steps=t["warmup_steps"],
        max_steps=max_steps,
        fp16=t["fp16"] and torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
        save_steps=t["save_steps"],
        logging_steps=t["logging_steps"],
        report_to=cfg.get("report_to", ["mlflow"]),
    )
    Trainer(model=model, args=args, train_dataset=ds, data_collator=collator).train()
    model.save_pretrained(cfg["out_dir"])
    tok.save_pretrained(cfg["out_dir"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    train_gpt(load_config(args.config), smoke=args.smoke)


if __name__ == "__main__":
    main()
```

Note: the project's trained tokenizer (`src/langembed/tokenizer/train_tokenizer.py`) always sets
`bos_token="<s>"`, `eos_token="</s>"`, `pad_token="<pad>"` explicitly, so `tok.bos_token_id`/
`tok.eos_token_id` are always populated and `DataCollatorForLanguageModeling` always has a
`pad_token` to batch-pad with — no fallback/defensive code needed for this project's tokenizers.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_train_gpt.py -v`
Expected: PASS (both tests)

- [ ] **Step 6: Extend the e2e fixture with a GPT smoke stage**

In `tests/e2e/test_pipeline_english.py`, inside the `pipeline` fixture, immediately after the
existing SimCSE step (after the line `assert simcse_dir.exists(), "simcse output directory not
created"` and before the `return {...}` statement), insert:

```python
    # ── Step 5: GPT-style LLM (smoke: warm-started from encoder) ─────────
    from langembed.llm.train_gpt import train_gpt

    gpt_dir = base / "gpt"
    gpt_cfg: dict = {
        "seed": 42,
        "encoder_dir": str(encoder_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "corpus_path": str(corpus_file),
        "report_to": [],
        "model": {
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 256,
            "max_position_embeddings": 64,
            "max_seq_length": 64,
        },
        "training": {
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-4,
            "weight_decay": 0.01,
            "warmup_steps": 5,
            "max_steps": 200,
            "fp16": False,
            "save_steps": 100,
            "logging_steps": 10,
        },
        "smoke": {"max_steps": 50},
        "out_dir": str(gpt_dir),
    }
    train_gpt(gpt_cfg, smoke=True)
    assert (gpt_dir / "config.json").exists(), "gpt config.json missing"
```

`model.hidden_size` here (`128`) matches the fixture's already-built `encoder_dir`'s
`hidden_size` (`128`, set earlier in this same fixture's `mlm_cfg`) — required for
`warm_start_embeddings` to succeed in this test, same constraint as the real pipeline's `256`.

Then change the fixture's `return {...}` statement from:

```python
    return {
        "corpus_file": corpus_file,
        "tokenizer_dir": tokenizer_dir,
        "encoder_dir": encoder_dir,
        "simcse_dir": simcse_dir,
        "metrics_file": metrics_file,
    }
```

to:

```python
    return {
        "corpus_file": corpus_file,
        "tokenizer_dir": tokenizer_dir,
        "encoder_dir": encoder_dir,
        "simcse_dir": simcse_dir,
        "gpt_dir": gpt_dir,
        "metrics_file": metrics_file,
    }
```

- [ ] **Step 7: Add GPT-stage assertions to the e2e test file**

In `tests/e2e/test_pipeline_english.py`, after the existing `test_simcse_model_saved` function,
add:

```python
def test_gpt_model_saved(pipeline: dict) -> None:
    assert pipeline["gpt_dir"].exists()
    cfg_file = pipeline["gpt_dir"] / "config.json"
    assert cfg_file.exists()
    weight_files = list(pipeline["gpt_dir"].glob("*.safetensors")) + list(
        pipeline["gpt_dir"].glob("pytorch_model.bin")
    )
    assert weight_files, "no weight files in gpt_dir"


def test_gpt_model_reloadable_and_generates(pipeline: dict) -> None:
    from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

    model = GPT2LMHeadModel.from_pretrained(str(pipeline["gpt_dir"]))
    tok = PreTrainedTokenizerFast.from_pretrained(str(pipeline["gpt_dir"]))
    inputs = tok("The dog", return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    assert output.shape[1] > inputs["input_ids"].shape[1]
```

- [ ] **Step 8: Run the e2e suite to verify it passes**

Run: `pytest tests/e2e/test_pipeline_english.py -v -m e2e`
Expected: PASS (all tests in the file, including the two new ones and every pre-existing one —
confirms the fixture extension didn't break the pretrain/SimCSE/eval stages it shares with them)

- [ ] **Step 9: Lint and type-check**

Run:
```bash
ruff format src/langembed/llm/train_gpt.py tests/test_train_gpt.py tests/e2e/test_pipeline_english.py
ruff check src/langembed/llm/train_gpt.py tests/test_train_gpt.py tests/e2e/test_pipeline_english.py
mypy src/langembed/llm/train_gpt.py
```
Expected: all clean.

- [ ] **Step 10: Commit**

```bash
git add src/langembed/llm/__init__.py src/langembed/llm/train_gpt.py tests/test_train_gpt.py tests/e2e/test_pipeline_english.py
git commit -m "feat(llm): add GPT-style causal LM training warm-started from the encoder"
```

---

### Task 2: `run_pipeline.py` — wire up `--train-llm`

**Files:**
- Modify: `scripts/run_pipeline.py`
- Test: `tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: `train_gpt` is invoked via subprocess (`python -m langembed.llm.train_gpt`), not
  imported directly — matches how `run_pipeline.py` invokes every other stage module. No direct
  Python-level dependency on Task 1's module beyond that CLI contract.
- Produces: `calibrate_llm_steps(config_path: Path, target_minutes: float) -> int`, a new
  `--train-llm`/`--llm-minutes` CLI surface, and `artifacts/gpt_<lang>/` as this stage's output.
  Nothing later in this plan depends on this task (it's the last task).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_run_pipeline.py`, immediately after `test_svd_label_cli_flag_set` and before
`test_auto_label_method_rejects_unknown_value`:

```python
def test_train_llm_cli_flag_defaults():
    ap = run_pipeline.build_arg_parser()
    args = ap.parse_args(["--lang", "ru", "--input", "book.pdf"])

    assert args.train_llm is False
    assert args.llm_minutes == 25.0


def test_train_llm_cli_flag_set():
    ap = run_pipeline.build_arg_parser()
    args = ap.parse_args(
        ["--lang", "ru", "--input", "book.pdf", "--train-llm", "--llm-minutes", "10"]
    )

    assert args.train_llm is True
    assert args.llm_minutes == 10.0


def test_main_calls_train_gpt_when_train_llm_set():
    """Same source-text-check approach as test_eval_cfg_records_label_source (main() runs
    a long unmocked subprocess pipeline with no test coverage by design)."""
    source = (Path(__file__).resolve().parent.parent / "scripts" / "run_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "if args.train_llm:" in source
    assert "langembed.llm.train_gpt" in source
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_run_pipeline.py -v -k "train_llm"`
Expected: FAIL — `--train-llm`/`--llm-minutes` are unrecognized arguments, and the source-text
assertions don't find their strings yet.

- [ ] **Step 3: Add the CLI flags**

In `scripts/run_pipeline.py`, inside `build_arg_parser()`, immediately after the existing
`--svd-components` argument block (right before `return ap`), add:

```python
    ap.add_argument(
        "--train-llm",
        action="store_true",
        help=(
            "also train a GPT-style causal LM warm-started from the encoder's "
            "embedding table, after the final embeddings are produced (optional, off by default)"
        ),
    )
    ap.add_argument(
        "--llm-minutes", type=float, default=25.0, help="target GPT training wall time"
    )
```

- [ ] **Step 4: Add `calibrate_llm_steps`**

Immediately after the existing `calibrate_pretrain_steps` function (before `def start_server`),
add:

```python
def calibrate_llm_steps(config_path: Path, target_minutes: float) -> int:
    t0 = time.time()
    run([sys.executable, "-m", "langembed.llm.train_gpt", "--config", str(config_path), "--smoke"])
    elapsed = time.time() - t0
    smoke_steps = yaml.safe_load(config_path.read_text(encoding="utf-8"))["smoke"]["max_steps"]
    return max(50, round(smoke_steps * (target_minutes * 60) / elapsed))
```

- [ ] **Step 5: Add the new stage to `main()`**

In `scripts/run_pipeline.py`'s `main()`, find this exact block:

```python
        shutil.copy(metrics_src, out_dir / "eval.json")

    print(f"\nDone. Final embeddings for '{lang}': {embeddings_path}")
```

Replace it with:

```python
        shutil.copy(metrics_src, out_dir / "eval.json")

    if args.train_llm:
        print(f"=== [{lang}] GPT-style LLM (optional, warm-started from encoder) ===")
        llm_cfg: dict[str, Any] = {
            "seed": 42,
            "encoder_dir": f"artifacts/encoder_{lang}",
            "tokenizer_dir": f"artifacts/tokenizer_{lang}",
            "corpus_path": corpus_path,
            "out_dir": f"artifacts/gpt_{lang}",
            "model": {
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "intermediate_size": 1024,
                "max_position_embeddings": 130,
                "max_seq_length": 128,
            },
            "training": {
                "per_device_train_batch_size": 16,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0005,
                "weight_decay": 0.01,
                "warmup_steps": 200,
                "max_steps": 1500,
                "fp16": False,
                "save_steps": 500,
                "logging_steps": 50,
            },
            "smoke": {"max_steps": 50},
        }
        llm_path = cfg_dir / "llm.yaml"
        write_yaml(llm_path, llm_cfg)
        max_steps = calibrate_llm_steps(llm_path, args.llm_minutes)
        llm_cfg["training"]["max_steps"] = max_steps
        write_yaml(llm_path, llm_cfg)
        print(f"  calibrated max_steps={max_steps} (target ~{args.llm_minutes} min)")
        run([sys.executable, "-m", "langembed.llm.train_gpt", "--config", str(llm_path)])

    print(f"\nDone. Final embeddings for '{lang}': {embeddings_path}")
```

(`model.hidden_size` is `256` here, matching `pretrain_cfg["model"]["hidden_size"]` earlier in
this same function — required for `warm_start_embeddings` to succeed at runtime.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_run_pipeline.py -v`
Expected: PASS (all tests, including every pre-existing one — confirms the default path,
without `--train-llm`, is unchanged)

- [ ] **Step 7: Lint and type-check**

Run:
```bash
ruff format scripts/run_pipeline.py tests/test_run_pipeline.py
ruff check scripts/run_pipeline.py tests/test_run_pipeline.py
mypy scripts/run_pipeline.py
```
Expected: all clean (a pre-existing, unrelated `mypy` finding at `scripts/run_pipeline.py:29`,
"Unused type: ignore comment", predates this plan — do not fix it as part of this task; if it's
still present, confirm the reported line/message is byte-for-byte identical to what `git stash`
shows before this plan's changes, proving it isn't something this task introduced).

- [ ] **Step 8: Run the full test suite**

Run: `pytest tests/ -q --ignore=tests/e2e`
Expected: all pass, no regressions. (The e2e suite was already verified in Task 1, Step 8; no
need to re-run it here unless this task's changes touched e2e-covered code, which they don't.)

- [ ] **Step 9: Commit**

```bash
git add scripts/run_pipeline.py tests/test_run_pipeline.py
git commit -m "feat(pipeline): wire --train-llm into run_pipeline.py"
```
