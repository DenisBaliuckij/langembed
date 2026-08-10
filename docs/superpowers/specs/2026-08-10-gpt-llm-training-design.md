# GPT-Style LLM Training Stage (Warm-Started from the Encoder) — Design

**Date:** 2026-08-10
**Status:** Approved

## Problem

The project's three architecture branches (from-scratch encoder + SimCSE, multilingual transfer,
LLM-as-embedder) all produce sentence *embeddings* — none produces a generative language model.
There is no path from a trained per-language encoder to an actual LLM a user could prompt for
text generation. Reference: [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch),
whose architecture (GPT-2-style decoder-only transformer, causal next-token-prediction training)
is the target shape for the new model — implemented via HuggingFace's off-the-shelf `GPT2Config`/
`GPT2LMHeadModel` rather than hand-rolled, to match this project's existing engineering style
(`RobertaForMaskedLM` for pretrain, `SentenceTransformer` for SimCSE, both HF-Trainer-based).

The existing per-language encoder (`artifacts/encoder_<lang>`, from `train_mlm.py`) is a
bidirectional RoBERTa-style encoder — its transformer blocks are not weight-compatible with a
causal decoder. "Warm-starting" a GPT from it is therefore scoped to what *is* transferable: the
token embedding table (same tokenizer → same vocabulary → same token IDs), not the attention
weights.

Must not become the default: training even a small GPT-style model is a meaningful extra
time/compute cost on top of the existing 6 pipeline stages, so it is opt-in, matching how
`--auto-label` and `--skip-eval` are already opt-in relative to `run_pipeline.py`'s default flow.

## Solution

Add `--train-llm` (opt-in, default off) and `--llm-minutes` (calibration target, default 25.0,
mirroring `--pretrain-minutes`) to `run_pipeline.py`. When set, a new stage runs after the
existing step 6/6, training a GPT-2-style causal LM on the same corpus and tokenizer already built
earlier in the pipeline.

### 1. `src/langembed/llm/train_gpt.py` (new package `src/langembed/llm/`)

Mirrors `train_mlm.py`'s shape and conventions exactly (HF `Trainer`, function-local heavy
imports, `--smoke` flag, `main()`/`build_arg_parser`-free — config-driven like every other stage):

```python
def warm_start_embeddings(gpt_model, encoder_model) -> None:
    """Copy the encoder's token embedding table into the GPT model's embedding table.
    Both share the same tokenizer/vocabulary (same vocab_size, same token IDs), so this
    is a direct weight copy, not a projection. Requires equal hidden_size on both models
    (enforced by configs/<lang>/llm.yaml mirroring pretrain.yaml's hidden_size) --
    raises ValueError naming both shapes if they don't match, since a silent skip would
    produce a model that looks warm-started but isn't.
    """

def train_gpt(cfg: dict[str, Any], smoke: bool = False) -> None:
    """Build a GPT2Config from cfg["model"], construct GPT2LMHeadModel, warm-start its
    embedding table from cfg["encoder_dir"]'s RobertaForMaskedLM, then train via HF
    Trainer for causal LM (next-token prediction, no MLM masking) on cfg["corpus_path"],
    saving to cfg["out_dir"]. Mirrors train_mlm.py's Trainer/TrainingArguments wiring;
    max_steps comes from cfg["training"] (smoke uses cfg["smoke"]["max_steps"]), never
    computed inline -- calibration happens the same way pretrain's does, in
    run_pipeline.py, not in this module.
    """
```

`train_gpt`'s corpus loading uses the same `datasets.load_dataset("text", ...)` + tokenizer `.map()`
pattern as `train_mlm.py` (Arrow-backed, not a Python list) — this project has had three separate
OOM incidents this session from corpus-size-proportional Python-list memory, so the new stage
starts from the pattern that's already proven safe at scale, not the one that wasn't.

### 2. `scripts/run_pipeline.py`

New CLI flags:

```
--train-llm       action="store_true", default False
--llm-minutes      type=float, default 25.0
```

New stage after the existing step 6/6 (none of the 6 existing stage-print statements change;
this one prints its own header, not folded into "N/6" numbering):

```python
if args.train_llm:
    print(f"=== [{lang}] GPT-style LLM (optional, warm-started from encoder) ===")
    llm_cfg: dict[str, Any] = {
        "seed": 42,
        "encoder_dir": f"artifacts/encoder_{lang}",
        "tokenizer_dir": f"artifacts/tokenizer_{lang}",
        "corpus_path": corpus_path,
        "out_dir": f"artifacts/gpt_{lang}",
        "model": {
            "hidden_size": 256,           # must match pretrain_cfg["model"]["hidden_size"]
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
```

`model.hidden_size` is hardcoded to `256` here (not `args.something`), deliberately: it must equal
`pretrain_cfg["model"]["hidden_size"]` (also `256`, also hardcoded in the existing code this
mirrors) for `warm_start_embeddings` to succeed, and the rest of `pretrain_cfg["model"]` is
hardcoded the same way today — this stage stays consistent with that existing precedent rather
than introducing a new independently-configurable size.

`calibrate_llm_steps(config_path, target_minutes)` is a near-duplicate of the existing
`calibrate_pretrain_steps`, invoking `langembed.llm.train_gpt --smoke` instead of
`langembed.pretrain.train_mlm --smoke`:

```python
def calibrate_llm_steps(config_path: Path, target_minutes: float) -> int:
    t0 = time.time()
    run([sys.executable, "-m", "langembed.llm.train_gpt", "--config", str(config_path), "--smoke"])
    elapsed = time.time() - t0
    smoke_steps = yaml.safe_load(config_path.read_text(encoding="utf-8"))["smoke"]["max_steps"]
    return max(50, round(smoke_steps * (target_minutes * 60) / elapsed))
```

Kept as a separate function rather than generalizing the existing one to take a module-path
parameter — `calibrate_pretrain_steps` is covered by existing usage/tests and this task doesn't
need to risk that stage's working code to add this one.

### Data flow

1. Stages 1–6 run exactly as today (unaffected whether or not `--train-llm` is set).
2. If `--train-llm`: a smoke run of `train_gpt` calibrates `max_steps` to hit `--llm-minutes`
   wall-time, then the real run reads `data/corpus_<lang>.txt` (already built) and
   `artifacts/tokenizer_<lang>` (already trained), constructs a fresh `GPT2LMHeadModel`, copies
   `artifacts/encoder_<lang>`'s embedding table into it, and trains via causal LM, saving to
   `artifacts/gpt_<lang>/`.

## Testing

- `tests/test_train_gpt.py`: `warm_start_embeddings` tested in isolation — construct a small real
  `RobertaForMaskedLM` and a small real `GPT2LMHeadModel` with matching `hidden_size`/`vocab_size`,
  call it, assert the GPT model's embedding weights now equal the encoder's; a mismatched
  `hidden_size` pair raises `ValueError` naming both shapes.
- `tests/e2e/test_pipeline_english.py`: extend with a real (unmocked) smoke-scale run — build a
  tiny encoder + tokenizer fixture (already exists in this file for the pretrain/SimCSE stages),
  call `train_gpt(cfg, smoke=True)`, assert `artifacts/gpt_.../` is created and non-empty and the
  saved model can be reloaded via `GPT2LMHeadModel.from_pretrained`. Matches how pretrain/SimCSE
  are tested today — real HF training on a tiny fixture, not mocked.
- `tests/test_run_pipeline.py`: CLI flag defaults/set tests (`--train-llm` off by default,
  `--llm-minutes` defaults to 25.0) using real `ArgumentParser` invocations, matching the existing
  convention for this file's other flags.

## Out of scope

- Generalizing `calibrate_pretrain_steps`/`calibrate_llm_steps` into one shared function — separate
  functions for now; revisit only if a third caller appears.
- Any transformer-block weight transfer beyond the embedding table (e.g. partial attention-weight
  initialization) — confirmed with the user during design that embedding-table warm-start is the
  full scope of "warm-start" for this feature.
- Making the GPT model's `hidden_size` independently configurable from the encoder's — would break
  the warm-start's direct weight copy; out of scope until there's a concrete need for different
  sizes (at which point a projection layer, not a direct copy, would be needed).
- Wiring `artifacts/gpt_<lang>` into `output/<lang>/` or the final deliverable directory the way
  `embed_corpus.py`'s output is — this stage's artifact lives under `artifacts/` like every other
  intermediate model, not `output/`, consistent with `encoder_<lang>`/`simcse_<lang>` never being
  copied to `output/` either (only the final `embeddings.jsonl` and `eval.json` are).
