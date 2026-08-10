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

    if not Path(cfg["encoder_dir"]).is_dir():
        raise FileNotFoundError(
            f"encoder_dir not found: {cfg['encoder_dir']!r} -- run the pretrain stage for "
            "this language first, or check the config's encoder_dir value"
        )
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
        save_total_limit=2,
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
