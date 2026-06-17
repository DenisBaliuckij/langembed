"""Phase 3: from-scratch MLM pre-training of the encoder (random init)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from langembed.config import load_config


def train_mlm(cfg: dict[str, Any], smoke: bool = False) -> None:
    import torch
    from datasets import load_dataset
    from transformers import (
        DataCollatorForLanguageModeling,
        PreTrainedTokenizerFast,
        RobertaConfig,
        RobertaForMaskedLM,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(cfg.get("seed", 42))
    tok = PreTrainedTokenizerFast.from_pretrained(cfg["tokenizer_dir"])
    m = cfg["model"]
    config = RobertaConfig(
        vocab_size=tok.vocab_size,
        max_position_embeddings=m["max_position_embeddings"],
        hidden_size=m["hidden_size"],
        num_hidden_layers=m["num_hidden_layers"],
        num_attention_heads=m["num_attention_heads"],
        intermediate_size=m["intermediate_size"],
        type_vocab_size=1,
    )
    model = RobertaForMaskedLM(config)  # random init == from scratch
    print("parameters:", model.num_parameters())

    ds = load_dataset("text", data_files={"train": cfg["corpus_path"]})["train"]
    ds = ds.map(
        lambda b: tok(b["text"], truncation=True, max_length=m["max_seq_length"]),
        batched=True,
        remove_columns=["text"],
    )
    collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=True, mlm_probability=cfg["training"]["mlm_probability"]
    )

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
    train_mlm(load_config(args.config), smoke=args.smoke)


if __name__ == "__main__":
    main()
