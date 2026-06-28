"""Branch C training: LoRA contrastive fine-tuning of an LLM embedder (Phase 4C).

Produces a SentenceTransformer-compatible model at train.out_dir, so the
existing eval harness scores branch C next to A (from scratch) and B (multilingual).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from langembed.config import load_config
from langembed.llm_embed.model import ST_POOLING


def build_model(cfg: dict[str, Any]) -> Any:
    from peft import LoraConfig
    from sentence_transformers import SentenceTransformer, models

    model_args: dict[str, Any] = {"torch_dtype": "float16"}
    if cfg.get("quantization", {}).get("load_in_4bit"):
        model_args["load_in_4bit"] = True

    base = cfg["base_model"]
    if cfg.get("mode") == "llm2vec" and cfg.get("mntp", {}).get("enable"):
        base = cfg["mntp"]["out_dir"]  # start from the MNTP-adapted checkpoint

    word = models.Transformer(base, max_seq_length=cfg["max_seq_length"], model_args=model_args)
    # LLM2Vec: enable bidirectional attention on the decoder backbone
    if cfg.get("mode") == "llm2vec":
        word.auto_model.config.is_causal = False
    lora = cfg["lora"]
    word.auto_model.add_adapter(
        LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            task_type="FEATURE_EXTRACTION",
        )
    )
    pool = models.Pooling(
        word.get_word_embedding_dimension(),
        pooling_mode=ST_POOLING.get(cfg["pooling"], "mean"),
    )
    # bake the instruction in as the default prompt so encode() (and the eval
    # harness) prepend it automatically and uniformly.
    instruction = cfg.get("instruction", "").strip()
    prompts = {"sts": instruction + ": "} if instruction else None
    return SentenceTransformer(
        modules=[word, pool], prompts=prompts, default_prompt_name="sts" if prompts else None
    )


def train_lora(cfg: dict[str, Any]) -> None:
    from sentence_transformers import InputExample, losses
    from torch.utils.data import DataLoader

    t = cfg["train"]
    path = Path(t["triplets_path"])
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"No triplets at {path}. Run Phase 5 (annotation export) first.")

    model = build_model(cfg)
    examples = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        examples.append(InputExample(texts=[r["anchor"], r["positive"], r["negative"]]))

    loader: DataLoader = DataLoader(examples, batch_size=t["batch_size"], shuffle=True)
    loss = losses.MultipleNegativesRankingLoss(model)  # InfoNCE: in-batch + hard negatives
    warmup = int(len(loader) * t["epochs"] * t["warmup_ratio"])
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=t["epochs"],
        warmup_steps=warmup,
        optimizer_params={"lr": t["learning_rate"]},
        output_path=t["out_dir"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train_lora(load_config(args.config))


if __name__ == "__main__":
    main()
