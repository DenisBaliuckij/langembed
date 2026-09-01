"""Phase 2: train a from-scratch sub-word tokenizer on the target corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from langembed.config import load_config
from langembed.data.reservoir_sample import reservoir_sample

# Bounds tokenizer training data size (and therefore peak memory) regardless of
# raw corpus size. BpeTrainer's word/pretoken frequency table grows with total
# input size; my's ~6M-sentence corpus and ml's ~18M-sentence corpus both
# SIGKILL-OOM'd training on the full corpus. Vocab quality saturates well
# before this many sentences, so bounding here doesn't cost coverage.
MAX_TOKENIZER_TRAIN_SENTENCES = 1_000_000


def train_tokenizer(
    corpus_path: str,
    out_dir: str,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    max_train_sentences: int = MAX_TOKENIZER_TRAIN_SENTENCES,
    seed: int = 42,
) -> Any:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    sentences = reservoir_sample(corpus_path, max_train_sentences, seed)
    tok.train_from_iterator(sentences, trainer)
    # Positional, not keyword args: newer tokenizers releases' pybind
    # bindings don't accept `cls=` as a keyword (TypeError: unexpected
    # keyword argument 'cls') even though the library's own docs show it as
    # a named parameter -- positional matches the library's own example.
    tok.post_processor = processors.RobertaProcessing(
        ("</s>", tok.token_to_id("</s>")),
        ("<s>", tok.token_to_id("<s>")),
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tok.save(str(Path(out_dir) / "tokenizer.json"))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    fast.save_pretrained(out_dir)
    return fast


def diagnose(tokenizer: Any, sample_path: str) -> dict[str, float]:
    """Sanity metrics: <unk> rate and sub-tokens per word."""
    unk_id = tokenizer.unk_token_id
    n_unk = n_tok = n_words = 0
    for line in Path(sample_path).open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        ids = tokenizer(line)["input_ids"]
        n_unk += sum(1 for i in ids if i == unk_id)
        n_tok += len(ids)
        n_words += len(line.split())
    return {
        "unk_rate": n_unk / max(n_tok, 1),
        "subtokens_per_word": n_tok / max(n_words, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    t = cfg["tokenizer"]
    fast = train_tokenizer(
        cfg["data"]["out_path"],
        t["out_dir"],
        t["vocab_size"],
        t["min_frequency"],
        seed=cfg.get("seed", 42),
    )
    stats = diagnose(fast, cfg["data"]["out_path"])
    print("diagnostics:", stats)
    assert stats["unk_rate"] <= t["unk_rate_max"], "unk_rate too high — revisit vocab_size"


if __name__ == "__main__":
    main()
