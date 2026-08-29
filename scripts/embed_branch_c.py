"""Branch C: LoRA-tuned decoder-LLM embedder (Qwen3-Embedding-0.6B by default),
fine-tuned on the same SVD-derived triplets used for Branches A/B so all methods are
compared on identical training signal, then embeds a corpus sample the same way
embed_corpus.py does. Produces artifacts/embed_<lang>_c_llm and
output/<lang>/embeddings_c_llm.jsonl.

Expects `data/triplets_<lang>_<label-method>.jsonl` to already exist -- produced by
`scripts/supervised_finetune_pass.py --lang <lang> --label-method <label-method>`,
which scripts/run_all_branches.py runs first for Branch A.

Usage:
    python scripts/embed_branch_c.py --lang mr
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_TAG = "c_llm"
DEFAULT_EMBED_SAMPLE_SIZE = 200


def build_llm_embed_config(
    lang: str, triplets_path: Path, base_model: str, out_dir: Path
) -> dict[str, Any]:
    return {
        "seed": 42,
        "mode": "ready_embedder",
        "base_model": base_model,
        "pooling": "last_token",
        "instruction": f"Represent this {lang} sentence for semantic similarity",
        "max_seq_length": 256,
        "quantization": {"load_in_4bit": True},
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "mntp": {
            "enable": False,
            "corpus_path": f"data/corpus_{lang}.txt",
            "out_dir": f"artifacts/llm_mntp_{lang}",
        },
        "train": {
            "triplets_path": str(triplets_path),
            "out_dir": str(out_dir),
            "batch_size": 16,
            "epochs": 3,
            "learning_rate": 0.0002,
            "warmup_ratio": 0.05,
        },
    }


def embed_branch_c(
    lang: str,
    label_method: str = "svd",
    base_model: str = "Qwen/Qwen3-Embedding-0.6B",
    embed_sample_size: int = DEFAULT_EMBED_SAMPLE_SIZE,
    min_free_gb: float = 5.0,
) -> int:
    from langembed import embed_io
    from langembed.llm_embed.train_lora import train_lora

    triplets_path = REPO_ROOT / f"data/triplets_{lang}_{label_method}.jsonl"
    if not triplets_path.is_file() or triplets_path.stat().st_size == 0:
        raise SystemExit(
            f"No triplets at {triplets_path}. Run scripts/supervised_finetune_pass.py "
            f"--lang {lang} --label-method {label_method} first (or use run_all_branches.py)."
        )

    out_dir = REPO_ROOT / f"artifacts/embed_{lang}_{OUT_TAG}"
    llm_cfg = build_llm_embed_config(lang, triplets_path, base_model, out_dir)
    cfg_path = REPO_ROOT / "configs" / lang / "llm_embed.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(llm_cfg, f, allow_unicode=True, sort_keys=False)

    print(f"=== [{lang}] Branch C: LoRA fine-tune ({base_model}) ===")
    train_lora(llm_cfg)
    print(f"  fine-tuned model -> {out_dir}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(out_dir))
    with (REPO_ROOT / f"data/corpus_{lang}.txt").open(encoding="utf-8") as f:
        sentences = [ln.strip() for ln in f if ln.strip()][:embed_sample_size]

    out_path = REPO_ROOT / f"output/{lang}/embeddings_{OUT_TAG}.jsonl"
    n = embed_io.encode_and_write_jsonl(model, sentences, out_path, "text", min_free_gb)
    print(f"  wrote embeddings -> {out_path}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lang", required=True)
    ap.add_argument("--label-method", default="svd", choices=["svd", "backtranslation", "native"])
    ap.add_argument("--base-model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--embed-sample-size", type=int, default=DEFAULT_EMBED_SAMPLE_SIZE)
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    args = ap.parse_args()
    n = embed_branch_c(
        args.lang,
        args.label_method,
        args.base_model,
        args.embed_sample_size,
        args.min_free_gb,
    )
    print(f"embedded sentences: {n}")


if __name__ == "__main__":
    main()
