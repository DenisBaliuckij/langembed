"""Encode every corpus sentence with the trained SimCSE model (Phase 6 deliverable, ru track)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langembed.config import load_config


def embed_corpus(config_path: str, out_path: str) -> int:
    import datasets  # noqa: F401
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    import torch  # noqa: F401
    from sentence_transformers import SentenceTransformer

    cfg = load_config(config_path)
    model = SentenceTransformer(cfg["simcse"]["out_dir"])
    with open(cfg["simcse"]["sentences_path"], encoding="utf-8") as f:
        sentences = [ln.strip() for ln in f if ln.strip()]
    vecs = model.encode(sentences, normalize_embeddings=True, show_progress_bar=True)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for text, vec in zip(sentences, vecs, strict=True):
            f.write(json.dumps({"text": text, "embedding": vec.tolist()}, ensure_ascii=False) + "\n")
    return len(sentences)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="artifacts/embeddings_ru/embeddings.jsonl")
    args = ap.parse_args()
    n = embed_corpus(args.config, args.out)
    print(f"embedded sentences: {n}")


if __name__ == "__main__":
    main()
