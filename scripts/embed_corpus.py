"""Encode every corpus sentence with the trained SimCSE model (Phase 6 deliverable, ru track)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langembed.config import load_config
from langembed.disk_guard import DiskSpaceError, check_free_space, estimate_jsonl_embedding_bytes

# How often (in written rows) to re-check free disk space during the write loop.
DISK_CHECK_INTERVAL = 2000


def embed_corpus(config_path: str, out_path: str, min_free_gb: float = 5.0) -> int:
    import datasets  # noqa: F401
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    import torch  # noqa: F401
    from sentence_transformers import SentenceTransformer

    cfg = load_config(config_path)
    model = SentenceTransformer(cfg["simcse"]["out_dir"])
    with open(cfg["simcse"]["sentences_path"], encoding="utf-8") as f:
        sentences = [ln.strip() for ln in f if ln.strip()]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Fail fast, before spending GPU time encoding, if the corpus is large
    # enough that its output would not fit.
    dim = model.get_sentence_embedding_dimension()
    estimated_bytes = estimate_jsonl_embedding_bytes(len(sentences), dim)
    check_free_space(out.parent, min_free_gb, reserve_bytes=estimated_bytes)

    vecs = model.encode(sentences, normalize_embeddings=True, show_progress_bar=True)

    with out.open("w", encoding="utf-8") as f:
        for i, (text, vec) in enumerate(zip(sentences, vecs, strict=True)):
            if i % DISK_CHECK_INTERVAL == 0:
                check_free_space(out.parent, min_free_gb)
            f.write(
                json.dumps({"text": text, "embedding": vec.tolist()}, ensure_ascii=False) + "\n"
            )
    return len(sentences)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="artifacts/embeddings_ru/embeddings.jsonl")
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=5.0,
        help="Abort before/during the write if free disk space drops below this many GB",
    )
    args = ap.parse_args()
    try:
        n = embed_corpus(args.config, args.out, min_free_gb=args.min_free_gb)
    except DiskSpaceError as e:
        print(f"aborted: {e}")
        raise SystemExit(1) from e
    print(f"embedded sentences: {n}")


if __name__ == "__main__":
    main()
