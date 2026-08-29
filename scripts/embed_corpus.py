"""Encode corpus sentences with the trained SimCSE model (Phase 6 deliverable, ru track).

Defaults to embedding only `--limit` sentences rather than the whole corpus: nothing
in this codebase consumes a full-corpus embedding dump today (the `/embed` API
computes vectors on demand -- see docs/IMPLEMENTATION_PLAN.md's pgvector design --
and verify_serve_skew.py only reads the first `--n` rows of this file). Embedding
entire multi-million-sentence corpora here produced 40-220GB JSONL files per
language and was a direct contributor to the disk/memory watchdog kills during the
2026-08-10 pipeline run. Pass a larger/absent --limit for an intentional full-corpus
dump, and `--format binary` to avoid the JSON float64-text bloat when you do (see
langembed.embed_io for the size comparison).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from langembed import embed_io
from langembed.config import load_config
from langembed.disk_guard import DiskSpaceError

DEFAULT_LIMIT = 200


def embed_corpus(
    config_path: str,
    out_path: str,
    min_free_gb: float = 5.0,
    limit: int | None = DEFAULT_LIMIT,
    fmt: str = "jsonl",
) -> int:
    import datasets  # noqa: F401
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    import torch  # noqa: F401
    from sentence_transformers import SentenceTransformer

    cfg = load_config(config_path)
    model = SentenceTransformer(cfg["simcse"]["out_dir"])
    with open(cfg["simcse"]["sentences_path"], encoding="utf-8") as f:
        sentences = [ln.strip() for ln in f if ln.strip()]
    if limit is not None:
        sentences = sentences[:limit]

    out = Path(out_path)
    if fmt == "binary":
        return embed_io.encode_and_write_binary(model, sentences, out, min_free_gb)
    return embed_io.encode_and_write_jsonl(model, sentences, out, "text", min_free_gb)


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
    ap.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=(
            f"embed only the first N sentences instead of the whole corpus "
            f"(default: {DEFAULT_LIMIT}). Pass a value larger than the corpus, or "
            "handle 0/negative as 'no limit' via your own wrapper, for an intentional "
            "full-corpus dump -- nothing in this codebase needs one today."
        ),
    )
    ap.add_argument(
        "--format",
        choices=["jsonl", "binary"],
        default="jsonl",
        help=(
            '\'jsonl\' (default): one {"text": ..., "embedding": [...]} row per line, '
            "human-readable, needed by verify_serve_skew.py. 'binary': a float16 .npy "
            "array plus a .meta.json sidecar -- ~5-6x smaller, no redundant sentence "
            "text (row i corresponds to input line i), for when you do want a large dump."
        ),
    )
    args = ap.parse_args()
    limit = None if args.limit is not None and args.limit <= 0 else args.limit
    try:
        n = embed_corpus(
            args.config, args.out, min_free_gb=args.min_free_gb, limit=limit, fmt=args.format
        )
    except DiskSpaceError as e:
        print(f"aborted: {e}")
        raise SystemExit(1) from e
    print(f"embedded sentences: {n}")


if __name__ == "__main__":
    main()
