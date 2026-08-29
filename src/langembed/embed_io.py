"""Shared encode-and-write helpers for embed_corpus.py and embed_vocab.py.

Two families of writers:
- `encode_and_write_*`: stream a list of strings through a SentenceTransformer-like
  `model.encode()` in bounded chunks, writing as it goes. Used when the input list can
  be very large (e.g. embed_corpus.py's full sentence corpus), so peak memory stays
  O(chunk size) regardless of input size -- see ENCODE_CHUNK_SIZE's docstring for why
  that bound matters here specifically.
- `write_*`: write an already-computed (items, vectors) pair in one shot. Used when the
  vectors were produced some other way (e.g. embed_vocab.py's --method cbow trains its
  own embedding matrix) and the item count is inherently small (a vocabulary, not a
  full corpus), so there is nothing left to stream.
"""

from __future__ import annotations

import json
from pathlib import Path

from langembed.disk_guard import (
    check_free_space,
    estimate_binary_embedding_bytes,
    estimate_jsonl_embedding_bytes,
)

# Encoding+writing in fixed-size chunks bounds peak memory to O(chunk size)
# regardless of item count -- a single model.encode() call over an entire corpus
# held every embedding vector in memory at once before any writing started, which
# OOM-killed embed_corpus.py on a ~13.8M-sentence corpus.
ENCODE_CHUNK_SIZE = 50_000


def encode_and_write_jsonl(
    model, items: list[str], out_path: Path, text_field: str, min_free_gb: float
) -> int:
    """Stream-encode `items` and write one `{text_field: ..., "embedding": [...]}` JSON
    row per line. Returns the number of rows written."""
    dim = model.get_sentence_embedding_dimension()
    estimated_bytes = estimate_jsonl_embedding_bytes(len(items), dim)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    check_free_space(out_path.parent, min_free_gb, reserve_bytes=estimated_bytes)

    with out_path.open("w", encoding="utf-8") as f:
        for chunk_start in range(0, len(items), ENCODE_CHUNK_SIZE):
            chunk = items[chunk_start : chunk_start + ENCODE_CHUNK_SIZE]
            check_free_space(out_path.parent, min_free_gb)
            vecs = model.encode(chunk, normalize_embeddings=True, show_progress_bar=True)
            for text, vec in zip(chunk, vecs, strict=True):
                f.write(
                    json.dumps({text_field: text, "embedding": vec.tolist()}, ensure_ascii=False)
                    + "\n"
                )
    return len(items)


def encode_and_write_binary(model, items: list[str], out_path: Path, min_free_gb: float) -> int:
    """Stream-encode `items` into a float16 `.npy` array at `out_path`, plus a
    `<out_path>.meta.json` sidecar recording `{n, dim, dtype}`. No text is stored --
    row i corresponds to `items[i]`; callers keep their own ordered item list to zip
    back against the array. Roughly 5-6x smaller than the JSONL format (no per-row
    JSON overhead, no redundant text, no float64-precision text serialization of
    what are really float32 values). Returns the number of rows written."""
    import numpy as np

    dim = model.get_sentence_embedding_dimension()
    n = len(items)
    estimated_bytes = estimate_binary_embedding_bytes(n, dim)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    check_free_space(out_path.parent, min_free_gb, reserve_bytes=estimated_bytes)

    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, dim))
    try:
        for chunk_start in range(0, n, ENCODE_CHUNK_SIZE):
            chunk = items[chunk_start : chunk_start + ENCODE_CHUNK_SIZE]
            check_free_space(out_path.parent, min_free_gb)
            vecs = model.encode(chunk, normalize_embeddings=True, show_progress_bar=True)
            mm[chunk_start : chunk_start + len(chunk)] = np.asarray(
                [v.tolist() for v in vecs], dtype=np.float16
            )
    finally:
        mm.flush()
        del mm

    _write_meta(out_path, n, dim)
    return n


def write_jsonl_rows(
    items: list[str],
    vectors: list[list[float]],
    out_path: Path,
    text_field: str,
    min_free_gb: float,
) -> int:
    """Write a precomputed (items, vectors) pair as JSONL, one shot (no chunking --
    for inputs already known to be small, e.g. a vocabulary table)."""
    dim = len(vectors[0]) if vectors else 0
    estimated_bytes = estimate_jsonl_embedding_bytes(len(items), dim)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    check_free_space(out_path.parent, min_free_gb, reserve_bytes=estimated_bytes)

    with out_path.open("w", encoding="utf-8") as f:
        for text, vec in zip(items, vectors, strict=True):
            f.write(json.dumps({text_field: text, "embedding": vec}, ensure_ascii=False) + "\n")
    return len(items)


def write_binary_array(
    items: list[str], vectors: list[list[float]], out_path: Path, min_free_gb: float
) -> int:
    """Write a precomputed (items, vectors) pair as a float16 `.npy` array plus a
    `<out_path>.meta.json` sidecar, one shot (see write_jsonl_rows)."""
    import numpy as np

    dim = len(vectors[0]) if vectors else 0
    n = len(items)
    estimated_bytes = estimate_binary_embedding_bytes(n, dim)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    check_free_space(out_path.parent, min_free_gb, reserve_bytes=estimated_bytes)

    arr = np.asarray(vectors, dtype=np.float16) if n else np.zeros((0, dim), dtype=np.float16)
    np.save(out_path, arr)
    # np.save appends a .npy suffix if out_path doesn't already end with one.
    saved_path = (
        out_path if out_path.suffix == ".npy" else out_path.with_name(out_path.name + ".npy")
    )
    _write_meta(saved_path, n, dim)
    return n


def _write_meta(npy_path: Path, n: int, dim: int) -> None:
    meta_path = Path(str(npy_path) + ".meta.json")
    meta_path.write_text(
        json.dumps({"n": n, "dim": dim, "dtype": "float16"}, ensure_ascii=False), encoding="utf-8"
    )
