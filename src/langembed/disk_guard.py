"""Guard against a long-running write filling up the disk.

A per-language corpus can be far larger than expected (NLLB coverage
varies wildly by language), and a write loop that streams one JSON line
per sentence has no way to know that in advance. This module lets a
long-running write estimate its footprint before spending GPU time on it,
and check available space periodically while it runs, so it stops itself
instead of silently consuming the disk.
"""

from __future__ import annotations

import shutil
from pathlib import Path

BYTES_PER_GB = 1024**3


class DiskSpaceError(RuntimeError):
    """Raised when free disk space is, or would become, below the configured minimum."""


def check_free_space(path: Path, min_free_gb: float, *, reserve_bytes: int = 0) -> None:
    """Raise DiskSpaceError if free space at `path`, minus `reserve_bytes`, is below min_free_gb."""
    free_bytes = shutil.disk_usage(path).free - reserve_bytes
    free_gb = free_bytes / BYTES_PER_GB
    if free_gb < min_free_gb:
        raise DiskSpaceError(
            f"only {free_gb:.2f}GB free at {path} after reserving "
            f"{reserve_bytes / BYTES_PER_GB:.2f}GB (minimum required: {min_free_gb}GB)"
        )


def estimate_jsonl_embedding_bytes(n_rows: int, dim: int, *, bytes_per_float: int = 18) -> int:
    """Rough upper-bound estimate for a JSONL file of `{"text": ..., "embedding": [floats]}` rows.

    `bytes_per_float` covers a serialized float like `-0.123456789,` (~18 bytes
    including its separator) plus per-row JSON overhead; deliberately generous
    so the estimate over- rather than under-shoots.
    """
    return n_rows * dim * bytes_per_float


def estimate_binary_embedding_bytes(n_rows: int, dim: int, *, bytes_per_value: int = 2) -> int:
    """Estimate for a raw float16 `.npy` embedding array (no per-row text/JSON overhead).

    `bytes_per_value=2` is float16; the `.npy` format's fixed header is a few hundred
    bytes at most, negligible next to the array itself.
    """
    return n_rows * dim * bytes_per_value + 4096
