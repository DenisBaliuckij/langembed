"""Pure helpers for the LLM embedder (Branch C).

The heavy model wrapping is done in train_lora.py via sentence-transformers so
that the resulting artifact is a drop-in SentenceTransformer (and the existing
eval harness can score branch C alongside A and B). These helpers are framework
free and unit-tested.
"""

from __future__ import annotations

import numpy as np

# sentence-transformers Pooling mode names
ST_POOLING = {"mean": "mean", "last_token": "lasttoken"}


def format_instruction(instruction: str, text: str) -> str:
    """e5-mistral / instruction-embedding prompt format."""
    instruction = instruction.strip()
    if not instruction:
        return text
    return f"Instruct: {instruction}\nQuery: {text}"


def last_token_indices(attention_mask: np.ndarray) -> np.ndarray:
    """Index of the last real token per row, handling left- or right-padding.

    Decoder LLMs are often left-padded for batching; in that case the last token
    is always the final column. Otherwise we use (length - 1).
    """
    attention_mask = np.asarray(attention_mask)
    n, seq = attention_mask.shape
    left_padded = bool(attention_mask[:, -1].sum() == n)
    if left_padded:
        return np.full(n, seq - 1, dtype=int)
    return attention_mask.sum(axis=1).astype(int) - 1
