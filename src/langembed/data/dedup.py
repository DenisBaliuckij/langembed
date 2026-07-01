"""Near-duplicate removal via MinHash + LSH (Phase 1)."""

from __future__ import annotations

from collections.abc import Sequence


def shingles(text: str, k: int = 5) -> set[str]:
    """Word k-shingles. Short texts collapse to a single shingle."""
    toks = text.split()
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def dedup(docs: Sequence[str], threshold: float = 0.8, num_perm: int = 128) -> list[str]:
    """Keep one representative per near-duplicate cluster."""
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[str] = []
    for i, d in enumerate(docs):
        m = MinHash(num_perm=num_perm)
        for s in shingles(d):
            m.update(s.encode("utf-8"))
        if not lsh.query(m):
            lsh.insert(str(i), m)
            kept.append(d)
    return kept
