"""Near-duplicate removal via MinHash + LSH (Phase 1)."""

from __future__ import annotations

from collections.abc import Sequence

# A MinHashLSH index grows with every document it holds (each insert costs
# a num_perm-sized signature plus LSH bucket entries), so an index scoped to
# the whole corpus makes peak memory scale with corpus size -- this is what
# ran gu's ~15M-sentence corpus out of RAM. Capping the index to one batch at
# a time bounds memory to O(batch_size) regardless of corpus size, at the
# cost of only catching near-duplicates that land in the same batch.
DEFAULT_BATCH_SIZE = 200_000


def shingles(text: str, k: int = 5) -> set[str]:
    """Word k-shingles. Short texts collapse to a single shingle."""
    toks = text.split()
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def dedup(
    docs: Sequence[str],
    threshold: float = 0.8,
    num_perm: int = 128,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[str]:
    """Keep one representative per near-duplicate cluster.

    Near-duplicate detection runs within fixed-size batches rather than
    against one LSH index for the whole corpus, so memory stays bounded
    (~batch_size documents) regardless of total corpus size. A duplicate
    pair split across two batches will not be caught -- a deliberate
    memory/precision tradeoff for corpora too large to LSH-index whole.
    """
    from datasketch import MinHash, MinHashLSH

    kept: list[str] = []
    for batch_start in range(0, len(docs), batch_size):
        batch = docs[batch_start : batch_start + batch_size]
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        for i, d in enumerate(batch):
            m = MinHash(num_perm=num_perm)
            for s in shingles(d):
                m.update(s.encode("utf-8"))
            if not lsh.query(m):
                lsh.insert(str(i), m)
                kept.append(d)
    return kept
