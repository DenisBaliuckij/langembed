import pytest

pytest.importorskip("datasketch")

from langembed.data.dedup import dedup, shingles  # noqa: E402


def test_shingles_short():
    assert shingles("a b") == {"a b"}


def test_dedup_collapses_near_duplicates():
    a = "the quick brown fox jumps over the lazy dog today here"
    b = "the quick brown fox jumps over the lazy dog now here"
    c = "completely different content about another subject entirely over here"
    out = dedup([a, b, c], threshold=0.4)
    assert len(out) == 2


def test_dedup_catches_near_duplicates_within_a_batch():
    a = "the quick brown fox jumps over the lazy dog today here"
    b = "the quick brown fox jumps over the lazy dog now here"
    c = "completely different content about another subject entirely over here"
    # Both near-duplicates land in the same (single) batch, so they're still caught.
    out = dedup([a, b, c], threshold=0.4, batch_size=3)
    assert len(out) == 2


def test_dedup_does_not_catch_near_duplicates_across_batch_boundaries():
    a = "the quick brown fox jumps over the lazy dog today here"
    b = "the quick brown fox jumps over the lazy dog now here"
    c = "completely different content about another subject entirely over here"
    # a and b are split across two separate batches, so each batch's LSH index
    # never sees both -- this is the documented memory/precision tradeoff.
    out = dedup([a, b, c], threshold=0.4, batch_size=1)
    assert len(out) == 3


def test_dedup_processes_more_docs_than_one_batch():
    docs = [f"unique sentence number {i} about something distinct" for i in range(25)]
    out = dedup(docs, batch_size=10)
    assert len(out) == 25
