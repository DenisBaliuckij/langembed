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
