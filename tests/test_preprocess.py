from langembed.preprocess import normalize


def test_idempotent():
    s = "  hello    world   "
    once = normalize(s)
    assert normalize(once) == once


def test_collapses_and_strips():
    assert normalize("a   b\tc\n") == "a b c"
