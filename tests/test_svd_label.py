from langembed.annotation import svd_label


def _sentences(n: int) -> list[str]:
    return [f"the quick brown fox sentence number {i} about topic {i % 4}" for i in range(n)]


def test_build_svd_sts_pairs_returns_n_pairs():
    pairs = svd_label.build_svd_sts_pairs(_sentences(20), n=9, n_components=3, seed=1)
    assert len(pairs) == 9


def test_build_svd_sts_pairs_scores_in_range():
    pairs = svd_label.build_svd_sts_pairs(_sentences(20), n=15, n_components=3, seed=1)
    for _, _, score in pairs:
        assert 0.0 <= score <= 5.0


def test_build_svd_sts_pairs_pairs_come_from_input_sentences():
    sentences = _sentences(20)
    pairs = svd_label.build_svd_sts_pairs(sentences, n=9, n_components=3, seed=1)
    for a, b, _ in pairs:
        assert a in sentences
        assert b in sentences


def test_build_svd_sts_pairs_subsamples_above_max_fit_sentences():
    pairs = svd_label.build_svd_sts_pairs(
        _sentences(20), n=9, n_components=3, seed=1, max_fit_sentences=5
    )
    used = {a for a, _, _ in pairs} | {b for _, b, _ in pairs}
    assert len(used) <= 5


def test_build_svd_sts_pairs_no_subsampling_below_threshold():
    """When the corpus is at or below max_fit_sentences, the whole corpus is used
    unmodified -- two calls with different (but both non-restrictive) max_fit_sentences
    values must produce identical output, since neither actually subsamples."""
    sentences = _sentences(20)
    a = svd_label.build_svd_sts_pairs(sentences, n=9, n_components=3, seed=7, max_fit_sentences=20)
    b = svd_label.build_svd_sts_pairs(
        sentences, n=9, n_components=3, seed=7, max_fit_sentences=1000
    )
    assert a == b


def test_build_svd_sts_pairs_deterministic():
    kwargs = dict(sentences=_sentences(20), n=9, n_components=3, seed=7)
    assert svd_label.build_svd_sts_pairs(**kwargs) == svd_label.build_svd_sts_pairs(**kwargs)


def test_build_svd_sts_pairs_too_few_sentences_returns_empty():
    assert svd_label.build_svd_sts_pairs(["only one sentence"], n=9) == []
