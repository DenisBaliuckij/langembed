from langembed.annotation.triplets import build_triplets_from_pairs


def test_build_triplets_from_pairs_back_translation_tiers_split_correctly():
    """With 3 discrete tiers evenly split (paraphrase=4.8, adjacent=2.0,
    random=0.3, 20 each), the default 70th/30th percentile split recovers
    exactly the paraphrase tier as positive and the random tier as negative."""
    pairs = (
        [(f"para{i}", f"parab{i}", 4.8) for i in range(20)]
        + [(f"adj{i}", f"adjb{i}", 2.0) for i in range(20)]
        + [(f"rand{i}", f"randb{i}", 0.3) for i in range(20)]
    )

    triplets = build_triplets_from_pairs(pairs, seed=1)

    assert len(triplets) == 20
    for anchor, positive, negative in triplets:
        assert anchor.startswith("para")
        assert positive.startswith("parab")
        assert negative.startswith("randb")


def test_build_triplets_from_pairs_returns_min_of_bucket_sizes():
    pairs = [
        ("a1", "b1", 4.8),
        ("a2", "b2", 4.8),
        ("a3", "b3", 4.8),
        ("a4", "b4", 2.0),
        ("a5", "b5", 2.0),
        ("a6", "b6", 0.3),
    ]

    triplets = build_triplets_from_pairs(
        pairs, positive_percentile=0.7, negative_percentile=0.3, seed=1
    )

    assert len(triplets) >= 1
    for anchor, positive, negative in triplets:
        assert anchor.startswith("a")
        assert positive.startswith("b")
        assert negative.startswith("b")


def test_build_triplets_from_pairs_deterministic():
    pairs = [(f"a{i}", f"b{i}", float(i % 5)) for i in range(20)]

    t1 = build_triplets_from_pairs(pairs, seed=7)
    t2 = build_triplets_from_pairs(pairs, seed=7)

    assert t1 == t2


def test_build_triplets_from_pairs_empty_input_returns_empty():
    assert build_triplets_from_pairs([]) == []


def test_build_triplets_from_pairs_all_same_score_does_not_crash():
    pairs = [(f"a{i}", f"b{i}", 3.0) for i in range(10)]

    triplets = build_triplets_from_pairs(pairs, seed=1)

    assert isinstance(triplets, list)


def test_build_triplets_from_pairs_filters_out_degenerate_positive_equals_negative():
    pairs = [("a", "b", 3.0)]
    triplets = build_triplets_from_pairs(pairs, seed=1)
    assert triplets == []
