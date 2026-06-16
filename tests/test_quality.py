from langembed.annotation.quality import aggregate, update_reliability, weighted_kappa


def test_kappa_perfect_agreement():
    assert weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0


def test_aggregate_respects_weights():
    assert aggregate([1.0, 5.0], [1.0, 0.0]) == 1.0


def test_reliability_in_unit_interval():
    r = update_reliability(8, 10)
    assert 0.0 <= r <= 1.0
