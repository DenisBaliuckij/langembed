import numpy as np

from langembed.annotation.active_learning import uncertainty_from_cosine


def test_boundary_is_most_uncertain():
    u = uncertainty_from_cosine(np.array([0.5, 1.0, 0.0]))
    assert u[0] == 1.0
    assert u[1] == 0.0
    assert u[2] == 0.0


def test_clipped_to_unit_interval():
    u = uncertainty_from_cosine(np.array([-0.5, 1.5]))
    assert (u >= 0).all() and (u <= 1).all()
