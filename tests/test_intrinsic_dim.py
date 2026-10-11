"""Intrinsic-dimension estimators on synthetic data with a known dimension."""

from __future__ import annotations

import numpy as np
import pytest

from langembed.eval.intrinsic_dim import (
    anisotropy,
    drop_duplicates,
    estimate_all,
    mle,
    mst_edge_weights,
    pairwise_distances,
    pca_dims,
    prepare,
    schweinhart,
    twonn,
)


def _embedded_cube(n: int, d: int, ambient: int = 64, seed: int = 0) -> np.ndarray:
    """n points uniform in [0, 1]^d, rotated into an ``ambient``-dimensional space."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(ambient, d)))
    return rng.uniform(size=(n, d)) @ q.T


def test_mst_matches_scipy() -> None:
    from scipy.sparse.csgraph import minimum_spanning_tree

    x = np.random.default_rng(1).normal(size=(300, 5))
    dist = pairwise_distances(x)
    ours = np.sort(mst_edge_weights(dist))
    ref = np.sort(minimum_spanning_tree(dist).data)
    assert np.allclose(ours, ref)


@pytest.mark.parametrize("d", [2, 5])
def test_estimators_recover_known_dimension(d: int) -> None:
    dist = pairwise_distances(_embedded_cube(2000, d))
    assert twonn(dist) == pytest.approx(d, rel=0.2)
    assert mle(dist) == pytest.approx(d, rel=0.2)
    sizes = np.geomspace(200, 2000, 8).astype(int).tolist()
    (res,) = schweinhart(dist, sizes, alphas=(1.0,), seed=0)
    assert res.dimension == pytest.approx(d, rel=0.3)


def test_pca_and_participation_ratio_on_flat_subspace() -> None:
    x = _embedded_cube(1000, 3)
    dims = pca_dims(x)
    assert dims["pca95"] <= 3
    assert 2.0 <= dims["participation_ratio"] <= 3.0


def test_prepare_and_anisotropy() -> None:
    rng = np.random.default_rng(2)
    shifted = rng.normal(size=(500, 16)) + 5.0  # all vectors point the same way
    assert anisotropy(shifted) > 0.8
    centered = prepare(shifted, "centered")
    assert np.allclose(np.linalg.norm(centered, axis=1), 1.0)
    assert anisotropy(centered) == pytest.approx(0.0, abs=0.05)
    with pytest.raises(ValueError):
        prepare(shifted, "whitened")


def test_drop_duplicates_and_estimate_all() -> None:
    x = _embedded_cube(600, 4)
    x = np.vstack([x, x[:50]])  # 50 duplicate rows
    _, keep = drop_duplicates(prepare(x))
    assert keep.size == 600
    out = estimate_all(x, prep="unit", sample_size=500, n_sizes=6, alphas=(1.0,), seed=3)
    assert out["n_points"] == 500 and out["n_unique_available"] == 600
    assert {"twonn", "mle", "schweinhart_a1", "pca90", "participation_ratio"} <= out.keys()
