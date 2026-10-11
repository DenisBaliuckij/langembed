"""Intrinsic-dimension (ID) estimators for embedding point clouds.

The intrinsic dimension of a set of embeddings is the number of degrees of freedom the points
actually use, as opposed to the vector length (e.g. 768). It is a label-free way to compare
embedding variants (branches A/B/C/CBOW, different models, training stages, languages).

Estimators (see docs/intrinsic-dimension.md for the background and how to read them):

- ``schweinhart``: the minimum-spanning-tree length estimator used by
  github.com/whiteroomlz/intrinsic-dimensions-computation. For n points sampled from a
  d-dimensional set, the MST weight sum ``E_alpha(n) = sum(|e|^alpha)`` grows like
  ``n^((d - alpha) / d)``; a log-log fit over nested subsamples gives the slope ``s`` and
  ``d = alpha / (1 - s)``. Here the MST is exact (Prim's algorithm on the full distance
  matrix), so no supercomputer is needed for samples up to ~20k points.
- ``twonn``: Facco et al. 2017, from the ratio of each point's second to first neighbour distance.
- ``mle``: Levina & Bickel 2004 maximum likelihood over k nearest neighbours, averaged as in
  MacKay & Ghahramani's correction, over a range of k.
- ``pca``: linear dimensions -- components explaining 90% / 95% of the variance and the
  participation ratio ``(sum l)^2 / sum l^2``. Not an intrinsic dimension of a curved set, but a
  useful upper reference.
- ``anisotropy``: mean pairwise cosine similarity; near 0 for isotropic embeddings, high when all
  vectors point the same way (then centering changes the geometry a lot).

Distances are Euclidean between L2-normalized vectors (the chord distance, monotone in cosine
distance), optionally after subtracting the mean vector (``prep="centered"``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PREPS = ("unit", "centered")


def prepare(vectors: np.ndarray, prep: str = "unit") -> np.ndarray:
    """``unit``: L2-normalize rows. ``centered``: subtract the mean vector, then L2-normalize."""
    if prep not in PREPS:
        raise ValueError(f"prep must be one of {PREPS}, got {prep!r}")
    x = np.asarray(vectors, dtype=np.float64)
    if prep == "centered":
        x = x - x.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def drop_duplicates(x: np.ndarray, decimals: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Remove rows that coincide (e.g. identical embeddings of different strings), which give
    zero-length edges and zero neighbour distances. Returns (unique rows, kept row indices)."""
    _, keep = np.unique(np.round(x, decimals), axis=0, return_index=True)
    keep = np.sort(keep)
    return x[keep], keep


def pairwise_distances(x: np.ndarray) -> np.ndarray:
    """Dense Euclidean distance matrix (float64) via the Gram matrix."""
    sq = np.einsum("ij,ij->i", x, x)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2)


def mst_edge_weights(dist: np.ndarray) -> np.ndarray:
    """Edge weights of the exact minimum spanning tree of a complete graph (Prim, O(n^2))."""
    n = dist.shape[0]
    if n < 2:
        return np.zeros(0)
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    best = dist[0].copy()
    best[0] = np.inf
    weights = np.empty(n - 1)
    for i in range(n - 1):
        j = int(np.argmin(best))
        weights[i] = best[j]
        in_tree[j] = True
        best[j] = np.inf
        np.minimum(best, np.where(in_tree, np.inf, dist[j]), out=best)
    return weights


@dataclass
class SchweinhartResult:
    alpha: float
    dimension: float
    slope: float
    sizes: list[int] = field(default_factory=list)
    log_e: list[float] = field(default_factory=list)


def schweinhart(
    dist: np.ndarray, sizes: list[int], alphas: tuple[float, ...] = (1.0,), seed: int = 0
) -> list[SchweinhartResult]:
    """MST-length estimator on nested random subsamples of the points of ``dist``.

    ``sizes``: subsample sizes (each <= n); the subsamples are prefixes of one random
    permutation, so they are nested like the 1%..100% batches of the original repository."""
    n = dist.shape[0]
    sizes = sorted({int(s) for s in sizes if 2 <= s <= n})
    if len(sizes) < 3:
        raise ValueError("need at least 3 distinct subsample sizes between 2 and n")
    order = np.random.default_rng(seed).permutation(n)
    weights = [mst_edge_weights(dist[np.ix_(order[:s], order[:s])]) for s in sizes]
    log_n = np.log(np.array(sizes, dtype=np.float64))
    results = []
    for alpha in alphas:
        log_e = np.log(np.array([np.sum(w**alpha) for w in weights]))
        slope = float(np.polyfit(log_n, log_e, 1)[0])
        dim = alpha / (1.0 - slope) if slope < 1.0 else float("inf")
        results.append(SchweinhartResult(alpha, float(dim), slope, sizes, log_e.tolist()))
    return results


def _neighbour_distances(dist: np.ndarray, k: int) -> np.ndarray:
    """Sorted distances to the k nearest other points, shape (n, k)."""
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    part = np.partition(d, k - 1, axis=1)[:, :k]
    return np.sort(part, axis=1)


def twonn(dist: np.ndarray, discard_fraction: float = 0.1) -> float:
    """TwoNN (Facco et al. 2017): fit -log(1 - F(mu)) = d * log(mu) through the origin, with the
    largest ``discard_fraction`` of the ratios mu = r2 / r1 dropped as outliers."""
    r = _neighbour_distances(dist, 2)
    ok = r[:, 0] > 0
    mu = np.sort(r[ok, 1] / r[ok, 0])
    n = mu.size
    keep = int(np.floor(n * (1.0 - discard_fraction)))
    mu = mu[:keep]
    f = np.arange(1, keep + 1) / n
    x = np.log(mu)
    y = -np.log(1.0 - f)
    return float(np.dot(x, y) / np.dot(x, x))


def mle(dist: np.ndarray, k_min: int = 10, k_max: int = 20) -> float:
    """Levina-Bickel MLE with the MacKay-Ghahramani correction (average the inverse local
    estimates over points), averaged over k in [k_min, k_max]."""
    r = _neighbour_distances(dist, k_max)
    r = r[r[:, 0] > 0]
    estimates = []
    for k in range(k_min, k_max + 1):
        logs = np.log(r[:, k - 1][:, None] / r[:, : k - 1])
        inv_local = logs.sum(axis=1) / (k - 1)  # 1 / m_k(x)
        estimates.append(1.0 / inv_local.mean())
    return float(np.mean(estimates))


def pca_dims(x: np.ndarray) -> dict[str, float]:
    """Linear dimensions of the (already prepared) vectors."""
    xc = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(xc, compute_uv=False)
    lam = s**2
    ratio = np.cumsum(lam) / lam.sum()
    return {
        "pca90": int(np.searchsorted(ratio, 0.90) + 1),
        "pca95": int(np.searchsorted(ratio, 0.95) + 1),
        "participation_ratio": float(lam.sum() ** 2 / np.sum(lam**2)),
    }


def anisotropy(vectors: np.ndarray, max_points: int = 5000, seed: int = 0) -> float:
    """Mean pairwise cosine similarity of the raw vectors (on a sample of at most max_points)."""
    x = prepare(vectors, "unit")
    if x.shape[0] > max_points:
        x = x[np.random.default_rng(seed).choice(x.shape[0], max_points, replace=False)]
    sim = x @ x.T
    n = x.shape[0]
    return float((sim.sum() - n) / (n * (n - 1)))


def estimate_all(
    vectors: np.ndarray,
    prep: str = "unit",
    sample_size: int = 10000,
    n_sizes: int = 10,
    alphas: tuple[float, ...] = (0.5, 1.0, 2.0),
    seed: int = 0,
) -> dict:
    """All estimators on one random sample of ``sample_size`` distinct prepared vectors."""
    x = prepare(vectors, prep)
    x, _ = drop_duplicates(x)
    n_unique = x.shape[0]
    rng = np.random.default_rng(seed)
    if n_unique > sample_size:
        x = x[rng.choice(n_unique, sample_size, replace=False)]
    n = x.shape[0]
    dist = pairwise_distances(x)
    sizes = np.unique(np.geomspace(max(n // 10, 50), n, n_sizes).astype(int)).tolist()
    out = {
        "prep": prep,
        "seed": seed,
        "n_points": n,
        "n_unique_available": n_unique,
        "dim_ambient": int(x.shape[1]),
        "twonn": twonn(dist),
        "mle": mle(dist),
    }
    for res in schweinhart(dist, sizes, alphas, seed=seed):
        out[f"schweinhart_a{res.alpha:g}"] = res.dimension
    out.update(pca_dims(x))
    return out
