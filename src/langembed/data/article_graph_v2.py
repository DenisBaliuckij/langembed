"""Per-article semantic graph v2 (2026-10-11): one node per normalized unit (see
langembed.data.unit_normalize), sized by frequency, fewer and more coherent clusters.

Differences from semantic_graph_lib.build_graph (v1, still used by the corpus graph):
- nodes are pre-merged units filtered by frequency (`select_nodes`), so the graph is smaller;
- embeddings are mean-centred per article before cosine similarity: the SimCSE vectors are
  anisotropic (any two phrases score ~0.6+), which blurs neighbourhoods and clusters;
- edges join each node to its k nearest neighbours by a mix of embedding similarity and
  co-occurrence in nearby sentences (normalized PMI over 3-sentence windows), so phrases used
  together in one passage are tied even when their embeddings differ;
- clustering is Louvain with a fixed seed (v1's greedy modularity broke ties differently from
  run to run), with the resolution lowered until there are at most `max_clusters` clusters;
- communities below `min_cluster_size` are attached to the most similar cluster (by centroid)
  instead of being lumped into one "Other" group, unless nothing is similar enough;
- optionally (`core_min_sim`, off by default) a node stays in its cluster only if it is close
  to the cluster centroid; looser nodes go to an "Other (weakly related)" group -- with weak
  embeddings this also evicts central concepts, so it is not used for the colleague graphs;
- a cluster is labeled by its most representative phrase (frequent and close to the cluster
  centroid) rather than three TF-IDF words joined with "/", and carries keywords and top
  phrases for the viewer.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import networkx as nx
import numpy as np


def _is_rare_unknown(u: dict, lexicon: frozenset[str] | set[str], min_count: int) -> bool:
    """A lower-case single word missing from the English word list and used fewer than
    `min_count` times: a word from a cited example in another language ("emakume", "liburu"),
    while the article's own frequent notation ("nprel", "rcs") passes."""
    t = u["text"]
    return (
        bool(lexicon)
        and " " not in t
        and t.isalpha()
        and t.islower()
        and t not in lexicon
        and u["key"] not in lexicon
        and u["count"] < min_count
    )


def select_nodes(
    units: list[dict],
    min_count: int = 2,
    max_nodes: int = 400,
    lexicon: frozenset[str] | set[str] = frozenset(),
    unknown_min_count: int = 6,
) -> list[dict]:
    """Units occurring at least `min_count` times, at most `max_nodes` (most frequent first,
    ties by first occurrence); returned in reading order."""
    kept = [
        u
        for u in units
        if u["count"] >= min_count and not _is_rare_unknown(u, lexicon, unknown_min_count)
    ]
    kept.sort(key=lambda u: (-u["count"], u["first_sent_idx"]))
    kept = kept[:max_nodes]
    kept.sort(key=lambda u: (u["first_sent_idx"], u["key"]))
    return kept


def _normalize(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return emb / norms


def cooccurrence_npmi(nodes: list[dict], window: int = 3) -> np.ndarray:
    """Normalized PMI of two units appearing in the same block of `window` sentences
    (negative values clipped to 0)."""
    blocks = [{s // window for s in u["sent_ids"]} for u in nodes]
    n_blocks = max((max(b) for b in blocks if b), default=0) + 1
    n = len(nodes)
    m = np.zeros((n, n_blocks))
    for i, b in enumerate(blocks):
        m[i, list(b)] = 1.0
    joint = (m @ m.T) / n_blocks
    p = m.sum(axis=1) / n_blocks
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(joint / np.outer(p, p))
        npmi = pmi / -np.log(joint)
    npmi[~np.isfinite(npmi)] = 0.0
    np.fill_diagonal(npmi, 0.0)
    return np.clip(npmi, 0.0, 1.0)


def knn_edges_from_sim(sim: np.ndarray, k: int) -> dict[tuple[int, int], float]:
    """Union of every node's top-k neighbours by `sim`, deduplicated."""
    sim = sim.copy()
    np.fill_diagonal(sim, -np.inf)
    edges: dict[tuple[int, int], float] = {}
    for i in range(sim.shape[0]):
        for j in np.argsort(-sim[i])[:k]:
            j = int(j)
            key = (i, j) if i < j else (j, i)
            edges[key] = max(edges.get(key, -1.0), float(sim[i, j]))
    return edges


def _communities(G: nx.Graph, resolution: float, seed: int) -> list[set[int]]:
    comms = nx.algorithms.community.louvain_communities(
        G, weight="weight", resolution=resolution, seed=seed
    )
    return sorted(comms, key=lambda c: (-len(c), min(c)))


def cluster_nodes(
    normed: np.ndarray,
    edges: dict[tuple[int, int], float],
    min_cluster_size: int,
    max_clusters: int,
    resolution: float = 1.0,
    seed: int = 42,
    attach_min_sim: float = 0.3,
) -> tuple[dict[int, int], float, int]:
    """Returns (cluster_of, resolution_used, n_raw_communities). Cluster ids are 0..n-1 by
    size; id -1 = not similar enough to any cluster (shown as "Other")."""
    n = normed.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (i, j), w in edges.items():
        G.add_edge(i, j, weight=max(w, 0.0))

    res = resolution
    while True:
        comms = _communities(G, res, seed)
        big = [c for c in comms if len(c) >= min_cluster_size]
        if len(big) <= max_clusters or res <= 0.2:
            break
        res = round(res - 0.1, 2)

    cluster_of: dict[int, int] = {}
    for cid, comm in enumerate(big):
        for node in comm:
            cluster_of[node] = cid
    if big:
        centroids = _normalize(np.array([normed[sorted(c)].mean(axis=0) for c in big]))
        for comm in comms:
            if len(comm) >= min_cluster_size:
                continue
            for node in comm:
                sims = centroids @ normed[node]
                best = int(np.argmax(sims))
                cluster_of[node] = best if sims[best] >= attach_min_sim else -1
    else:
        cluster_of = {i: -1 for i in range(n)}

    return _renumber(cluster_of), res, len(comms)


def prune_to_core(
    normed: np.ndarray, cluster_of: dict[int, int], core_min_sim: float, min_core_size: int = 4
) -> dict[int, int]:
    """Move nodes whose cosine to their cluster's centroid is below `core_min_sim` to -1 (and
    whole clusters left with fewer than `min_core_size` nodes), then renumber by size."""
    members: dict[int, list[int]] = {}
    for i, c in cluster_of.items():
        if c >= 0:
            members.setdefault(c, []).append(i)
    pruned = dict(cluster_of)
    for idx in members.values():
        centroid = normed[idx].mean(axis=0)
        centroid /= np.linalg.norm(centroid) or 1e-9
        for i in idx:
            if float(normed[i] @ centroid) < core_min_sim:
                pruned[i] = -1
    sizes = Counter(c for c in pruned.values() if c >= 0)
    for i, c in pruned.items():
        if c >= 0 and sizes[c] < min_core_size:  # too little left to be a topic
            pruned[i] = -1
    return _renumber(pruned)


def _renumber(cluster_of: dict[int, int]) -> dict[int, int]:
    # renumber by final size so the viewer's colour ranks follow prominence
    sizes = Counter(c for c in cluster_of.values() if c >= 0)
    order = {
        old: new
        for new, (old, _) in enumerate(sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])))
    }
    return {i: order.get(c, -1) for i, c in cluster_of.items()}


_WORD_RE = re.compile(r"[a-z][a-z\-']+")


def describe_clusters(
    nodes: list[dict],
    normed: np.ndarray,
    cluster_of: dict[int, int],
    n_keywords: int = 6,
    n_phrases: int = 8,
) -> dict[int, dict]:
    """label = the phrase maximizing cos(node, cluster centroid) * log(1 + count), preferring
    multi-word phrases; keywords = class-TF-IDF over the lemma keys' words."""
    members: dict[int, list[int]] = {}
    for i, c in cluster_of.items():
        members.setdefault(c, []).append(i)

    word_tf: dict[int, Counter[str]] = {}
    for c, idx in members.items():
        tf: Counter[str] = Counter()
        for i in idx:
            for w in set(_WORD_RE.findall(nodes[i]["key"])):
                tf[w] += nodes[i]["count"]
        word_tf[c] = tf
    df: Counter[str] = Counter()
    for tf in word_tf.values():
        df.update(tf.keys())
    n_clusters = max(len(members), 1)

    out: dict[int, dict] = {}
    for c, idx in members.items():
        weights = np.array([nodes[i]["count"] for i in idx], dtype=np.float64)
        centroid = (normed[idx] * weights[:, None]).sum(axis=0)
        centroid /= np.linalg.norm(centroid) or 1e-9
        scored = []
        for i in idx:
            multi = 1.4 if " " in nodes[i]["key"] else 1.0
            cos = max(float(normed[i] @ centroid), 0.0)
            scored.append((cos * cos * math.log1p(nodes[i]["count"]) * multi, i))
        scored.sort(reverse=True)
        # a multi-word phrase names a topic better than a lone frequent word: take the best
        # multi-word phrase when it scores at least half as well as the overall best
        best = scored[0]
        multi_best = next((s for s in scored if " " in nodes[s[1]]["key"]), None)
        if multi_best is not None and multi_best[0] >= 0.5 * best[0]:
            best = multi_best
        rep = nodes[best[1]]["text"]
        total = sum(word_tf[c].values()) or 1
        kw = sorted(
            word_tf[c].items(),
            key=lambda kv: -(kv[1] / total) * math.log(1 + n_clusters / df[kv[0]]),
        )
        label_words = set(_WORD_RE.findall(nodes[best[1]]["key"]))
        keywords = [w for w, _ in kw if w not in label_words][:n_keywords]
        top = sorted(idx, key=lambda i: (-nodes[i]["count"], nodes[i]["first_sent_idx"]))[
            :n_phrases
        ]
        out[c] = {
            "label": "Other (weakly related)" if c == -1 else rep[:1].upper() + rep[1:],
            "keywords": keywords,
            "top_phrases": [nodes[i]["text"] for i in top],
            "size": len(idx),
            "occurrences": int(weights.sum()),
        }
    return out


def build_article_graph(
    units: list[dict],
    k: int = 6,
    alpha: float = 1.0,
    core_min_sim: float = -1.0,
    center: bool = True,
    min_count: int = 2,
    max_nodes: int = 400,
    min_cluster_size: int = 6,
    max_clusters: int = 12,
    seed: int = 42,
    lexicon: frozenset[str] | set[str] = frozenset(),
    embedding_model: str = "",
) -> dict:
    """units: merged units of ONE article, each with an 'embedding'. Returns the viewer's
    article record (nodes, edges, clusters, meta)."""
    nodes = select_nodes(units, min_count=min_count, max_nodes=max_nodes, lexicon=lexicon)
    emb = np.array([u["embedding"] for u in nodes], dtype=np.float64)
    normed = _normalize(emb - emb.mean(axis=0) if center else emb)
    sim = alpha * (normed @ normed.T) + (1 - alpha) * cooccurrence_npmi(nodes)
    edges = knn_edges_from_sim(sim, k)
    cluster_of, res, n_raw = cluster_nodes(normed, edges, min_cluster_size, max_clusters, seed=seed)
    cluster_of = prune_to_core(normed, cluster_of, core_min_sim)
    clusters = describe_clusters(nodes, normed, cluster_of)

    degree: Counter[int] = Counter()
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1

    nodes_out = []
    for i, u in enumerate(nodes):
        nodes_out.append(
            {
                "id": i,
                "label": u["text"],
                "text": u["text"],
                "key": u["key"],
                "count": u["count"],
                "variants": u["variants"],
                "contexts": u["contexts"],
                "cluster": cluster_of[i],
                "degree": degree[i],
                "meta": f"{u['count']} occurrences in {u['n_sentences']} sentences",
            }
        )
    return {
        "nodes": nodes_out,
        "edges": [{"source": i, "target": j, "weight": round(w, 4)} for (i, j), w in edges.items()],
        "clusters": [
            {"id": c, **d} for c, d in sorted(clusters.items(), key=lambda kv: (kv[0] == -1, kv[0]))
        ],
        "meta": {
            "version": "v2 (normalized, merged units)",
            "embedding_model": embedding_model,
            "k": k,
            "edge_weight": f"{alpha:g} * embedding cosine + {1 - alpha:g} * co-occurrence NPMI "
            "(3-sentence windows)",
            "min_count": min_count,
            "num_units_before_selection": len(units),
            "num_nodes": len(nodes_out),
            "num_edges": len(edges),
            "num_clusters": sum(1 for c in clusters if c != -1),
            "num_raw_communities": n_raw,
            "louvain_resolution": res,
            "core_min_sim": core_min_sim,
            "num_other": sum(1 for c in cluster_of.values() if c == -1),
            "embedding_dim": int(normed.shape[1]),
            "embeddings_centered": center,
        },
    }
