"""Shared k-NN similarity graph + TF-IDF cluster labeling, used by the
per-article (semantic-unit granularity) graph builder. Ported from the
original semantic_graph/graphlib_semgraph.py prototype, unchanged in its
core graph/clustering logic."""

from __future__ import annotations

import re
from collections import Counter

import networkx as nx
import numpy as np

STOPWORDS = set(
    """
    the a an and or of to in on for with is are was were be been being this that
    these those as by at from into over under between among it its it's we our
    us they their them he she his her you your i not no but if then than so such
    can could will would should may might must do does did done have has had
    which who whom what when where why how all each both more most other some
    only own same too very s t can will just don should now also using use used
    based via abstract introduction et al arxiv preprint paper figure table
    section results method approach show shown work paper propose present
    university institute department college school laboratory sciences
    technology email corresponding author authors china usa institution
    however every dated given let consider since thus hence therefore
    moreover furthermore yet though whose upon within without respectively
    following considered provide provides denote denoted general specific
    particular case cases new well known recent recently existing given
    january february march april june july august september october
    november december center centre
    """.split()
)


def word_bag(text: str) -> set[str]:
    return {w for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower()) if w not in STOPWORDS}


def cluster_label(cluster_texts: list[str], all_texts: list[str], top_n: int = 3) -> str:
    """TF-IDF-ish: score by how much more often a word appears in this
    cluster's texts than across the whole (corpus or article) population, so
    shared boilerplate doesn't win every cluster's label."""
    doc_freq_all: Counter[str] = Counter()
    for t in all_texts:
        doc_freq_all.update(word_bag(t))
    doc_freq_cluster: Counter[str] = Counter()
    for t in cluster_texts:
        doc_freq_cluster.update(word_bag(t))

    n_cluster = len(cluster_texts)
    n_all = len(all_texts)
    scored = []
    for w, cf in doc_freq_cluster.items():
        if cf < 2:
            continue
        cluster_rate = cf / n_cluster
        global_rate = doc_freq_all[w] / n_all
        score = cluster_rate * (cluster_rate / global_rate)
        scored.append((score, w))
    scored.sort(reverse=True)
    words = [w for _, w in scored[:top_n]]
    return " / ".join(w.capitalize() for w in words) if words else "Misc"


def make_label(text: str, n: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


# Matches a leading PDF->LaTeX running-header/masthead (journal name,
# "PREPRINT VERSION", an "ACCEPTED <month>, <year> <page>" line) that some
# converted PDFs glue onto the front of the extracted text, ahead of the
# real title. Deliberately narrow (explicit masthead vocabulary only) so it
# never eats a genuine all-caps paper title, which is a common journal style
# in its own right and must not be treated as a header.
_MASTHEAD_RE = re.compile(
    r"^(?:IEEE\s+[A-Z][A-Z ,.\-]{3,80}?(?:LETTERS|TRANSACTIONS|JOURNAL|MAGAZINE)\.?\s*)?"
    r"(?:PRE-?PRINT\s+VERSION\.?\s*)?"
    r"(?:(?:SUBMITTED|ACCEPTED)(?:\s+(?:TO|FOR|ON))?\s+[A-Za-z]+,?\s+\d{4}\s*\d{0,3}\s*)?",
    re.IGNORECASE,
)


def pick_title(sentences: list[str], n: int = 90) -> str:
    """Best-effort article title for a UI label. Strips a leading journal
    masthead (spanning into the second sentence, since header text and the
    real title sometimes land in the same split-sentence unit) when one is
    recognized; otherwise falls back to the first sentence as-is, which is
    right for the common case (including genuinely all-caps titles)."""
    if not sentences:
        return "Untitled"
    combined = sentences[0] if len(sentences) == 1 else sentences[0] + " " + sentences[1]
    m = _MASTHEAD_RE.match(combined)
    candidate = combined[m.end() :].strip() if m and m.end() > 0 else sentences[0]
    return make_label(candidate, n=n)


def knn_edges(embeddings: np.ndarray, k: int) -> dict[tuple[int, int], float]:
    """embeddings: (n, dim) array. Returns {(i, j): cosine_similarity} for
    the union of each node's top-k neighbors, deduped, max similarity kept."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normed = embeddings / norms
    sim = normed @ normed.T
    np.fill_diagonal(sim, -1.0)

    edges: dict[tuple[int, int], float] = {}
    n = embeddings.shape[0]
    for i in range(n):
        top = np.argsort(-sim[i])[:k]
        for j in top:
            j = int(j)
            if j == i:
                continue
            key = (i, j) if i < j else (j, i)
            w = float(sim[i, j])
            if key not in edges or w > edges[key]:
                edges[key] = w
    return edges


def build_graph(
    records: list[dict], texts: list[str], k: int, min_cluster_size: int
) -> tuple[dict[tuple[int, int], float], dict[int, int], dict[int, int], dict[int, str], int]:
    """records: list of dicts with an 'embedding' key (aligned with texts).
    Returns (edges dict, degree dict, cluster_of dict, cluster_labels dict,
    num_raw_communities)."""
    embeddings = np.array([r["embedding"] for r in records], dtype=np.float64)
    edges = knn_edges(embeddings, k)

    n = len(records)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (i, j), w in edges.items():
        G.add_edge(i, j, weight=w)

    communities = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
    communities = sorted(communities, key=len, reverse=True)

    big = [c for c in communities if len(c) >= min_cluster_size]
    small = [c for c in communities if len(c) < min_cluster_size]

    cluster_of: dict[int, int] = {}
    cluster_labels: dict[int, str] = {}
    for cid, comm in enumerate(big):
        for node in comm:
            cluster_of[node] = cid
        cluster_labels[cid] = cluster_label([texts[i] for i in comm], texts)

    if small:
        other_id = len(big)
        for comm in small:
            for node in comm:
                cluster_of[node] = other_id
        cluster_labels[other_id] = "Other"

    degree = dict(G.degree())
    return edges, degree, cluster_of, cluster_labels, len(communities)
