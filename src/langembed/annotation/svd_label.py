"""Silver-standard STS pair generation via TF-IDF + truncated SVD (LSA) -- fully offline,
no network calls, no human labeler needed. A second --auto-label-method alongside
back-translation (see auto_label.py); not the default."""

from __future__ import annotations

import random
from collections.abc import Sequence

# Bounds TF-IDF+SVD fit cost regardless of corpus size -- gu's ~13.8M-sentence corpus is
# why train_simcse and dedup both needed the same kind of bound this session; LSA needs far
# fewer documents than neural training to capture corpus-level semantic structure, so this
# constant is an order of magnitude smaller than train_simcse.MAX_TRAIN_EXAMPLES.
MAX_FIT_SENTENCES = 200_000


def build_svd_sts_pairs(
    sentences: Sequence[str],
    n: int,
    n_components: int = 100,
    seed: int = 42,
    max_fit_sentences: int = MAX_FIT_SENTENCES,
) -> list[tuple[str, str, float]]:
    """Silver STS pairs scored by real cosine similarity in TF-IDF+SVD (LSA) space --
    unlike back-translation's three fixed-score tiers, every pair gets a computed score.

    If `sentences` is larger than `max_fit_sentences`, a uniform random subsample is fit
    instead of the full corpus (bounded memory/time regardless of corpus size); pairs are
    then drawn only from that fit set. Returns [] if fewer than 2 sentences are available
    to pair.
    """
    if len(sentences) < 2:
        return []

    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    rng = random.Random(seed)
    fit_sentences = (
        rng.sample(list(sentences), max_fit_sentences)
        if len(sentences) > max_fit_sentences
        else list(sentences)
    )

    tfidf = TfidfVectorizer().fit_transform(fit_sentences)
    # Clamp n_components to avoid sklearn ValueError on small vocabularies or corpora.
    # TruncatedSVD requires n_components < n_features and n_components + 1 <= n_samples.
    n_features = tfidf.shape[1]
    effective_n_components = min(n_components, n_features - 1, len(fit_sentences) - 1)
    if effective_n_components < 1:
        # Degenerate vocabulary (e.g. empty or a single repeated term across all
        # sentences) -- no meaningful similarity space to build, same "can't produce
        # pairs" contract as the len(sentences) < 2 guard above.
        return []
    svd = TruncatedSVD(n_components=effective_n_components, random_state=seed)
    vectors = svd.fit_transform(tfidf)

    pairs: list[tuple[str, str, float]] = []
    for _ in range(n):
        i, j = rng.sample(range(len(fit_sentences)), 2)
        similarity = cosine_similarity(vectors[i : i + 1], vectors[j : j + 1])[0, 0]
        score = max(0.0, min(1.0, float(similarity))) * 5.0
        pairs.append((fit_sentences[i], fit_sentences[j], score))
    return pairs
