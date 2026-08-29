"""CBOW as a 4th, directly comparable sentence-embedding method: train an
independent Continuous Bag-of-Words model (langembed.wordembed.train_cbow) on the
corpus, then represent each sentence by mean-pooling its words' CBOW vectors -- the
standard "sentence = average of its word embeddings" baseline (see e.g. "Static Word
Embeddings for Sentence Semantic Representation", arXiv:2506.04624). Produces
output/<lang>/embeddings_cbow.jsonl in the same {"text", "embedding"} shape as
Branches A/B/C, over the *same* corpus prefix embed_corpus.py uses, so evaluate.py
can score all four side by side on identical sentences.

Usage:
    python scripts/embed_branch_cbow.py --lang mr
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from langembed import embed_io
from langembed.disk_guard import DiskSpaceError
from langembed.wordembed.train_cbow import train_cbow

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBED_SAMPLE_SIZE = 200


def pool_sentence_vector(
    words: list[str], word_to_vec: dict[str, list[float]], dim: int
) -> list[float]:
    """Mean-pool the CBOW vectors of `words` that are in-vocabulary; an
    all-out-of-vocabulary sentence gets a zero vector rather than raising."""
    vecs = [word_to_vec[w] for w in words if w in word_to_vec]
    if not vecs:
        return [0.0] * dim
    return [sum(col) / len(vecs) for col in zip(*vecs, strict=True)]


def embed_branch_cbow(
    lang: str,
    out_path: str,
    cbow_cfg: dict[str, Any] | None = None,
    embed_sample_size: int = DEFAULT_EMBED_SAMPLE_SIZE,
    min_free_gb: float = 5.0,
    seed: int = 42,
) -> int:
    from langembed.preprocess import normalize

    corpus_path = REPO_ROOT / f"data/corpus_{lang}.txt"
    section = dict(cbow_cfg or {})
    section.setdefault("sentences_path", str(corpus_path))
    words, vectors = train_cbow({"cbow": section, "language": lang, "seed": seed})
    if not words:
        raise RuntimeError(f"CBOW training produced an empty vocabulary for lang={lang!r}")
    dim = len(vectors[0])
    word_to_vec = dict(zip(words, vectors, strict=True))

    with corpus_path.open(encoding="utf-8") as f:
        sentences = [ln.strip() for ln in f if ln.strip()][:embed_sample_size]

    pooled = [
        pool_sentence_vector(normalize(s, lang=lang).split(" "), word_to_vec, dim)
        for s in sentences
    ]

    out = Path(out_path)
    return embed_io.write_jsonl_rows(sentences, pooled, out, "text", min_free_gb)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out", default=None, help="default: output/<lang>/embeddings_cbow.jsonl")
    ap.add_argument("--embed-sample-size", type=int, default=DEFAULT_EMBED_SAMPLE_SIZE)
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    args = ap.parse_args()
    out_path = args.out or str(REPO_ROOT / f"output/{args.lang}/embeddings_cbow.jsonl")
    try:
        n = embed_branch_cbow(
            args.lang,
            out_path,
            embed_sample_size=args.embed_sample_size,
            min_free_gb=args.min_free_gb,
        )
    except DiskSpaceError as e:
        print(f"aborted: {e}")
        raise SystemExit(1) from e
    print(f"embedded sentences (cbow, mean-pooled): {n}")


if __name__ == "__main__":
    main()
