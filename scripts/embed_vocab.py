"""Encode every unique word in the corpus, producing a compact vocabulary-level
embedding table -- one vector per unique word, not per sentence.

Vocabulary size is orders of magnitude smaller than sentence count (tens/hundreds of
thousands of unique words vs. tens of millions of sentences for the larger corpora
this project processes), so this artifact is small by construction, unlike a
full per-sentence dump (see embed_corpus.py's default --limit).

Two independent switches, chosen up front like run_pipeline.py's
--auto-label-method:

--method direct (default): "direct token encoding" -- forward-pass each unique word
    through the already-trained SimCSE sentence encoder, i.e. distillation (see e.g.
    Model2Vec / "Static Word Embeddings for Sentence Semantic Representation"). Vectors
    live in the same embedding space as the sentence embeddings.
--method cbow: train an independent Continuous Bag-of-Words model from scratch on the
    corpus (langembed.wordembed.train_cbow) -- the classic word2vec formulation,
    unrelated to the sentence encoder's embedding space.

--format {jsonl,binary} controls the output encoding, same as embed_corpus.py.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from langembed import embed_io
from langembed.config import load_config
from langembed.disk_guard import DiskSpaceError

_PUNCT_EDGES = re.compile(r"^\W+|\W+$", re.UNICODE)


def _strip_punct(token: str) -> str:
    """Strip leading/trailing non-word characters (quotes, brackets, sentence
    punctuation) glued onto a token by whitespace tokenization -- e.g. a word
    quoted in the source text splits as `"word"`, not `word`. Internal punctuation
    (apostrophes, hyphens) is left alone since it can be part of the word itself."""
    return _PUNCT_EDGES.sub("", token)


def _is_word(token: str) -> bool:
    """True if every whitespace-separated part of `token` (1 part for a unigram, 2
    for a bigram) contains at least one alphabetic character. Filters out bare
    punctuation, digits, and symbol-only tokens -- naive whitespace splitting alone
    previously let e.g. "!" through as a "word", one of many noise tokens that
    inflated a single language's vocabulary to 1M+ entries (see the "as" language
    run: 1,139,840 tokens / 6.4GB from an unfiltered corpus scan)."""
    return all(any(ch.isalpha() for ch in part) for part in token.split(" "))


def extract_vocab(
    sentences_path: str,
    lang: str,
    spacy_model: str | None = None,
    min_frequency: int = 5,
    include_bigrams: bool = True,
    max_vocab_size: int | None = 100_000,
) -> list[str]:
    """Frequency-filtered words (and, if enabled, adjacent word pairs) across the
    corpus, sorted for a deterministic output order.

    Without a minimum-frequency threshold, every whitespace-delimited token in the
    corpus -- including one-off typos, foreign-script fragments, and bare
    punctuation -- becomes a "word", which is what let a single language's
    vocabulary balloon to 1.14M entries and 6.4GB. `min_frequency` (word2vec's own
    min_count default is 5) drops that long tail of noise; `_is_word` additionally
    drops tokens that contain no alphabetic character at all. `max_vocab_size`, if
    set, keeps only the most frequent tokens after filtering as a hard safety cap.
    """
    from collections import Counter

    from langembed.preprocess import normalize

    counts: Counter[str] = Counter()
    with open(sentences_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_tokens = normalize(line, lang=lang, spacy_model=spacy_model).split(" ")
            tokens = [stripped for t in raw_tokens if t and (stripped := _strip_punct(t))]
            counts.update(tokens)
            if include_bigrams:
                counts.update(f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False))

    vocab = [tok for tok, count in counts.items() if count >= min_frequency and _is_word(tok)]
    if max_vocab_size is not None and len(vocab) > max_vocab_size:
        vocab.sort(key=lambda t: (-counts[t], t))
        vocab = vocab[:max_vocab_size]
    return sorted(vocab)


def embed_vocab(
    config_path: str,
    out_path: str,
    lang: str,
    spacy_model: str | None = None,
    min_free_gb: float = 5.0,
    fmt: str = "jsonl",
    method: str = "direct",
    min_frequency: int | None = None,
    include_bigrams: bool | None = None,
    max_vocab_size: int | None = -1,
) -> int:
    """`min_frequency`/`include_bigrams`/`max_vocab_size` override the `vocab:`
    section of the config when given explicitly (CLI re-runs); otherwise they fall
    back to the config's `vocab:` section, then to extract_vocab's own defaults.
    `max_vocab_size=-1` (the sentinel default, distinct from `None`) means "use the
    config/extract_vocab default" since `None` itself is a valid override meaning
    "no cap"."""
    cfg = load_config(config_path)
    out = Path(out_path)
    vocab_cfg = cfg.get("vocab", {})

    if method == "cbow":
        from langembed.wordembed.train_cbow import train_cbow

        cbow_section = dict(cfg.get("cbow", {}))
        cbow_section.setdefault("sentences_path", cfg["simcse"]["sentences_path"])
        words, vectors = train_cbow(
            {
                "cbow": cbow_section,
                "language": lang,
                "spacy_model": spacy_model,
                "seed": cfg.get("seed", 42),
            }
        )
        if fmt == "binary":
            return embed_io.write_binary_array(words, vectors, out, min_free_gb)
        return embed_io.write_jsonl_rows(words, vectors, out, "word", min_free_gb)

    import datasets  # noqa: F401
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    import torch  # noqa: F401
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg["simcse"]["out_dir"])
    words = extract_vocab(
        cfg["simcse"]["sentences_path"],
        lang,
        spacy_model,
        min_frequency=min_frequency
        if min_frequency is not None
        else vocab_cfg.get("min_frequency", 5),
        include_bigrams=(
            include_bigrams
            if include_bigrams is not None
            else vocab_cfg.get("include_bigrams", True)
        ),
        max_vocab_size=(
            max_vocab_size if max_vocab_size != -1 else vocab_cfg.get("max_vocab_size", 100_000)
        ),
    )
    if fmt == "binary":
        return embed_io.encode_and_write_binary(model, words, out, min_free_gb)
    return embed_io.encode_and_write_jsonl(model, words, out, "word", min_free_gb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lang", required=True, help="language code, for text normalization")
    ap.add_argument("--out", default="artifacts/vocab/vocab_embeddings.jsonl")
    ap.add_argument("--spacy-model", default=None)
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    ap.add_argument(
        "--method",
        choices=["direct", "cbow"],
        default="direct",
        help=(
            "'direct' (default): forward-pass each unique word through the trained "
            "SimCSE encoder. 'cbow': train an independent Continuous Bag-of-Words "
            "model on the corpus instead (langembed.wordembed.train_cbow)."
        ),
    )
    ap.add_argument(
        "--format",
        choices=["jsonl", "binary"],
        default="jsonl",
        help=(
            '\'jsonl\': one {"word": ..., "embedding": [...]} row per line. '
            "'binary': float16 .npy + .meta.json."
        ),
    )
    ap.add_argument(
        "--min-frequency",
        type=int,
        default=None,
        help="Override the config's vocab.min_frequency (default: 5 if unset anywhere).",
    )
    ap.add_argument(
        "--no-bigrams",
        action="store_true",
        help="Disable adjacent-word-pair extraction (unigrams only).",
    )
    ap.add_argument(
        "--max-vocab-size",
        type=int,
        default=-1,
        help=(
            "Override the config's vocab.max_vocab_size (default: use config, or "
            "100000 if unset there). Pass 0 to disable the cap."
        ),
    )
    args = ap.parse_args()
    try:
        n = embed_vocab(
            args.config,
            args.out,
            args.lang,
            spacy_model=args.spacy_model,
            min_free_gb=args.min_free_gb,
            fmt=args.format,
            method=args.method,
            min_frequency=args.min_frequency,
            include_bigrams=False if args.no_bigrams else None,
            max_vocab_size=(None if args.max_vocab_size == 0 else args.max_vocab_size),
        )
    except DiskSpaceError as e:
        print(f"aborted: {e}")
        raise SystemExit(1) from e
    print(f"embedded unique words: {n}")


if __name__ == "__main__":
    main()
