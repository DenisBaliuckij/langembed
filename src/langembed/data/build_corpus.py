"""Build a clean monolingual corpus with a hard guard against test leakage (Phase 1)."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from langembed.config import load_config
from langembed.data.dedup import dedup
from langembed.preprocess import normalize


def _h(s: str, lang: str = "gu", spacy_model: str | None = None) -> str:
    return hashlib.sha1(normalize(s, lang, spacy_model).encode("utf-8")).hexdigest()


def load_test_hashes(
    test_path: str | Path, lang: str = "gu", spacy_model: str | None = None
) -> set[str]:
    """Hash every sentence of the STS test set so we can detect leakage."""
    hashes: set[str] = set()
    p = Path(test_path)
    if not p.exists():
        return hashes
    for line in p.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        for key in ("sentence_a", "sentence_b"):
            if key in r:
                hashes.add(_h(r[key], lang, spacy_model))
    return hashes


def build_corpus(
    raw_paths: Sequence[str],
    out_path: str | Path,
    test_hashes: set[str],
    lang: str = "gu",
    spacy_model: str | None = None,
) -> int:
    """Normalize -> dedup -> guard -> write JSONL-free one-sentence-per-line corpus."""
    docs: list[str] = []
    for rp in raw_paths:
        # errors="replace": some source archives are truncated mid-download (confirmed
        # via matching .crdownload files sitting alongside them), which can cut off a
        # raw-text file mid-UTF-8-character. Strict decoding crashes the whole corpus
        # build on that single broken tail; replacing the few undecodable bytes lets the
        # millions of valid preceding lines through instead.
        for line in Path(rp).open(encoding="utf-8", errors="replace"):
            t = normalize(line, lang, spacy_model)
            if t:
                docs.append(t)
    docs = dedup(docs)
    leaked: Iterable[str] = (d for d in docs if _h(d, lang, spacy_model) in test_hashes)
    n_leaked = sum(1 for _ in leaked)
    if n_leaked:
        raise RuntimeError(f"Test leakage: {n_leaked} corpus lines overlap the STS test set")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(d + "\n")
    return len(docs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    spacy_model = cfg.get("spacy_model")
    th = load_test_hashes(cfg["data"]["test_path"], cfg["language"], spacy_model)
    n = build_corpus(
        cfg["data"]["raw_paths"], cfg["data"]["out_path"], th, cfg["language"], spacy_model
    )
    print(f"corpus lines: {n}")


if __name__ == "__main__":
    main()
