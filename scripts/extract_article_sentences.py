"""Stage 1 (host/CPU, no ML deps besides plain-text parsing): extract
sentences for N randomly-sampled arxiv .tex articles, for per-article
semantic graphs. Writes the sampled article-id list to a manifest so
downstream steps (and anything copying source files back out) share a single
source of truth for which articles were used, instead of re-deriving the
sample from the seed independently."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from langembed.data.extract_text import split_sentences, tex_to_text  # noqa: E402


def extract_article_sentences(
    tex_dir: Path,
    out_path: Path,
    manifest_path: Path,
    n_articles: int,
    seed: int,
    max_sentences_per_article: int,
    min_sentence_chars: int,
) -> int:
    tex_files = sorted(tex_dir.rglob("*.tex"))
    random.seed(seed)
    sample = random.sample(tex_files, n_articles)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for tex_path in sample:
            try:
                tex = tex_path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                print(f"skip {tex_path}: {exc}")
                continue
            text = tex_to_text(tex)
            sentences = [
                s.strip() for s in split_sentences(text) if len(s.strip()) >= min_sentence_chars
            ]
            sentences = sentences[:max_sentences_per_article]
            for i, s in enumerate(sentences):
                f.write(
                    json.dumps(
                        {"article_id": tex_path.stem, "sent_idx": i, "text": s},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_written += 1
            print(f"{tex_path.stem}: {len(sentences)} sentences")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "n_articles": n_articles,
                "tex_dir": str(tex_dir),
                "article_ids": [p.stem for p in sample],
                "tex_paths": [str(p) for p in sample],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return n_written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex-dir", default="/opt/latex/arxiv")
    ap.add_argument("--out", default="data/article_sentences_en.jsonl")
    ap.add_argument("--manifest-out", default="data/article_sample_manifest.json")
    ap.add_argument("--n-articles", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-sentences-per-article", type=int, default=120)
    ap.add_argument("--min-sentence-chars", type=int, default=25)
    args = ap.parse_args()

    n = extract_article_sentences(
        tex_dir=Path(args.tex_dir),
        out_path=Path(args.out),
        manifest_path=Path(args.manifest_out),
        n_articles=args.n_articles,
        seed=args.seed,
        max_sentences_per_article=args.max_sentences_per_article,
        min_sentence_chars=args.min_sentence_chars,
    )
    print(f"wrote {n} sentences across {args.n_articles} articles -> {args.out}")


if __name__ == "__main__":
    main()
