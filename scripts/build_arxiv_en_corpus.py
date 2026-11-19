"""Build an English raw-text corpus from converted arxiv LaTeX sources.

Reads every *.tex file under a directory (recursively), strips LaTeX markup
via langembed.data.extract_text.tex_to_text (pure regex, no AI), splits into
sentences via split_sentences, and writes one sentence per line to the
output file - ready for full_pipeline's source_mode=existing_text path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from langembed.data.extract_text import split_sentences, tex_to_text


def build_corpus(tex_dir: Path, out_path: Path) -> tuple[int, int]:
    tex_files = sorted(tex_dir.rglob("*.tex"))
    print(f"found {len(tex_files)} .tex files under {tex_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_sentences = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, tex_path in enumerate(tex_files, 1):
            try:
                tex = tex_path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                print(f"skip {tex_path}: {exc}")
                continue
            sentences = split_sentences(tex_to_text(tex))
            for s in sentences:
                f.write(s + "\n")
            total_sentences += len(sentences)
            if i % 50 == 0 or i == len(tex_files):
                print(f"[{i}/{len(tex_files)}] {total_sentences} sentences so far")

    return len(tex_files), total_sentences


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex-dir", required=True, help="Directory containing *.tex files (recursive)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n_files, n_sentences = build_corpus(Path(args.tex_dir), Path(args.out))
    print(f"done: {n_files} files, {n_sentences} sentences -> {args.out}")


if __name__ == "__main__":
    main()
