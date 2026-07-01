"""Extract sentences from data/raw/voina-i-mir.pdf into the ru raw-corpus format (ru track)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_SENTENCE_END_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ«\"'(])")
_JUNK_RE = re.compile(r"^\s*\d+\s*$")
_MIN_LEN = 20


def extract_pdf_text(pdf_path: str | Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def split_sentences(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and not _JUNK_RE.match(ln)]

    joined = ""
    for ln in lines:
        if joined.endswith("-"):
            joined = joined[:-1] + ln
        else:
            joined = joined + (" " if joined else "") + ln
    joined = re.sub(r"\s{2,}", " ", joined).strip()

    sentences = []
    for s in _SENTENCE_END_RE.split(joined):
        s = s.strip()
        if len(s) < _MIN_LEN:
            continue
        sentences.append(s)
    return sentences


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    text = extract_pdf_text(args.pdf)
    sentences = split_sentences(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")
    print(f"sentences: {len(sentences)}")


if __name__ == "__main__":
    main()
