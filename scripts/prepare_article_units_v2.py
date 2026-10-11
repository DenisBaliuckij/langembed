"""Stages 1 + 1.5 of the v2 per-article semantic graph (2026-10-11), run in langembed-ml:
PDF text (pdftotext output, made on the host -- the image has no poppler) -> text without
page furniture, front/back matter and foreign example lines -> sentences without inline
citations -> normalized phrase units (see langembed.data.unit_normalize) merged per article into
one record per lemma key, with frequency, variants and context sentences.

Outputs:
  <out-dir>/article_sentences_en.jsonl  {article_id, sent_idx, text}
  <out-dir>/article_units_v2.jsonl      merged units (input of embed_article_units_cpu.py)
  <out-dir>/article_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from langembed.data.extract_text import split_sentences
from langembed.data.unit_normalize import (
    cited_names,
    clean_article_text,
    is_foreign_line,
    merge_units,
    normalized_units,
    person_names,
    strip_citations,
)

FURNITURE_RE = re.compile(r"^\s*(\(?\d{1,4}[a-z]?\)?|[ivxlc]+\.?|[a-h]\.)\s*$")


def pdf_lines(txt: Path) -> list[str]:
    """pdftotext lines minus page numbers, example numbers like "(13)" and running headers
    (short lines repeated on four or more pages)."""
    out = txt.read_text(encoding="utf-8")
    out = out.replace("\f", "\n").replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    # control characters (pdftotext keeps e.g. BEL in front of example numbers) would glue
    # onto the next word and hide it from the stop-word checks
    out = re.sub(r"[\x00-\x08\x0b\x0e-\x1f]", "", out)
    # private-use glyphs (symbol fonts: example bullets, logic symbols) render as boxes
    out = re.sub(r"[\ue000-\uf8ff]", " ", out)
    lines = [ln.strip() for ln in out.split("\n")]
    counts = Counter(ln for ln in lines if 0 < len(ln) <= 60)
    headers = {ln for ln, c in counts.items() if c >= 4 and not ln.endswith(".")}
    return [ln for ln in lines if ln not in headers and not FURNITURE_RE.match(ln)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--text-dir", required=True, help="pdftotext -enc UTF-8 output, one <stem>.txt per PDF"
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-sentence-chars", type=int, default=25)
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument(
        "--lexicon",
        default="",
        help="English word list, one word per line (e.g. a copy of "
        "/usr/share/dict/american-english); enables the foreign-example filter",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lexicon: frozenset[str] = frozenset()
    if args.lexicon:
        lexicon = frozenset(
            w.strip().lower()
            for w in Path(args.lexicon).read_text(encoding="utf-8").splitlines()
            if w.strip() and "'" not in w
        )
    pdfs = sorted(Path(args.text_dir).glob("*.txt"))
    occurrences: list[dict] = []
    manifest = {"source": "pdf", "text_dir": args.text_dir, "articles": []}
    with open(out_dir / "article_sentences_en.jsonl", "w", encoding="utf-8") as fs:
        for pdf in pdfs:
            lines = pdf_lines(pdf)
            names = cited_names(lines)  # + PERSON entities, once the sentences are known
            kept = clean_article_text(lines)
            n_back = len(lines) - len(kept)
            kept = [ln for ln in kept if not is_foreign_line(ln, lexicon)]
            n_foreign = len(lines) - n_back - len(kept)
            sents = [strip_citations(s.strip()) for s in split_sentences("\n".join(kept))]
            sents = [s for s in sents if len(s) >= args.min_sentence_chars]
            names = names | person_names(sents, args.spacy_model)
            n_occ = 0
            for i, s in enumerate(sents):
                fs.write(
                    json.dumps(
                        {"article_id": pdf.stem, "sent_idx": i, "text": s}, ensure_ascii=False
                    )
                    + "\n"
                )
                for key, surface in normalized_units(s, args.spacy_model, names, lexicon):
                    occurrences.append(
                        {
                            "article_id": pdf.stem,
                            "sent_idx": i,
                            "key": key,
                            "surface": surface,
                            "parent_sentence": s,
                        }
                    )
                    n_occ += 1
            manifest["articles"].append(
                {
                    "article_id": pdf.stem,
                    "text": str(pdf),
                    "lines": len(lines),
                    "front_back_matter_lines": n_back,
                    "foreign_example_lines": n_foreign,
                    "sentences": len(sents),
                    "unit_occurrences": n_occ,
                    "cited_names": sorted(names),
                }
            )
            print(
                f"{pdf.stem}: {len(names)} cited names, {len(lines)} lines "
                f"({n_back} front/back matter, "
                f"{n_foreign} foreign example lines cut), "
                f"{len(sents)} sentences, {n_occ} unit occurrences"
            )

    merged = merge_units(occurrences)
    with open(out_dir / "article_units_v2.jsonl", "w", encoding="utf-8") as f:
        for m in merged:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    per_article = Counter(m["article_id"] for m in merged)
    for a in manifest["articles"]:
        a["unique_units"] = per_article[a["article_id"]]
        print(
            f"{a['article_id']}: {a['unit_occurrences']} occurrences -> "
            f"{a['unique_units']} unique units"
        )
    (out_dir / "article_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
