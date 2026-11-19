"""Stage 1.5 (needs spaCy + a language model, runs in the langembed-ml
container): explode each extracted sentence into short phrase-level semantic
units (noun phrases / verb phrases / idioms), the actual granularity the
semantic graph and its embeddings are built over -- not whole sentences.
See langembed.data.semantic_units for the chunking logic itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langembed.data.semantic_units import extract_semantic_units


def chunk_article_sentences(
    in_path: Path, out_path: Path, method: str, spacy_model: str
) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_sentences = 0
    n_units = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_sentences += 1
            units = extract_semantic_units(rec["text"], method=method, model_name=spacy_model)
            for unit_idx, unit_text in enumerate(units):
                fout.write(
                    json.dumps(
                        {
                            "article_id": rec["article_id"],
                            "sent_idx": rec["sent_idx"],
                            "unit_idx": unit_idx,
                            "text": unit_text,
                            "parent_sentence": rec["text"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_units += 1
    return n_units, n_sentences


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/article_sentences_en.jsonl")
    ap.add_argument("--out", default="data/article_semantic_units_en.jsonl")
    ap.add_argument("--method", choices=["spacy", "regex"], default="spacy")
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    args = ap.parse_args()

    n_units, n_sentences = chunk_article_sentences(
        Path(args.in_path), Path(args.out), args.method, args.spacy_model
    )
    avg = n_units / n_sentences if n_sentences else 0.0
    print(
        f"chunked {n_sentences} sentences -> {n_units} semantic units "
        f"({avg:.1f} units/sentence avg) -> {args.out}"
    )


if __name__ == "__main__":
    main()
