"""Stage 2 (inside langembed-ml container, GPU): embed every phrase-level
semantic unit for the sampled articles with the same branch-A model used for
the pipeline run. Unit granularity, not whole-sentence -- see
langembed.data.semantic_units. `parent_sentence` is carried through so the
graph UI can show surrounding context on hover.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def embed_article_units(in_path: Path, out_path: Path, model_dir: str, batch_size: int) -> int:
    from sentence_transformers import SentenceTransformer

    records = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    model = SentenceTransformer(model_dir)
    texts = [r["text"] for r in records]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r, emb in zip(records, embeddings, strict=True):
            f.write(
                json.dumps(
                    {
                        "article_id": r["article_id"],
                        "sent_idx": r["sent_idx"],
                        "unit_idx": r["unit_idx"],
                        "text": r["text"],
                        "parent_sentence": r["parent_sentence"],
                        "embedding": emb.tolist(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/article_semantic_units_en.jsonl")
    ap.add_argument("--out", default="output/en/article_unit_embeddings.jsonl")
    ap.add_argument("--model-dir", default="artifacts/simcse_en")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    n = embed_article_units(Path(args.in_path), Path(args.out), args.model_dir, args.batch_size)
    print(f"embedded {n} semantic units -> {args.out}")


if __name__ == "__main__":
    main()
