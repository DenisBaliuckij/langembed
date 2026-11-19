"""Stage 2 (CPU-only -- host-wide policy while the GPU stays idle pending the
PSU upgrade, see sciparse.ollama_utils.GPU_ALLOWED for the equivalent policy
on the Ollama side): embed every phrase-level semantic unit for the sampled
articles with the same branch-A model used for the pipeline run. Unit
granularity, not whole-sentence -- see langembed.data.semantic_units.
`parent_sentence` is carried through so the graph UI can show surrounding
context on hover.

Copy of scripts/embed_article_units.py with an explicit device="cpu" passed
to SentenceTransformer (which otherwise auto-selects CUDA when available),
and default --batch-size lowered since CPU throughput is much lower than GPU.
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

    model = SentenceTransformer(model_dir, device="cpu")
    texts = [r["text"] for r in records]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r, emb in zip(records, embeddings, strict=True):
            f.write(
                # every input field is carried through (v2 merged units add count,
                # variants, contexts, ...); v1 records keep exactly their old fields
                json.dumps(
                    {**r, "embedding": emb.tolist()},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", default="artifacts/simcse_en")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    n = embed_article_units(Path(args.in_path), Path(args.out), args.model_dir, args.batch_size)
    print(f"embedded {n} semantic units (CPU) -> {args.out}")


if __name__ == "__main__":
    main()
