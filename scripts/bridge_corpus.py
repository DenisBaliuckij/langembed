"""Corpus-conversion orchestrator for the full-pipeline DAG's `convert_documents`
source mode. For each --source-document: normalizes non-PDF formats to PDF
(normalize_to_pdf), then either extracts text directly (conversion_method=fast) or
just emits the normalized PDF path for the sciparse bridging task to pick up
(conversion_method=sciparse, handled by a separate task -- see
docs/superpowers/specs/2026-08-31-full-pipeline-dag-design.md).

Usage:
    python scripts/bridge_corpus.py --lang mr --conversion-method fast \
        --source-documents data/incoming/book1.pdf data/incoming/book2.djvu

    python scripts/bridge_corpus.py --lang mr --conversion-method sciparse \
        --source-documents data/incoming/book1.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_fast(
    lang: str, source_documents: list[Path], out_dir: Path, normalized_dir: Path
) -> list[str]:
    """conversion_method=fast: normalize each doc to PDF if needed, extract text
    directly via langembed's own extract_pdf_text, write one sentence-per-line raw
    text file per document (same format run_pipeline.py's --input already produces).
    Returns the written file paths as strings (repo-relative where possible)."""
    from langembed.data.extract_text import extract_pdf_text, split_sentences
    from langembed.data.normalize_to_pdf import normalize_to_pdf

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = []
    for doc in source_documents:
        pdf_path = normalize_to_pdf(doc, normalized_dir)
        sentences = split_sentences(extract_pdf_text(pdf_path))
        out_path = out_dir / f"{lang}_bridge_{doc.stem}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            for s in sentences:
                f.write(s + "\n")
        try:
            raw_paths.append(str(out_path.relative_to(REPO_ROOT)))
        except ValueError:
            raw_paths.append(str(out_path))
    return raw_paths


def run_sciparse_normalize_only(source_documents: list[Path], normalized_dir: Path) -> list[str]:
    """conversion_method=sciparse: only normalize to PDF here -- the actual LaTeX
    conversion, waiting, and text extraction happens in a separate native Airflow task
    (text-corpuses-processing's dags/sciparse_bridge.py) that has direct access to
    sciparse's DB/FTP config. Returns the normalized PDF paths (absolute strings) for
    that task to consume."""
    from langembed.data.normalize_to_pdf import normalize_to_pdf

    return [str(normalize_to_pdf(doc, normalized_dir)) for doc in source_documents]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lang", required=True)
    ap.add_argument("--conversion-method", required=True, choices=["fast", "sciparse"])
    ap.add_argument("--source-documents", nargs="+", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--normalized-dir", type=Path, default=REPO_ROOT / "data" / "bridge_normalized")
    ap.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help="if given, also write the JSON result here (in addition to stdout)",
    )
    args = ap.parse_args()

    if args.conversion_method == "fast":
        result = {
            "raw_text_paths": run_fast(
                args.lang, args.source_documents, args.out_dir, args.normalized_dir
            )
        }
    else:
        result = {
            "normalized_pdf_paths": run_sciparse_normalize_only(
                args.source_documents, args.normalized_dir
            )
        }

    print(json.dumps(result))
    if args.result_json:
        args.result_json.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
