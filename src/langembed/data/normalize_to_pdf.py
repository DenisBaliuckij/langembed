"""Format normalization: converts DjVu/DOC/DOCX/EPUB/FB2 source documents into PDF so
sciparse's and langembed's own PDF-only extraction paths can handle any source format.
PDF input passes through unchanged. Each format-specific converter shells out to an
external tool (no Python library handles these formats reliably) -- this module only
detects format and invokes the right converter.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

_DJVU_EXTS = {".djvu"}
_OFFICE_EXTS = {".doc", ".docx"}
_EBOOK_EXTS = {".epub", ".fb2"}
_PDF_EXTS = {".pdf"}


class UnsupportedFormatError(ValueError):
    pass


def detect_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _PDF_EXTS:
        return "pdf"
    if ext in _DJVU_EXTS:
        return "djvu"
    if ext in _OFFICE_EXTS:
        return "office"
    if ext in _EBOOK_EXTS:
        return "ebook"
    raise UnsupportedFormatError(f"unsupported source format: {path.suffix!r} ({path})")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _djvu_to_pdf(src: Path, out_dir: Path) -> Path:
    out_path = out_dir / f"{src.stem}.pdf"
    _run(["ddjvu", "-format=pdf", str(src), str(out_path)])
    return out_path


def _office_to_pdf(src: Path, out_dir: Path) -> Path:
    _run(
        [
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(out_dir),
            str(src),
        ]
    )
    out_path = out_dir / f"{src.stem}.pdf"
    if not out_path.is_file():
        raise RuntimeError(f"LibreOffice did not produce expected output: {out_path}")
    return out_path


def _ebook_to_pdf(src: Path, out_dir: Path) -> Path:
    out_path = out_dir / f"{src.stem}.pdf"
    _run(["pandoc", str(src), "-o", str(out_path)])
    return out_path


_CONVERTERS = {
    "djvu": _djvu_to_pdf,
    "office": _office_to_pdf,
    "ebook": _ebook_to_pdf,
}


def normalize_to_pdf(src: Path, out_dir: Path) -> Path:
    """Returns a PDF path for `src`: `src` itself if already a PDF, otherwise a newly
    converted PDF written under `out_dir`. Raises UnsupportedFormatError for unknown
    extensions and RuntimeError if the external converter tool is missing or fails."""
    fmt = detect_format(src)
    if fmt == "pdf":
        return src
    out_dir.mkdir(parents=True, exist_ok=True)
    return _CONVERTERS[fmt](src, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="source document")
    ap.add_argument("--out-dir", required=True, type=Path, help="directory for converted PDFs")
    args = ap.parse_args()
    result = normalize_to_pdf(args.input, args.out_dir)
    print(result)


if __name__ == "__main__":
    main()
