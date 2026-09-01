# Full-Pipeline DAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A manually-triggered Airflow DAG (`full_pipeline`) on `corpus-host` that runs
corpus preparation, embeddings generation (branches A/B/C/CBOW), and branch C's LLM
training for one language per run, with every technology/method choice exposed as a
trigger parameter.

**Architecture:** Two repos change. `langembed_skeleton/langembed` gets the
format-normalization + fast-extraction pieces (run inside the `langembed-ml` container
via SSH). `text-corpuses-processing` gets the sciparse-bridging DB/FTP logic (runs
natively in the Airflow worker) and the DAG file itself, which wires everything together
and launches every compute-heavy step via `SSHOperator` + a parameterized, hardened
`docker run` wrapper.

**Tech Stack:** Python 3.11, Airflow 3 (TaskFlow API), pyodbc/MSSQL, ftplib, pytest,
Docker, LibreOffice/djvulibre/pandoc (new system deps).

**Spec:** `langembed/docs/superpowers/specs/2026-08-31-full-pipeline-dag-design.md`

## Global Constraints

- No new logic duplicates existing pipeline code — every compute task shells out to an
  already-existing script (`run_pipeline.py`, `supervised_finetune_pass.py`,
  `embed_branch_c.py`, `embed_branch_cbow.py`) unchanged.
- Twirpx/glottolog scraper stack is untouched and not wired into this DAG.
- `run_all_branches.py` and `all_branches_queue.sh` are untouched.
- ML-heavy imports (torch, transformers, etc.) stay inside function bodies, matching
  this repo's `CLAUDE.md` convention — none of the new files in this plan need torch at
  all, so this mostly means: don't import them at module level anywhere.
- All new stored-proc access goes through a `repositories/*.py` static-method class,
  matching `text-corpuses-processing`'s existing convention (see `latex_repository.py`).
- `max_active_runs=1` on the DAG (standing rule: never run two embedding-pipeline jobs
  at once on this host).

---

## Task 1: Add format-conversion system dependencies to the Docker image

**Files:**
- Modify: `Dockerfile:34-36` (the `ml` stage)

- [ ] **Step 1: Add the apt packages**

Edit the `ml` stage in `Dockerfile` to install the three new external converters used
by `normalize_to_pdf.py` (Task 2):

```dockerfile
FROM base AS ml

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        djvulibre-bin \
        libreoffice \
        pandoc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --timeout 300 -e ".[ml,translate]"
```

- [ ] **Step 2: Verify the build succeeds locally**

Run: `docker build --target ml -t langembed-ml:plan-check .` from `langembed/`.
Expected: build completes; `docker run --rm langembed-ml:plan-check which ddjvu soffice pandoc`
prints three paths with no error.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "build: add djvulibre/libreoffice/pandoc to the ml image for format normalization"
```

---

## Task 2: `normalize_to_pdf.py` — format detection + PDF normalization

**Files:**
- Create: `src/langembed/data/normalize_to_pdf.py`
- Test: `tests/test_normalize_to_pdf.py`

**Interfaces:**
- Produces: `detect_format(path: Path) -> str` (returns `"pdf"|"djvu"|"office"|"ebook"`,
  raises `UnsupportedFormatError`), `normalize_to_pdf(src: Path, out_dir: Path) -> Path`,
  `UnsupportedFormatError` exception class. Consumed by Task 3 (`bridge_corpus.py`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_normalize_to_pdf.py`:

```python
from pathlib import Path

import pytest

from langembed.data.normalize_to_pdf import (
    UnsupportedFormatError,
    detect_format,
    normalize_to_pdf,
)


def test_detect_format_pdf():
    assert detect_format(Path("book.pdf")) == "pdf"


def test_detect_format_djvu():
    assert detect_format(Path("book.djvu")) == "djvu"


def test_detect_format_office():
    assert detect_format(Path("book.doc")) == "office"
    assert detect_format(Path("book.docx")) == "office"


def test_detect_format_ebook():
    assert detect_format(Path("book.epub")) == "ebook"
    assert detect_format(Path("book.fb2")) == "ebook"


def test_detect_format_unsupported_raises():
    with pytest.raises(UnsupportedFormatError):
        detect_format(Path("book.rar"))


def test_normalize_pdf_passthrough(tmp_path):
    src = tmp_path / "a.pdf"
    src.write_bytes(b"%PDF-1.4 fake")
    result = normalize_to_pdf(src, tmp_path / "out")
    assert result == src


def test_normalize_djvu_calls_ddjvu(tmp_path, monkeypatch):
    src = tmp_path / "a.djvu"
    src.write_bytes(b"fake djvu")
    out_dir = tmp_path / "out"
    calls = []

    def fake_run(cmd, check, capture_output, text):
        calls.append(cmd)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "a.pdf").write_bytes(b"%PDF fake")

    monkeypatch.setattr("langembed.data.normalize_to_pdf.subprocess.run", fake_run)
    result = normalize_to_pdf(src, out_dir)
    assert result == out_dir / "a.pdf"
    assert calls[0][0] == "ddjvu"


def test_normalize_office_calls_soffice(tmp_path, monkeypatch):
    src = tmp_path / "a.docx"
    src.write_bytes(b"fake docx")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def fake_run(cmd, check, capture_output, text):
        (out_dir / "a.pdf").write_bytes(b"%PDF fake")

    monkeypatch.setattr("langembed.data.normalize_to_pdf.subprocess.run", fake_run)
    result = normalize_to_pdf(src, out_dir)
    assert result == out_dir / "a.pdf"


def test_normalize_office_missing_output_raises(tmp_path, monkeypatch):
    src = tmp_path / "a.docx"
    src.write_bytes(b"fake docx")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    monkeypatch.setattr("langembed.data.normalize_to_pdf.subprocess.run", lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        normalize_to_pdf(src, out_dir)


def test_normalize_ebook_calls_pandoc(tmp_path, monkeypatch):
    src = tmp_path / "a.epub"
    src.write_bytes(b"fake epub")
    out_dir = tmp_path / "out"
    calls = []

    def fake_run(cmd, check, capture_output, text):
        calls.append(cmd)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "a.pdf").write_bytes(b"%PDF fake")

    monkeypatch.setattr("langembed.data.normalize_to_pdf.subprocess.run", fake_run)
    result = normalize_to_pdf(src, out_dir)
    assert result == out_dir / "a.pdf"
    assert calls[0][0] == "pandoc"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_normalize_to_pdf.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'langembed.data.normalize_to_pdf'`

- [ ] **Step 3: Write the implementation**

Create `src/langembed/data/normalize_to_pdf.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_normalize_to_pdf.py -v`
Expected: 9 passed

- [ ] **Step 5: Lint and typecheck**

Run: `ruff format src/langembed/data/normalize_to_pdf.py tests/test_normalize_to_pdf.py && ruff check src/langembed/data/normalize_to_pdf.py tests/test_normalize_to_pdf.py && mypy src/langembed/data/normalize_to_pdf.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/langembed/data/normalize_to_pdf.py tests/test_normalize_to_pdf.py
git commit -m "feat(data): add format normalization (DjVu/DOC/EPUB -> PDF)"
```

---

## Task 3: `tex_to_text.py` — LaTeX-to-plaintext stripper

**Files:**
- Create: `src/langembed/data/tex_to_text.py`
- Test: `tests/test_tex_to_text.py`

**Interfaces:**
- Produces: `tex_to_text(tex: str) -> str`. Consumed by Task 7's port
  (`text-corpuses-processing/dags/tex_to_text.py` is a deliberate small duplicate of
  this exact function, kept in-repo there to avoid a cross-repo runtime dependency for
  a ~20-line pure function — see Task 7).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tex_to_text.py`:

```python
from langembed.data.tex_to_text import tex_to_text


def test_strips_comments():
    assert tex_to_text("Hello % this is a comment\nworld") == "Hello \nworld"


def test_strips_inline_and_display_math():
    result = tex_to_text(r"The value $x^2 + 1$ and $$\int_0^1 f(x)dx$$ matter.")
    assert "$" not in result
    assert "The value" in result
    assert "matter." in result


def test_unwraps_section_command_to_its_text():
    assert tex_to_text(r"\section{Introduction}") == "Introduction"


def test_unwraps_nested_commands():
    assert tex_to_text(r"\section{\textbf{Introduction}}") == "Introduction"


def test_drops_bare_commands():
    result = tex_to_text("Page one.\\newpage Page two.")
    assert "\\newpage" not in result
    assert "Page one." in result
    assert "Page two." in result


def test_collapses_excess_blank_lines():
    result = tex_to_text("Para one.\n\n\n\n\nPara two.")
    assert "\n\n\n" not in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tex_to_text.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'langembed.data.tex_to_text'`

- [ ] **Step 3: Write the implementation**

Create `src/langembed/data/tex_to_text.py`:

```python
"""Strips LaTeX markup from sciparse's .tex conversion output, leaving clean prose
text suitable as a langembed training corpus. Not exhaustive LaTeX parsing -- targets
the constructs sciparse itself emits rather than being a general-purpose LaTeX engine.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
_DISPLAY_MATH_RE = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\$[^$]*\$")
_COMMAND_WITH_ARG_RE = re.compile(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?\{([^{}]*)\}")
_BARE_COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def tex_to_text(tex: str) -> str:
    """Best-effort LaTeX -> plain text: drops comments and math, unwraps
    \\command{text} to its argument text (keeps prose like \\section{Introduction}),
    drops bare commands with no argument (\\newpage etc.), and collapses excess
    whitespace."""
    text = _COMMENT_RE.sub("", tex)
    text = _DISPLAY_MATH_RE.sub(" ", text)
    text = _INLINE_MATH_RE.sub(" ", text)

    # Repeatedly unwrap nested \command{...} to their innermost argument text,
    # since a single pass leaves outer commands wrapping already-unwrapped inner
    # ones (e.g. \section{\textbf{Intro}}) unresolved.
    prev = None
    while prev != text:
        prev = text
        text = _COMMAND_WITH_ARG_RE.sub(lambda m: m.group(2), text)

    text = _BARE_COMMAND_RE.sub(" ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help=".tex file to strip")
    ap.add_argument("--output", required=True, type=Path, help="plain-text output path")
    args = ap.parse_args()
    tex = args.input.read_text(encoding="utf-8")
    args.output.write_text(tex_to_text(tex), encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tex_to_text.py -v`
Expected: 6 passed

- [ ] **Step 5: Lint and typecheck**

Run: `ruff format src/langembed/data/tex_to_text.py tests/test_tex_to_text.py && ruff check src/langembed/data/tex_to_text.py tests/test_tex_to_text.py && mypy src/langembed/data/tex_to_text.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/langembed/data/tex_to_text.py tests/test_tex_to_text.py
git commit -m "feat(data): add LaTeX-to-plaintext stripper for sciparse bridging"
```

---

## Task 4: `bridge_corpus.py` — conversion orchestrator CLI

**Files:**
- Create: `scripts/bridge_corpus.py`
- Test: `tests/test_bridge_corpus.py`

**Interfaces:**
- Consumes: `langembed.data.normalize_to_pdf.normalize_to_pdf` (Task 2),
  `langembed.data.extract_text.extract_pdf_text`/`split_sentences` (existing).
- Produces: `run_fast(lang, source_documents, out_dir, normalized_dir) -> list[str]`,
  `run_sciparse_normalize_only(source_documents, normalized_dir) -> list[str]`. CLI
  writes `{"raw_text_paths": [...]}` (fast) or `{"normalized_pdf_paths": [...]}`
  (sciparse) as JSON to stdout and, if `--result-json` given, to that file. Consumed by
  the DAG's `normalize_and_extract`/`corpus_ready`/`sciparse_convert` tasks (Task 10).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bridge_corpus.py`:

```python
import json

from scripts import bridge_corpus


def test_run_fast_writes_sentence_per_line(tmp_path, monkeypatch):
    doc = tmp_path / "book.pdf"
    doc.write_bytes(b"%PDF fake")
    out_dir = tmp_path / "raw"
    normalized_dir = tmp_path / "normalized"

    monkeypatch.setattr(
        "langembed.data.extract_text.extract_pdf_text",
        lambda p: "Hello world. Second sentence.",
    )
    monkeypatch.setattr(
        "langembed.data.extract_text.split_sentences",
        lambda t: ["Hello world.", "Second sentence."],
    )
    monkeypatch.setattr("langembed.data.normalize_to_pdf.normalize_to_pdf", lambda src, out: src)

    paths = bridge_corpus.run_fast("mr", [doc], out_dir, normalized_dir)
    assert len(paths) == 1
    written = (out_dir / "mr_bridge_book.txt").read_text(encoding="utf-8")
    assert written == "Hello world.\nSecond sentence.\n"


def test_run_sciparse_normalize_only_returns_pdf_paths(tmp_path, monkeypatch):
    doc = tmp_path / "book.djvu"
    doc.write_bytes(b"fake djvu")
    normalized_dir = tmp_path / "normalized"
    fake_pdf = normalized_dir / "book.pdf"

    monkeypatch.setattr(
        "langembed.data.normalize_to_pdf.normalize_to_pdf", lambda src, out: fake_pdf
    )

    paths = bridge_corpus.run_sciparse_normalize_only([doc], normalized_dir)
    assert paths == [str(fake_pdf)]


def test_main_fast_writes_result_json(tmp_path, monkeypatch):
    doc = tmp_path / "book.pdf"
    doc.write_bytes(b"%PDF fake")
    out_dir = tmp_path / "raw"
    normalized_dir = tmp_path / "normalized"
    result_json = tmp_path / "result.json"

    monkeypatch.setattr(bridge_corpus, "run_fast", lambda *a, **k: ["raw/mr_bridge_book.txt"])
    monkeypatch.setattr(
        "sys.argv",
        [
            "bridge_corpus.py",
            "--lang",
            "mr",
            "--conversion-method",
            "fast",
            "--source-documents",
            str(doc),
            "--out-dir",
            str(out_dir),
            "--normalized-dir",
            str(normalized_dir),
            "--result-json",
            str(result_json),
        ],
    )
    bridge_corpus.main()
    data = json.loads(result_json.read_text(encoding="utf-8"))
    assert data == {"raw_text_paths": ["raw/mr_bridge_book.txt"]}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bridge_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.bridge_corpus'` (or
`ImportError` — `scripts/` needs an `__init__.py` if it doesn't already have one; check
`ls scripts/__init__.py` first and create an empty one if missing, matching however the
existing `tests/test_run_all_branches.py` already imports from `scripts`)

- [ ] **Step 3: Write the implementation**

Create `scripts/bridge_corpus.py`:

```python
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
            "raw_text_paths": run_fast(args.lang, args.source_documents, args.out_dir, args.normalized_dir)
        }
    else:
        result = {
            "normalized_pdf_paths": run_sciparse_normalize_only(args.source_documents, args.normalized_dir)
        }

    print(json.dumps(result))
    if args.result_json:
        args.result_json.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_bridge_corpus.py -v`
Expected: 3 passed

- [ ] **Step 5: Lint and typecheck**

Run: `ruff format scripts/bridge_corpus.py tests/test_bridge_corpus.py && ruff check scripts/bridge_corpus.py tests/test_bridge_corpus.py && mypy scripts/bridge_corpus.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add scripts/bridge_corpus.py tests/test_bridge_corpus.py
git commit -m "feat(scripts): add bridge_corpus.py conversion orchestrator"
```

---

## Task 5: `docker_run_watchdog.sh` — reusable hardened launcher

**Files:**
- Create: `scripts/docker_run_watchdog.sh`

**Interfaces:**
- Produces: a shell entrypoint `docker_run_watchdog.sh <name-prefix> <timeout-minutes>
  <use-gpu:true|false> <script-args...>`. Consumed by the DAG's `SSHOperator` tasks
  (Task 10), which run this on the host via SSH.

- [ ] **Step 1: Write the script**

Create `scripts/docker_run_watchdog.sh`:

```bash
#!/bin/bash
# Reusable launcher for the full-pipeline DAG's compute-heavy tasks: runs a
# `docker run` against langembed-ml with a unique container name, a disk/memory
# watchdog, and an outer timeout -- the same hardened pattern already proven in
# all_branches_queue.sh, parameterized instead of hardcoded to one nightly job.
#
# Usage:
#   docker_run_watchdog.sh <unique-name-prefix> <timeout-minutes> <use-gpu:true|false> <script-args...>
#
# <script-args...> is forwarded as-is to `python <script-args...>` inside the
# container, e.g.:
#   docker_run_watchdog.sh mr-run-123-shared-corpus-prep 240 true \
#       scripts/run_pipeline.py --lang mr --raw-input data/raw/mr_nllb.txt \
#       --auto-label --auto-label-method svd --embed-sample-size 200
set -euo pipefail

NAME_PREFIX="$1"; shift
TIMEOUT_MINUTES="$1"; shift
USE_GPU="$1"; shift

BASE=/home/s939/langembed_deploy/langembed
MIN_FREE_GB=50
MIN_AVAIL_MEM_GB=2
CONTAINER="${NAME_PREFIX}-$(date +%s)"

GPU_FLAG=()
if [ "$USE_GPU" = "true" ]; then
  GPU_FLAG=(--gpus all)
fi

(
  while true; do
    sleep 20
    FREE_GB=$(df --output=avail -B1G / | tail -1 | tr -d ' ')
    AVAIL_MEM_GB=$(free -g | awk '/^Mem:/{print $7}')
    if [ -n "$FREE_GB" ] && [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
      echo "SEVERE: disk free ${FREE_GB}GB < ${MIN_FREE_GB}GB - killing $CONTAINER" >&2
      docker kill "$CONTAINER" >/dev/null 2>&1 || true
      break
    fi
    if [ -n "$AVAIL_MEM_GB" ] && [ "$AVAIL_MEM_GB" -lt "$MIN_AVAIL_MEM_GB" ]; then
      echo "SEVERE: available memory ${AVAIL_MEM_GB}GB < ${MIN_AVAIL_MEM_GB}GB - killing $CONTAINER" >&2
      docker kill "$CONTAINER" >/dev/null 2>&1 || true
      break
    fi
    if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
      break
    fi
  done
) &
WATCHDOG_PID=$!

cd "$BASE"
timeout "${TIMEOUT_MINUTES}m" docker run --name "$CONTAINER" \
  "${GPU_FLAG[@]}" \
  --add-host host.docker.internal:host-gateway \
  -v "$BASE":/app -v /mnt/nvme-mssql:/mnt/nvme-mssql -w /app \
  langembed-ml:latest \
  python "$@"
RC=$?

kill "$WATCHDOG_PID" 2>/dev/null || true
wait "$WATCHDOG_PID" 2>/dev/null || true
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

exit $RC
```

- [ ] **Step 2: Make it executable and syntax-check it**

Run: `chmod +x scripts/docker_run_watchdog.sh && bash -n scripts/docker_run_watchdog.sh`
Expected: no output (syntax OK)

- [ ] **Step 3: Manual dry-run verification (no pytest framework for shell scripts in
  this repo)**

Run locally (adjust `BASE`/image if testing outside the deployed host):
`scripts/docker_run_watchdog.sh test-dryrun 1 false -c "print('ok')"` against any
available Python image, or defer full verification to Task 12's deployment smoke test.
Expected: prints `ok`, exits 0, `docker ps -a | grep test-dryrun` shows the container
was removed after completion.

- [ ] **Step 4: Commit**

```bash
git add scripts/docker_run_watchdog.sh
git commit -m "feat(scripts): add reusable docker-run watchdog wrapper for the DAG"
```

---

## Task 6: MSSQL migration — `RegisterPdfForLatexConversion`

**Repo:** `text-corpuses-processing`

**Files:**
- Create: `Database/database-v0.34.sql`

**Interfaces:**
- Produces: stored procedure `dbo.RegisterPdfForLatexConversion(@pdfUrl nvarchar(max),
  @locationInFileSystem nvarchar(max))`. Consumed by Task 7's
  `LatexRepository.register_for_conversion`.

- [ ] **Step 1: Write the migration**

Create `Database/database-v0.34.sql`:

```sql
USE [TextCorpuses]
GO

-- Adds a stored procedure to register an already-local PDF (produced by langembed's
-- document-normalization bridge, not downloaded by any of the existing download-*
-- DAGs) directly into the LaTeX conversion queue. AddPdfUrl (earlier migration)
-- inserts with LocationInFileSystem='' for the *download* queue -- it does not set
-- NeedsLatexConversion and assumes the file still needs fetching. This proc is for
-- the opposite case: the file already exists on the FTP server (uploaded by
-- dags/sciparse_bridge.py) and only needs LaTeX conversion.
--
-- Note: the filtered index that required SET QUOTED_IDENTIFIER ON for every write to
-- PdfDocuments (database-v0.32.sql) was already replaced with a plain index in
-- database-v0.33.sql, so no such requirement applies here -- SET QUOTED_IDENTIFIER ON
-- is kept only for consistency with this file's sibling stored procedures.
SET QUOTED_IDENTIFIER ON
GO
SET ANSI_NULLS ON
GO

CREATE PROCEDURE [dbo].[RegisterPdfForLatexConversion]
    @pdfUrl nvarchar(max),
    @locationInFileSystem nvarchar(max)
AS
BEGIN
    SET NOCOUNT ON;

    IF NOT EXISTS(SELECT * FROM dbo.PdfDocuments WHERE LocationInFileSystem = @locationInFileSystem)
    BEGIN
        INSERT INTO dbo.PdfDocuments (PDFUrl, LocationInFileSystem, NeedsLatexConversion, InsertedAt)
        VALUES (@pdfUrl, @locationInFileSystem, 1, SYSUTCDATETIME())
    END
END
GO
```

- [ ] **Step 2: Commit** (deployment to the live MSSQL instance happens in Task 12,
  matching how prior `database-v0.N.sql` migrations in this repo are version-controlled
  separately from being applied)

```bash
git add Database/database-v0.34.sql
git commit -m "feat(db): add RegisterPdfForLatexConversion stored procedure"
```

---

## Task 7: `LatexRepository` additions — register + poll for conversion result

**Repo:** `text-corpuses-processing`

**Files:**
- Modify: `dags/repositories/latex_repository.py`
- Modify: `dags/tests/test_repositories.py`

**Interfaces:**
- Consumes: stored proc from Task 6.
- Produces: `LatexRepository.register_for_conversion(pdf_url: str,
  location_in_filesystem: str) -> None`, `LatexRepository.get_latex_location(pdf_location:
  str) -> str | None`. Consumed by Task 9 (`sciparse_bridge.py`).

- [ ] **Step 1: Write the failing tests**

Add to `dags/tests/test_repositories.py` (after the existing `test_latex_*` tests):

```python
def test_latex_register_for_conversion_calls_stored_proc():
    with patch('repositories.latex_repository.getConfig', return_value=_CFG), \
         patch('repositories.latex_repository.pyodbc.connect') as mock_conn:
        mock_cur = mock_conn.return_value.cursor.return_value
        LatexRepository.register_for_conversion(
            'langembed-bridge:mr/book.pdf', 'langembed_bridge/mr/book.pdf'
        )
        mock_cur.execute.assert_called_once_with(
            "execute [dbo].[RegisterPdfForLatexConversion] @pdfUrl = ?, @locationInFileSystem = ?",
            ('langembed-bridge:mr/book.pdf', 'langembed_bridge/mr/book.pdf')
        )
        mock_conn.return_value.commit.assert_called_once()


def test_latex_get_latex_location_returns_path_when_ready():
    with patch('repositories.latex_repository.getConfig', return_value=_CFG), \
         patch('repositories.latex_repository.pyodbc.connect') as mock_conn:
        mock_conn.return_value.cursor.return_value.fetchone.return_value = ('Tex/book.tex',)
        result = LatexRepository.get_latex_location('langembed_bridge/mr/book.pdf')
        assert result == 'Tex/book.tex'


def test_latex_get_latex_location_returns_none_when_not_ready():
    with patch('repositories.latex_repository.getConfig', return_value=_CFG), \
         patch('repositories.latex_repository.pyodbc.connect') as mock_conn:
        mock_conn.return_value.cursor.return_value.fetchone.return_value = ('',)
        result = LatexRepository.get_latex_location('langembed_bridge/mr/book.pdf')
        assert result is None


def test_latex_get_latex_location_returns_none_when_row_missing():
    with patch('repositories.latex_repository.getConfig', return_value=_CFG), \
         patch('repositories.latex_repository.pyodbc.connect') as mock_conn:
        mock_conn.return_value.cursor.return_value.fetchone.return_value = None
        result = LatexRepository.get_latex_location('langembed_bridge/mr/book.pdf')
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dags && pytest tests/test_repositories.py -v -k latex_register_for_conversion or latex_get_latex_location`
Expected: FAIL with `AttributeError: type object 'LatexRepository' has no attribute
'register_for_conversion'`

- [ ] **Step 3: Add the methods**

Edit `dags/repositories/latex_repository.py`, adding after `save_location`:

```python
    @staticmethod
    def register_for_conversion(pdf_url: str, location_in_filesystem: str) -> None:
        cnxn = pyodbc.connect(getConfig()['ConnectionString'])
        cursor = cnxn.cursor()
        cursor.execute(
            "execute [dbo].[RegisterPdfForLatexConversion] @pdfUrl = ?, @locationInFileSystem = ?",
            (pdf_url, location_in_filesystem)
        )
        cnxn.commit()
        cursor.close()
        cnxn.close()

    @staticmethod
    def get_latex_location(pdf_location: str) -> str | None:
        cnxn = pyodbc.connect(getConfig()['ConnectionString'])
        cursor = cnxn.cursor()
        cursor.execute(
            "SELECT LatexLocation FROM dbo.LatexDocuments WHERE PDFLocation = ?",
            (pdf_location,)
        )
        row = cursor.fetchone()
        cursor.close()
        cnxn.close()
        if row is None or not row[0]:
            return None
        return row[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dags && pytest tests/test_repositories.py -v`
Expected: all pass (existing + 4 new)

- [ ] **Step 5: Commit**

```bash
git add dags/repositories/latex_repository.py dags/tests/test_repositories.py
git commit -m "feat(repositories): add LatexRepository register/poll methods for the bridge"
```

---

## Task 8: `tex_to_text.py` port

**Repo:** `text-corpuses-processing`

**Files:**
- Create: `dags/tex_to_text.py`
- Create: `dags/tests/test_tex_to_text.py`

This is a deliberate small duplicate of Task 3's function — kept here so
`sciparse_bridge.py` (Task 9) has no cross-repo runtime dependency for a ~20-line pure
function. If this drifts from the `langembed` copy in the future, that's an acceptable
cost for the decoupling; do not attempt to import across repos to "fix" it.

**Interfaces:**
- Produces: `tex_to_text(tex: str) -> str` (identical signature/behavior to Task 3).
  Consumed by Task 9.

- [ ] **Step 1: Copy the test file from Task 3**

Create `dags/tests/test_tex_to_text.py` with the same content as
`langembed/tests/test_tex_to_text.py` from Task 3, changing only the import line:

```python
from tex_to_text import tex_to_text


def test_strips_comments():
    assert tex_to_text("Hello % this is a comment\nworld") == "Hello \nworld"


def test_strips_inline_and_display_math():
    result = tex_to_text(r"The value $x^2 + 1$ and $$\int_0^1 f(x)dx$$ matter.")
    assert "$" not in result
    assert "The value" in result
    assert "matter." in result


def test_unwraps_section_command_to_its_text():
    assert tex_to_text(r"\section{Introduction}") == "Introduction"


def test_unwraps_nested_commands():
    assert tex_to_text(r"\section{\textbf{Introduction}}") == "Introduction"


def test_drops_bare_commands():
    result = tex_to_text("Page one.\\newpage Page two.")
    assert "\\newpage" not in result
    assert "Page one." in result
    assert "Page two." in result


def test_collapses_excess_blank_lines():
    result = tex_to_text("Para one.\n\n\n\n\nPara two.")
    assert "\n\n\n" not in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dags && pytest tests/test_tex_to_text.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tex_to_text'`

- [ ] **Step 3: Copy the implementation from Task 3**

Create `dags/tex_to_text.py` with identical content to
`langembed/src/langembed/data/tex_to_text.py` from Task 3 (the module itself, not the
`main()` CLI wrapper — this copy is imported directly by `sciparse_bridge.py`, never run
as a CLI):

```python
"""Strips LaTeX markup from sciparse's .tex conversion output, leaving clean prose
text suitable as a langembed training corpus. Not exhaustive LaTeX parsing -- targets
the constructs sciparse itself emits rather than being a general-purpose LaTeX engine.

Deliberate small duplicate of langembed/src/langembed/data/tex_to_text.py -- kept here
so sciparse_bridge.py has no cross-repo runtime dependency for this ~20-line pure
function. See that file's plan task for the reasoning.
"""

from __future__ import annotations

import re

_COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
_DISPLAY_MATH_RE = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\$[^$]*\$")
_COMMAND_WITH_ARG_RE = re.compile(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?\{([^{}]*)\}")
_BARE_COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def tex_to_text(tex: str) -> str:
    text = _COMMENT_RE.sub("", tex)
    text = _DISPLAY_MATH_RE.sub(" ", text)
    text = _INLINE_MATH_RE.sub(" ", text)

    prev = None
    while prev != text:
        prev = text
        text = _COMMAND_WITH_ARG_RE.sub(lambda m: m.group(2), text)

    text = _BARE_COMMAND_RE.sub(" ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dags && pytest tests/test_tex_to_text.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add dags/tex_to_text.py dags/tests/test_tex_to_text.py
git commit -m "feat: port tex_to_text stripper for sciparse_bridge.py"
```

---

## Task 9: `sciparse_bridge.py` — register, wait, fetch, strip

**Repo:** `text-corpuses-processing`

**Files:**
- Create: `dags/sciparse_bridge.py`
- Create: `dags/tests/test_sciparse_bridge.py`

**Interfaces:**
- Consumes: `ftpConnector` (existing), `repositories.latex_repository.LatexRepository`
  (Task 7), `tex_to_text.tex_to_text` (Task 8).
- Produces: `register_and_wait(pdf_path: Path, lang: str, poll_interval_s: int = 15,
  timeout_s: int = 3600) -> str`, `ConversionTimeoutError`. Consumed by the DAG's
  `sciparse_convert` task (Task 10).

- [ ] **Step 1: Write the failing tests**

Create `dags/tests/test_sciparse_bridge.py`:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import io
from unittest.mock import patch

import pytest

from sciparse_bridge import ConversionTimeoutError, register_and_wait


def test_register_and_wait_uploads_registers_and_returns_stripped_text(tmp_path):
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF fake")

    with patch('sciparse_bridge.ftpConnector') as mock_ftp, \
         patch('sciparse_bridge.LatexRepository') as mock_repo, \
         patch('sciparse_bridge.time.sleep'):
        mock_repo.get_latex_location.return_value = 'Tex/book.tex'
        mock_ftp.getFile.return_value = io.BytesIO(r"\section{Hello}".encode('utf-8'))

        result = register_and_wait(pdf, 'mr', poll_interval_s=1, timeout_s=10)

        mock_ftp.storeFile.assert_called_once()
        assert mock_ftp.storeFile.call_args[0][0] == 'langembed_bridge/mr/book.pdf'
        mock_repo.register_for_conversion.assert_called_once_with(
            pdf_url='langembed-bridge:mr/book.pdf',
            location_in_filesystem='langembed_bridge/mr/book.pdf',
        )
        mock_ftp.getFile.assert_called_once_with('Tex/book.tex', 'Tex')
        assert result == 'Hello'


def test_register_and_wait_polls_until_ready(tmp_path):
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF fake")

    with patch('sciparse_bridge.ftpConnector') as mock_ftp, \
         patch('sciparse_bridge.LatexRepository') as mock_repo, \
         patch('sciparse_bridge.time.sleep') as mock_sleep:
        mock_repo.get_latex_location.side_effect = [None, None, 'Tex/book.tex']
        mock_ftp.getFile.return_value = io.BytesIO(b"text")

        register_and_wait(pdf, 'mr', poll_interval_s=1, timeout_s=100)

        assert mock_repo.get_latex_location.call_count == 3
        assert mock_sleep.call_count == 2


def test_register_and_wait_times_out(tmp_path):
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF fake")

    with patch('sciparse_bridge.ftpConnector') as mock_ftp, \
         patch('sciparse_bridge.LatexRepository') as mock_repo, \
         patch('sciparse_bridge.time.monotonic', side_effect=[0, 1, 2, 100]), \
         patch('sciparse_bridge.time.sleep'):
        mock_repo.get_latex_location.return_value = None

        with pytest.raises(ConversionTimeoutError):
            register_and_wait(pdf, 'mr', poll_interval_s=1, timeout_s=10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dags && pytest tests/test_sciparse_bridge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sciparse_bridge'`

- [ ] **Step 3: Write the implementation**

Create `dags/sciparse_bridge.py`:

```python
"""Bridges langembed's document-normalization output into sciparse's existing LaTeX
conversion queue: uploads a normalized PDF to FTP, registers it via
LatexRepository.register_for_conversion, polls until pdf_conversion (the existing
Airflow DAG) has processed it, fetches the resulting .tex, and strips it to plain
text. Runs natively inside the airflow-worker container (not via SSH/docker run) --
this is pure DB/FTP orchestration with no GPU/heavy-compute cost, and needs direct
access to this repo's own configs.py/ftpConnector.py/repositories. See "Why SSH, not
DockerOperator" in langembed's
docs/superpowers/specs/2026-08-31-full-pipeline-dag-design.md for why the
compute-heavy tasks go through SSH while this one stays in-worker.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ftpConnector import ftpConnector
from repositories.latex_repository import LatexRepository
from tex_to_text import tex_to_text

BRIDGE_FTP_PREFIX = "langembed_bridge"


class ConversionTimeoutError(RuntimeError):
    pass


def register_and_wait(
    pdf_path: Path, lang: str, poll_interval_s: int = 15, timeout_s: int = 3600
) -> str:
    """Uploads `pdf_path` to FTP, registers it for LaTeX conversion, and blocks until
    pdf_conversion has produced a .tex file for it (or raises ConversionTimeoutError
    after `timeout_s`). Returns the .tex file's plain-text content (already stripped
    via tex_to_text)."""
    remote_pdf_path = f"{BRIDGE_FTP_PREFIX}/{lang}/{pdf_path.name}"
    with pdf_path.open("rb") as f:
        ftpConnector.storeFile(remote_pdf_path, f)

    LatexRepository.register_for_conversion(
        pdf_url=f"langembed-bridge:{lang}/{pdf_path.name}",
        location_in_filesystem=remote_pdf_path,
    )

    deadline = time.monotonic() + timeout_s
    latex_location = None
    while time.monotonic() < deadline:
        latex_location = LatexRepository.get_latex_location(remote_pdf_path)
        if latex_location:
            break
        time.sleep(poll_interval_s)

    if not latex_location:
        raise ConversionTimeoutError(f"{remote_pdf_path} was not converted within {timeout_s}s")

    tex_bytes = ftpConnector.getFile(latex_location, "Tex")
    tex_content = tex_bytes.getvalue().decode("utf-8", errors="replace")
    return tex_to_text(tex_content)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--poll-interval-s", type=int, default=15)
    ap.add_argument("--timeout-s", type=int, default=3600)
    args = ap.parse_args()

    text = register_and_wait(args.pdf, args.lang, args.poll_interval_s, args.timeout_s)
    args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dags && pytest tests/test_sciparse_bridge.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add dags/sciparse_bridge.py dags/tests/test_sciparse_bridge.py
git commit -m "feat: add sciparse_bridge.py (register/wait/fetch/strip)"
```

---

## Task 10: Add the SSH provider dependency

**Repo:** `text-corpuses-processing`

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the provider package**

Edit `requirements.txt`, adding one line:

```
apache-airflow-providers-ssh
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "build: add apache-airflow-providers-ssh for the full-pipeline DAG"
```

---

## Task 11: `full_pipeline_dag.py` — the DAG itself

**Repo:** `text-corpuses-processing`

**Files:**
- Create: `dags/full_pipeline_dag.py`
- Create: `dags/tests/test_full_pipeline_dag.py`

**Interfaces:**
- Consumes: `sciparse_bridge.register_and_wait` (Task 9), `docker_run_watchdog.sh`
  (langembed repo, Task 5, deployed to `/home/s939/langembed_deploy/langembed/scripts/`
  in Task 12), `bridge_corpus.py` (langembed repo, Task 4).

- [ ] **Step 1: Write the failing DAG-import test**

Create `dags/tests/test_full_pipeline_dag.py` (mirrors how Airflow DAGs are typically
smoke-tested — importing the module must not raise, and the expected task ids must be
present):

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_dag_imports_and_has_expected_tasks():
    import full_pipeline_dag

    dag = full_pipeline_dag.dag
    task_ids = set(dag.task_ids)
    assert task_ids == {
        "resolve_corpus",
        "wait_for_corpus_size",
        "normalize_and_extract",
        "route_after_normalize",
        "sciparse_convert",
        "corpus_ready",
        "shared_corpus_prep",
        "select_branches",
        "branch_a_finetune",
        "branch_b_finetune",
        "branch_c_lora",
        "branch_cbow",
    }


def test_dag_has_max_active_runs_one():
    import full_pipeline_dag

    assert full_pipeline_dag.dag.max_active_runs == 1


def test_dag_requires_lang_param():
    import full_pipeline_dag

    assert "lang" in full_pipeline_dag.dag.params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dags && pytest tests/test_full_pipeline_dag.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'full_pipeline_dag'`

- [ ] **Step 3: Write the DAG**

Create `dags/full_pipeline_dag.py`:

```python
"""Manually-triggered full-pipeline DAG: corpus (existing text or converted documents)
-> embeddings (branches A/B/C/CBOW) -> LLM training (branch C's LoRA). See
docs/superpowers/specs/2026-08-31-full-pipeline-dag-design.md (langembed repo) for the
full design.
"""

import pendulum

from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.sdk import DAG, Param, task

SSH_CONN_ID = "corpus_host_ssh"
LANGEMBED_BASE = "/home/s939/langembed_deploy/langembed"
WATCHDOG = f"{LANGEMBED_BASE}/scripts/docker_run_watchdog.sh"

_BRANCH_TASK_IDS = {
    "A": "branch_a_finetune",
    "B": "branch_b_finetune",
    "C": "branch_c_lora",
    "CBOW": "branch_cbow",
}

with DAG(
    dag_id="full_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2026, 8, 31, tz="UTC"),
    catchup=False,
    is_paused_upon_creation=True,
    max_active_runs=1,
    tags=["langembed", "manual"],
    params={
        "lang": Param(default="", type="string"),
        "source_mode": Param(default="existing_text", enum=["existing_text", "convert_documents"]),
        "raw_text_path": Param(default="", type="string"),
        "source_documents": Param(default=[], type="array"),
        "conversion_method": Param(default="fast", enum=["fast", "sciparse"]),
        "min_corpus_size_mb": Param(default=50, type="integer"),
        "label_method": Param(default="svd", enum=["svd", "backtranslation", "native"]),
        "branches": Param(default=["A", "B", "C", "CBOW"], type="array"),
        "embed_sample_size": Param(default=200, type="integer"),
        "base_model_b": Param(default="sentence-transformers/LaBSE", type="string"),
        "use_gpu": Param(default=True, type="boolean"),
        "no_clean": Param(default=False, type="boolean"),
        "timeout_conversion_minutes": Param(default=60, type="integer"),
        "timeout_corpus_prep_minutes": Param(default=240, type="integer"),
        "timeout_branch_minutes": Param(default=480, type="integer"),
    },
) as dag:

    @task.branch()
    def resolve_corpus(**context) -> str:
        params = context["params"]
        if not params["lang"]:
            raise ValueError("`lang` is required")
        if not params["branches"]:
            raise ValueError("`branches` must select at least one of A/B/C/CBOW")
        if params["source_mode"] == "convert_documents" and not params["source_documents"]:
            raise ValueError("`source_documents` is required when source_mode=convert_documents")
        return "wait_for_corpus_size" if params["source_mode"] == "existing_text" else "normalize_and_extract"

    @task()
    def wait_for_corpus_size(**context) -> list[str]:
        import time
        from pathlib import Path

        params = context["params"]
        raw_text_path = params["raw_text_path"] or f"data/raw/{params['lang']}_nllb.txt"
        full_path = Path(LANGEMBED_BASE) / raw_text_path
        min_bytes = params["min_corpus_size_mb"] * 1024 * 1024
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if full_path.is_file() and full_path.stat().st_size >= min_bytes:
                return [raw_text_path]
            time.sleep(15)
        raise TimeoutError(f"{full_path} did not reach {params['min_corpus_size_mb']}MB within 600s")

    normalize_and_extract = SSHOperator(
        task_id="normalize_and_extract",
        ssh_conn_id=SSH_CONN_ID,
        command=(
            f"{WATCHDOG} {{{{ dag_run.run_id }}}}-normalize "
            "{{ params.timeout_conversion_minutes }} false "
            "scripts/bridge_corpus.py --lang {{ params.lang }} "
            "--conversion-method {{ params.conversion_method }} "
            "--source-documents {{ params.source_documents | join(' ') }} "
            "--result-json /tmp/{{ dag_run.run_id }}_bridge_result.json"
        ),
    )

    @task.branch()
    def route_after_normalize(**context) -> str:
        return "sciparse_convert" if context["params"]["conversion_method"] == "sciparse" else "corpus_ready"

    @task()
    def sciparse_convert(**context) -> list[str]:
        """Runs natively in the worker (not via SSH) -- see sciparse_bridge.py's
        module docstring for why."""
        import json
        import subprocess
        from pathlib import Path

        params = context["params"]
        run_id = context["dag_run"].run_id
        result_path = Path(f"/tmp/{run_id}_bridge_result.json")
        # normalize_and_extract wrote this over SSH on the host; the worker
        # container and the host share no filesystem, so fetch it via scp.
        subprocess.run(["scp", f"corpus_host:{result_path}", str(result_path)], check=True)
        pdf_paths = json.loads(result_path.read_text(encoding="utf-8"))["normalized_pdf_paths"]

        from sciparse_bridge import register_and_wait

        raw_paths = []
        for i, pdf_path in enumerate(pdf_paths):
            text = register_and_wait(
                Path(pdf_path), params["lang"], timeout_s=params["timeout_conversion_minutes"] * 60
            )
            out_path = f"/tmp/{run_id}_sciparse_{i}.txt"
            Path(out_path).write_text(text, encoding="utf-8")
            raw_paths.append(out_path)
        return raw_paths

    @task(trigger_rule="none_failed_min_one_success")
    def corpus_ready(**context) -> list[str]:
        """Regardless of which upstream branch actually ran (existing_text,
        convert_documents+fast, or convert_documents+sciparse), figures out the
        resulting raw-text file paths to feed shared_corpus_prep."""
        import json
        import subprocess
        from pathlib import Path

        params = context["params"]
        ti = context["ti"]
        run_id = context["dag_run"].run_id

        if params["source_mode"] == "existing_text":
            return ti.xcom_pull(task_ids="wait_for_corpus_size")

        if params["conversion_method"] == "sciparse":
            return ti.xcom_pull(task_ids="sciparse_convert")

        result_path = Path(f"/tmp/{run_id}_bridge_result.json")
        subprocess.run(["scp", f"corpus_host:{result_path}", str(result_path)], check=True)
        return json.loads(result_path.read_text(encoding="utf-8"))["raw_text_paths"]

    shared_corpus_prep = SSHOperator(
        task_id="shared_corpus_prep",
        ssh_conn_id=SSH_CONN_ID,
        command=(
            f"{WATCHDOG} {{{{ dag_run.run_id }}}}-corpus-prep "
            "{{ params.timeout_corpus_prep_minutes }} {{ params.use_gpu | lower }} "
            "scripts/run_pipeline.py --lang {{ params.lang }} "
            "--raw-input {{ ti.xcom_pull(task_ids='corpus_ready') | join(' ') }} "
            "--auto-label --auto-label-method {{ params.label_method }} "
            "--embed-sample-size {{ params.embed_sample_size }}"
        ),
    )

    @task.branch()
    def select_branches(**context) -> list[str]:
        return [_BRANCH_TASK_IDS[b] for b in context["params"]["branches"]]

    branch_a_finetune = SSHOperator(
        task_id="branch_a_finetune",
        ssh_conn_id=SSH_CONN_ID,
        command=(
            f"{WATCHDOG} {{{{ dag_run.run_id }}}}-branch-a "
            "{{ params.timeout_branch_minutes }} {{ params.use_gpu | lower }} "
            "scripts/supervised_finetune_pass.py --lang {{ params.lang }} "
            "--label-method {{ params.label_method }}"
        ),
    )

    branch_b_finetune = SSHOperator(
        task_id="branch_b_finetune",
        ssh_conn_id=SSH_CONN_ID,
        command=(
            f"{WATCHDOG} {{{{ dag_run.run_id }}}}-branch-b "
            "{{ params.timeout_branch_minutes }} {{ params.use_gpu | lower }} "
            "scripts/supervised_finetune_pass.py --lang {{ params.lang }} "
            "--label-method {{ params.label_method }} "
            "--base-model {{ params.base_model_b }} --out-tag b_mling"
        ),
    )

    branch_c_lora = SSHOperator(
        task_id="branch_c_lora",
        ssh_conn_id=SSH_CONN_ID,
        command=(
            f"{WATCHDOG} {{{{ dag_run.run_id }}}}-branch-c "
            "{{ params.timeout_branch_minutes }} {{ params.use_gpu | lower }} "
            "scripts/embed_branch_c.py --lang {{ params.lang }} "
            "--label-method {{ params.label_method }} "
            "--embed-sample-size {{ params.embed_sample_size }}"
        ),
    )

    branch_cbow = SSHOperator(
        task_id="branch_cbow",
        ssh_conn_id=SSH_CONN_ID,
        command=(
            f"{WATCHDOG} {{{{ dag_run.run_id }}}}-branch-cbow "
            "{{ params.timeout_branch_minutes }} false "
            "scripts/embed_branch_cbow.py --lang {{ params.lang }} "
            "--embed-sample-size {{ params.embed_sample_size }}"
        ),
    )

    corpus_branch = resolve_corpus()
    existing_result = wait_for_corpus_size()
    corpus_branch >> existing_result
    corpus_branch >> normalize_and_extract

    route_result = route_after_normalize()
    normalize_and_extract >> route_result

    sciparse_result = sciparse_convert()
    route_result >> sciparse_result

    ready = corpus_ready()
    existing_result >> ready
    route_result >> ready
    sciparse_result >> ready

    ready >> shared_corpus_prep

    branch_selection = select_branches()
    shared_corpus_prep >> branch_selection
    branch_selection >> [branch_a_finetune, branch_b_finetune, branch_c_lora, branch_cbow]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dags && pytest tests/test_full_pipeline_dag.py -v`
Expected: 3 passed. If Airflow's `Param`/`SSHOperator` import signature differs from
what's written here (verify against the actual installed
`apache-airflow-providers-ssh`/`apache-airflow` versions once Task 10 is deployed —
provider APIs occasionally rename kwargs across major versions), fix the specific
mismatch and re-run; the task graph and Jinja template strings do not change.

- [ ] **Step 5: Commit**

```bash
git add dags/full_pipeline_dag.py dags/tests/test_full_pipeline_dag.py
git commit -m "feat: add full_pipeline DAG wiring corpus/embeddings/LLM-training tasks"
```

---

## Task 12: Deploy and run the spec's testing plan

This task is operational, not code — it deploys everything from Tasks 1-11 to
`corpus-host` and works through the 5-point testing plan from the design spec. Perform
each step in order; do not proceed to the next until the current one's expected result
is confirmed.

- [ ] **Step 1: Add the SSH connection to Airflow**

In the Airflow UI (`http://172.21.128.103:5335/`) → Admin → Connections → Add:
`Connection Id: corpus_host_ssh`, `Connection Type: SSH`, `Host: 127.0.0.1` (the
worker container reaches the host's own sshd via the loopback-equivalent — verify
which address is actually reachable from inside the container first, e.g.
`docker exec apache-airflow-airflow-worker-1 ssh -o StrictHostKeyChecking=no s939@host.docker.internal echo ok`;
use `host.docker.internal` instead of `127.0.0.1` if that's what resolves), `Username:
s939`, and the existing automation private key
(`~/.ssh/id_ed25519_automation`'s contents) pasted into the `Private Key` extra field.
Expected: "Test Connection" in the UI succeeds.

- [ ] **Step 2: Deploy langembed repo changes and rebuild the image**

```bash
# from langembed_skeleton/langembed
scp Dockerfile scripts/bridge_corpus.py scripts/docker_run_watchdog.sh \
    src/langembed/data/normalize_to_pdf.py src/langembed/data/tex_to_text.py \
    corpus-host:/home/s939/langembed_deploy/langembed/  # adjust destination subpaths to match the deployed tree
ssh corpus-host "cd /home/s939/langembed_deploy/langembed && docker build --target ml -t langembed-ml:latest ."
```

Expected: image builds successfully; `docker images | grep langembed-ml` shows it.

- [ ] **Step 3: Apply the MSSQL migration**

```bash
ssh corpus-host "docker cp /home/s939/text-corpuses-processing/Database/database-v0.34.sql apache-airflow-mssql-1:/tmp/"
ssh corpus-host "docker exec apache-airflow-mssql-1 /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P 'Qwerty123!' -C -i /tmp/database-v0.34.sql"
```

Expected: no errors; `SELECT OBJECT_ID('dbo.RegisterPdfForLatexConversion')` returns
non-null.

- [ ] **Step 4: Deploy text-corpuses-processing repo changes and rebuild the Airflow
  image**

```bash
scp dags/repositories/latex_repository.py dags/sciparse_bridge.py dags/tex_to_text.py \
    dags/full_pipeline_dag.py requirements.txt \
    corpus-host:/home/s939/apache-airflow/  # adjust destination subpaths to match the deployed tree
ssh corpus-host "cd /home/s939/apache-airflow && docker compose build airflow-worker airflow-scheduler airflow-dag-processor"
ssh corpus-host "cd /home/s939/apache-airflow && docker compose up -d"
```

Expected: rebuild succeeds; `docker ps | grep airflow` shows all services healthy; the
Airflow UI's DAGs list shows `full_pipeline` with no import errors.

- [ ] **Step 5: Unit tests (spec testing-plan item 1)**

Already covered by Tasks 2, 3, 4, 7, 8, 9, 11's own steps — confirm once more post-deploy:

Run on host: `ssh corpus-host "cd /home/s939/apache-airflow && docker compose exec airflow-worker pytest dags/tests/ -v -k 'latex or tex_to_text or sciparse_bridge or full_pipeline_dag'"`
Expected: all pass.

- [ ] **Step 6: Cheap DAG smoke test (spec testing-plan item 2)**

Trigger `full_pipeline` via the Airflow UI with: `lang=mr`, `source_mode=existing_text`,
`raw_text_path` pointing at a truncated ~1MB copy of `mr_nllb.txt` (create one:
`ssh corpus-host "head -c 1000000 /home/s939/langembed_deploy/langembed/data/raw/mr_nllb.txt > /home/s939/langembed_deploy/langembed/data/raw/mr_nllb_smoke.txt"`),
`min_corpus_size_mb=0`, `branches=["CBOW"]`, `use_gpu=false`.
Expected: `wait_for_corpus_size`, `corpus_ready`, `shared_corpus_prep`, `branch_cbow`
all succeed; `branch_a_finetune`/`branch_b_finetune`/`branch_c_lora` show as `skipped`;
`normalize_and_extract`/`route_after_normalize`/`sciparse_convert` show as `skipped`.

- [ ] **Step 7: Per-format conversion smoke test (spec testing-plan item 3)**

For each of a small sample `.pdf`, `.djvu`, `.docx`, `.epub` file (place under e.g.
`/home/s939/langembed_deploy/langembed/data/smoke_samples/`), run `bridge_corpus.py`
directly on host for both `--conversion-method fast` and `--conversion-method sciparse`:

```bash
ssh corpus-host "cd /home/s939/langembed_deploy/langembed && docker run --rm -v \$(pwd):/app -w /app langembed-ml:latest python scripts/bridge_corpus.py --lang mr --conversion-method fast --source-documents data/smoke_samples/sample.docx"
```

Expected: each invocation prints a JSON result with a non-empty `raw_text_paths` (fast)
or `normalized_pdf_paths` (sciparse) list, and the referenced output file(s) contain
readable, non-garbled text.

- [ ] **Step 8: Full end-to-end run (spec testing-plan item 4)**

Trigger `full_pipeline` once with `source_mode=existing_text`, `branches=["A","B","C","CBOW"]`,
a real language; trigger it a second time with `source_mode=convert_documents` and a
real set of `source_documents`. Expected: both complete successfully, producing
`output/<lang>/embeddings_*.jsonl` for every selected branch (matching
`run_all_branches.py`'s existing output convention).

- [ ] **Step 9: Concurrency guard check (spec testing-plan item 5)**

While one of the Step 8 runs is still in progress, trigger `full_pipeline` again.
Expected: Airflow refuses/queues the second run rather than starting it concurrently
(`max_active_runs=1`).

- [ ] **Step 10: Commit any smoke-test fixture files that should stay in the repo**
  (sample documents, if kept for future regression checks; skip if none)

```bash
git add data/smoke_samples/ 2>/dev/null || true
git commit -m "test: add smoke-test sample documents for the full-pipeline DAG" || true
```

Once all 10 steps pass, the DAG is considered tested per the design spec's gate, and the
documentation sub-project (IEEE-standard, Russian, screenshots) can begin as a separate
follow-on spec.

---

## Self-review notes

- **Spec coverage:** every spec section has a task — corpus bridging (Tasks 2-9),
  execution mechanism (Task 5), DAG/params (Task 11), Dockerfile deps (Task 1), testing
  plan (Task 12) maps 1:1 to the spec's 5-point list.
- **Placeholder scan:** no TBD/TODO; the one disclosed uncertainty (Task 11 Step 4,
  SSHOperator/Param kwarg names against the actual installed provider version) is a
  named, bounded verification step with a clear fallback action, not an unresolved gap.
- **Type/name consistency:** `bridge_corpus.py`'s `run_fast`/`run_sciparse_normalize_only`
  return shapes match what `full_pipeline_dag.py`'s `corpus_ready`/`sciparse_convert`
  read from the JSON; `LatexRepository.register_for_conversion`/`get_latex_location`
  signatures match both their test mocks and `sciparse_bridge.py`'s calls;
  `tex_to_text(tex: str) -> str` is identical across both repos' copies.
