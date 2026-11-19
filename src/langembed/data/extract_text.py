"""Extract sentences from data/raw/voina-i-mir.pdf into the ru raw-corpus format (ru track)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_SENTENCE_END_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ«\"'(])")
_JUNK_RE = re.compile(r"^\s*\d+\s*$")
_MIN_LEN = 20

_TEX_COMMENT_RE = re.compile(r"(?<!\\)%.*")
# Environments whose content isn't prose - drop them (and their content)
# entirely rather than leave stray markup or math fragments behind.
_TEX_DROP_ENV_RE = re.compile(
    r"\\begin\{(equation\*?|align\*?|eqnarray\*?|array|tabular\*?|figure\*?"
    r"|table\*?|verbatim|lstlisting|algorithm\*?)\}.*?\\end\{\1\}",
    re.DOTALL,
)
_TEX_MATH_INLINE_RE = re.compile(r"\$\$.*?\$\$|\$[^$]*\$", re.DOTALL)
_TEX_MATH_DISPLAY_RE = re.compile(r"\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL)
# Commands whose argument is not prose (citations, cross-references, includes)
# - drop the whole command, not just the backslash.
_TEX_DROP_COMMAND_RE = re.compile(
    r"\\(usepackage|documentclass|newcommand|renewcommand|input|include"
    r"|bibliography|bibliographystyle|maketitle|label|ref|eqref|cite[tp]?"
    r"|footnote)\*?(?:\[[^\]]*\])?\{[^{}]*\}"
)
# Formatting commands whose argument IS prose (\textbf{...}, \section{...},
# \emph{...}, ...) - unwrap to keep the text, drop the command itself.
_TEX_KEEP_TEXT_COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}")
_TEX_BEGIN_END_RE = re.compile(r"\\(begin|end)\{[a-zA-Z*]+\}")
_TEX_STRAY_COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?")

# sciparse (PDF->LaTeX conversion, see sciparse.substitute._label_block())
# wraps formulas/figures/tables in its own bracket-tag blocks instead of
# real LaTeX markup, so _TEX_DROP_ENV_RE above never sees them -- left
# alone they leak straight through as if they were prose (e.g.
# "[FIGURE 2 REVIEW] The image shows..." reads as a real sentence). Strip
# just the bracket-tag wrapper; keep any actual plain-English description
# that follows, since that's real semantic content worth having.
_SCIPARSE_FORMULA_TAG_RE = re.compile(r"\[FORMULA \d+ · (?:inline|display)\]\s*(?:—\s*)?")
_SCIPARSE_FIGURE_TAG_RE = re.compile(r"\[FIGURE \d+(?: ⚠ REVIEW)?\]\s*")
_SCIPARSE_TABLE_TAG_RE = re.compile(r"\[TABLE \d+\]\s*")
# sciparse's own failure-fallback strings (Ollama unreachable, no image
# crop, describer never configured, ...) -- these are placeholders, not
# content; drop outright rather than keep as fake prose.
_SCIPARSE_PLACEHOLDER_RE = re.compile(
    r"\[FIGURE DESCRIPTION PLACEHOLDER[^\]]*\]"
    r"|\[No description available\]"
    r"|\[No image available\]"
    r"|\[No text detected in figure via OCR\]"
    r"|\[FORMULA_PLACEHOLDER\]"
)
# Markdown table rows (sciparse's html_table_to_markdown output, following
# a [TABLE N] tag) -- not prose, would otherwise get chunked into nonsense
# "sentences" by split_sentences()/the phrase-unit chunker.
_MARKDOWN_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$\n?", re.MULTILINE)


def tex_to_text(tex: str) -> str:
    """Strip LaTeX markup down to plain natural-language text.

    Pure regex, no AI/parsing library - good enough to feed split_sentences,
    not a faithful LaTeX renderer. Math and non-prose environments (figures,
    tables, code listings, equations) are dropped rather than kept garbled.
    """
    s = _SCIPARSE_PLACEHOLDER_RE.sub(" ", tex)
    s = _SCIPARSE_FORMULA_TAG_RE.sub("", s)
    s = _SCIPARSE_FIGURE_TAG_RE.sub("", s)
    s = _SCIPARSE_TABLE_TAG_RE.sub("", s)
    s = _MARKDOWN_TABLE_ROW_RE.sub(" ", s)
    s = _TEX_COMMENT_RE.sub("", s)
    s = _TEX_DROP_ENV_RE.sub(" ", s)
    s = _TEX_MATH_INLINE_RE.sub(" ", s)
    s = _TEX_MATH_DISPLAY_RE.sub(" ", s)
    s = _TEX_DROP_COMMAND_RE.sub(" ", s)
    # Strip \begin{...}/\end{...} wrappers for surviving (prose) environments
    # before the generic keep-text unwrap below, which would otherwise match
    # \begin{document}/\end{name} too and leak the environment name as text.
    s = _TEX_BEGIN_END_RE.sub(" ", s)
    # Unwrap formatting commands around prose; repeat a few times for the
    # common shallow-nesting case (e.g. \textbf{\emph{word}}).
    for _ in range(3):
        s = _TEX_KEEP_TEXT_COMMAND_RE.sub(r"\1", s)
    s = _TEX_STRAY_COMMAND_RE.sub(" ", s)
    s = s.replace("{", " ").replace("}", " ")
    return re.sub(r"\s{2,}", " ", s).strip()


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
