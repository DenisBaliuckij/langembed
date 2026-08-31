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
_DROP_ENTIRELY_RE = re.compile(
    r"\\(begin|end|label|ref|cite|includegraphics|bibliography)\*?(\[[^\]]*\])?\{[^{}]*\}"
)
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def tex_to_text(tex: str) -> str:
    """Best-effort LaTeX -> plain text: drops comments and math, drops structural
    commands entirely (\\begin/\\end, \\label, \\ref, \\cite, \\includegraphics,
    \\bibliography -- these have no prose value and would otherwise leak their
    argument as garbage text), unwraps other \\command{text} to its argument text
    (keeps prose like \\section{Introduction}), drops bare commands with no argument
    (\\newpage etc.), and collapses excess whitespace."""
    text = _COMMENT_RE.sub("", tex)
    text = _DISPLAY_MATH_RE.sub(" ", text)
    text = _INLINE_MATH_RE.sub(" ", text)
    text = _DROP_ENTIRELY_RE.sub(" ", text)

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
