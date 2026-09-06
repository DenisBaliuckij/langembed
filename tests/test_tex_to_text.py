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


def test_drops_begin_end_environment_markers():
    result = tex_to_text(r"\begin{document}Hello\end{document}")
    assert "document" not in result
    assert "Hello" in result


def test_drops_label_ref_cite_includegraphics_bibliography():
    result = tex_to_text(
        r"See \label{eq:1}\ref{eq:1}\cite{smith2020}"
        r"\includegraphics{fig3.png}\bibliography{refs} text."
    )
    assert "eq:1" not in result
    assert "smith2020" not in result
    assert "fig3.png" not in result
    assert "refs" not in result
    assert "See" in result
    assert "text." in result
