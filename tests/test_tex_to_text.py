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
