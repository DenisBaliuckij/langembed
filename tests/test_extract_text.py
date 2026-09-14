"""Unit tests for the PDF-to-sentences extraction module (ru track, Phase 1)."""

from __future__ import annotations

from langembed.data.extract_text import split_sentences, tex_to_text


def test_split_sentences_basic() -> None:
    text = "Это первое предложение книги. Это второе предложение книги."
    result = split_sentences(text)
    assert result == [
        "Это первое предложение книги.",
        "Это второе предложение книги.",
    ]


def test_split_sentences_quotation_marks() -> None:
    text = "Он сказал ей тихо: «Пойдём скорее домой». Она согласилась молча и тихо."
    result = split_sentences(text)
    assert result[0] == "Он сказал ей тихо: «Пойдём скорее домой»."
    assert result[1] == "Она согласилась молча и тихо."


def test_split_sentences_multiple_terminators() -> None:
    text = "Неужели это была правда?! Никто не мог поверить в случившееся."
    result = split_sentences(text)
    assert result[0] == "Неужели это была правда?!"
    assert result[1] == "Никто не мог поверить в случившееся."


def test_split_sentences_filters_junk_and_short_fragments() -> None:
    text = "42\nЭто нормальное предложение книги, длиннее лимита символов."
    result = split_sentences(text)
    assert result == ["Это нормальное предложение книги, длиннее лимита символов."]


def test_split_sentences_dehyphenates_wrapped_words() -> None:
    text = "Это слово было перене-\nсено на новую строку книги, вот так вот."
    result = split_sentences(text)
    assert result == ["Это слово было перенесено на новую строку книги, вот так вот."]


def test_tex_to_text_strips_comments_and_commands() -> None:
    tex = r"This is prose. % a comment that should vanish" "\n" r"\label{sec:intro}More prose here."
    result = tex_to_text(tex)
    assert "comment" not in result
    assert "label" not in result
    assert "This is prose." in result
    assert "More prose here." in result


def test_tex_to_text_drops_math_and_non_prose_environments() -> None:
    tex = (
        r"Before the equation. \begin{equation} E = mc^2 \end{equation} After the equation. "
        r"Inline math $x^2 + y^2$ stays out too."
    )
    result = tex_to_text(tex)
    assert "mc^2" not in result
    assert "x^2" not in result
    assert "Before the equation." in result
    assert "After the equation." in result


def test_tex_to_text_unwraps_formatting_commands_keeping_their_text() -> None:
    tex = r"\textbf{Important} results from \emph{this} study."
    result = tex_to_text(tex)
    assert result == "Important results from this study."


def test_tex_to_text_keeps_formula_description_drops_tag_and_raw_math() -> None:
    """sciparse (PDF->LaTeX conversion) wraps formulas in a bracket-tag
    block, not real LaTeX markup -- see sciparse.substitute._label_block().
    The bare tag + garbled OCR'd LaTeX must not leak through as if it were
    prose, but a real plain-English description sciparse generated is
    genuine content worth keeping."""
    tex = (
        "Before the formula. "
        r"[FORMULA 1 · inline]  $\mathbf{\hat{\Pi}}^{\star}$ — "
        "The real part of a tensor product involving real numbers. "
        "After the formula."
    )
    result = tex_to_text(tex)
    assert "FORMULA" not in result
    assert r"\mathbf" not in result
    assert "The real part of a tensor product involving real numbers." in result
    assert "Before the formula." in result
    assert "After the formula." in result


def test_tex_to_text_drops_undescribed_formula_entirely() -> None:
    tex = r"Before. [FORMULA 6 · display]  $${\mathfrak{I}}_{\ell}^{\neq}$$ After."
    result = tex_to_text(tex)
    assert "FORMULA" not in result
    assert "mathfrak" not in result
    assert "Before." in result
    assert "After." in result


def test_tex_to_text_keeps_figure_description_drops_tag() -> None:
    tex = (
        "Before the figure.\n\n"
        "[FIGURE 2 ⚠ REVIEW]\n"
        "The image presents a detailed diagram of a computer system.\n\n"
        "After the figure."
    )
    result = tex_to_text(tex)
    assert "FIGURE" not in result
    assert "REVIEW" not in result
    assert "The image presents a detailed diagram of a computer system." in result
    assert "Before the figure." in result
    assert "After the figure." in result


def test_tex_to_text_drops_figure_placeholder_entirely() -> None:
    tex = (
        "Before.\n\n"
        "[FIGURE 1532 ⚠ REVIEW]\n"
        "[FIGURE DESCRIPTION PLACEHOLDER — caption: 'no caption']\n\n"
        "After."
    )
    result = tex_to_text(tex)
    assert "FIGURE" not in result
    assert "PLACEHOLDER" not in result
    assert "Before." in result
    assert "After." in result


def test_tex_to_text_drops_table_tag_and_markdown_rows() -> None:
    tex = (
        "Before the table.\n\n"
        "[TABLE 4]\n"
        "| Model | Accuracy |\n"
        "| --- | --- |\n"
        "| CNN | 0.91 |\n\n"
        "After the table."
    )
    result = tex_to_text(tex)
    assert "TABLE" not in result
    assert "|" not in result
    assert "Before the table." in result
    assert "After the table." in result
