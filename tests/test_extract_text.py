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


def test_tex_to_text_strips_comments() -> None:
    tex = "Real text here. % this is a comment, not prose\nMore text follows."
    result = tex_to_text(tex)
    assert "this is a comment" not in result
    assert "Real text here." in result
    assert "More text follows." in result


def test_tex_to_text_drops_math() -> None:
    tex = r"The energy is $E = mc^2$ and also \[F = ma\], as shown above."
    result = tex_to_text(tex)
    assert "E = mc^2" not in result
    assert "F = ma" not in result
    assert "The energy is" in result
    assert "as shown above." in result


def test_tex_to_text_drops_non_prose_environments() -> None:
    tex = (
        "Intro sentence here.\n"
        r"\begin{table}\begin{tabular}{cc}a & b \\ c & d\end{tabular}\end{table}"
        "\nConclusion sentence here."
    )
    result = tex_to_text(tex)
    assert "a & b" not in result
    assert "Intro sentence here." in result
    assert "Conclusion sentence here." in result


def test_tex_to_text_unwraps_formatting_commands_keeping_text() -> None:
    tex = r"This is \textbf{very important} and \emph{also emphasized} text."
    result = tex_to_text(tex)
    assert "very important" in result
    assert "also emphasized" in result
    assert "\\textbf" not in result
    assert "\\emph" not in result


def test_tex_to_text_unwraps_nested_formatting_commands() -> None:
    tex = r"A \textbf{\emph{doubly wrapped}} phrase."
    result = tex_to_text(tex)
    assert "doubly wrapped" in result
    assert "\\" not in result


def test_tex_to_text_drops_citations_and_labels() -> None:
    tex = r"As shown in prior work \cite{smith2020}, the result holds \label{eq:main}."
    result = tex_to_text(tex)
    assert "smith2020" not in result
    assert "eq:main" not in result
    assert "As shown in prior work" in result


def test_tex_to_text_drops_document_environment_wrapper_without_leaking_name() -> None:
    tex = r"\begin{document}Real sentence content here.\end{document}"
    result = tex_to_text(tex)
    assert result == "Real sentence content here."
    assert "document" not in result


def test_tex_to_text_drops_section_markers_keeping_heading_text() -> None:
    tex = r"\section{Introduction}" "\nThis section introduces the topic."
    result = tex_to_text(tex)
    assert "Introduction" in result
    assert "\\section" not in result
    assert "This section introduces the topic." in result
