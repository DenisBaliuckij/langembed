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
