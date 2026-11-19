import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "bridge_corpus", Path(__file__).resolve().parent.parent / "scripts" / "bridge_corpus.py"
)
assert _SPEC is not None and _SPEC.loader is not None
bridge_corpus_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bridge_corpus_module)


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

    paths = bridge_corpus_module.run_fast("mr", [doc], out_dir, normalized_dir)
    assert len(paths) == 1
    written = (out_dir / "mr_bridge_book.txt").read_text(encoding="utf-8")
    assert written == "Hello world.\nSecond sentence.\n"


def test_run_fast_raises_on_zero_sentences(tmp_path, monkeypatch):
    doc = tmp_path / "scanned.pdf"
    doc.write_bytes(b"%PDF fake")
    out_dir = tmp_path / "raw"
    normalized_dir = tmp_path / "normalized"

    monkeypatch.setattr(
        "langembed.data.extract_text.extract_pdf_text",
        lambda p: "",
    )
    monkeypatch.setattr(
        "langembed.data.extract_text.split_sentences",
        lambda t: [],
    )
    monkeypatch.setattr("langembed.data.normalize_to_pdf.normalize_to_pdf", lambda src, out: src)

    with pytest.raises(RuntimeError, match="0 sentences"):
        bridge_corpus_module.run_fast("mr", [doc], out_dir, normalized_dir)


def test_run_sciparse_normalize_only_returns_pdf_paths(tmp_path, monkeypatch):
    doc = tmp_path / "book.djvu"
    doc.write_bytes(b"fake djvu")
    normalized_dir = tmp_path / "normalized"
    fake_pdf = normalized_dir / "book.pdf"

    monkeypatch.setattr(
        "langembed.data.normalize_to_pdf.normalize_to_pdf", lambda src, out: fake_pdf
    )

    paths = bridge_corpus_module.run_sciparse_normalize_only([doc], normalized_dir)
    assert paths == [str(fake_pdf)]


def test_main_fast_writes_result_json(tmp_path, monkeypatch):
    doc = tmp_path / "book.pdf"
    doc.write_bytes(b"%PDF fake")
    out_dir = tmp_path / "raw"
    normalized_dir = tmp_path / "normalized"
    result_json = tmp_path / "result.json"

    monkeypatch.setattr(
        bridge_corpus_module, "run_fast", lambda *a, **k: ["raw/mr_bridge_book.txt"]
    )
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
    bridge_corpus_module.main()
    data = json.loads(result_json.read_text(encoding="utf-8"))
    assert data == {"raw_text_paths": ["raw/mr_bridge_book.txt"]}
