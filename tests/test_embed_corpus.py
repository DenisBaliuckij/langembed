import importlib.util
import json
import shutil
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "embed_corpus", Path(__file__).resolve().parent.parent / "scripts" / "embed_corpus.py"
)
assert _SPEC is not None and _SPEC.loader is not None
embed_corpus_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(embed_corpus_module)


class _FakeVector:
    """Minimal stand-in for a numpy row: just enough for `.tolist()`."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class _FakeModel:
    """Stands in for SentenceTransformer: fixed-dim zero vectors, no real ML work."""

    def __init__(self, _out_dir: str, dim: int = 4) -> None:
        self._dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, sentences, normalize_embeddings=True, show_progress_bar=False):
        return [_FakeVector([0.0] * self._dim) for _ in sentences]


def _write_config(tmp_path: Path, sentences_path: Path) -> Path:
    config_path = tmp_path / "contrastive.yaml"
    config_path.write_text(
        f"simcse:\n  out_dir: {tmp_path / 'model'}\n  sentences_path: {sentences_path}\n",
        encoding="utf-8",
    )
    return config_path


def test_embed_corpus_writes_one_line_per_sentence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda out_dir: _FakeModel(out_dir)
    )
    sentences_path = tmp_path / "corpus.txt"
    sentences_path.write_text("hello\nworld\n", encoding="utf-8")
    config_path = _write_config(tmp_path, sentences_path)
    out_path = tmp_path / "out" / "embeddings.jsonl"

    n = embed_corpus_module.embed_corpus(str(config_path), str(out_path))

    assert n == 2
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row["text"] == "hello"
    assert len(row["embedding"]) == 4


def test_embed_corpus_aborts_before_encoding_if_estimated_output_wont_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda out_dir: _FakeModel(out_dir)
    )
    sentences_path = tmp_path / "corpus.txt"
    sentences_path.write_text("hello\nworld\n", encoding="utf-8")
    config_path = _write_config(tmp_path, sentences_path)
    out_path = tmp_path / "out" / "embeddings.jsonl"

    monkeypatch.setattr(shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(100, 99, 1))

    with pytest.raises(embed_corpus_module.DiskSpaceError):
        embed_corpus_module.embed_corpus(str(config_path), str(out_path), min_free_gb=5.0)

    assert not out_path.exists()


def test_embed_corpus_aborts_mid_write_if_space_runs_out(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda out_dir: _FakeModel(out_dir)
    )
    sentences_path = tmp_path / "corpus.txt"
    sentences_path.write_text("\n".join(f"sentence {i}" for i in range(5)) + "\n", encoding="utf-8")
    config_path = _write_config(tmp_path, sentences_path)
    out_path = tmp_path / "out" / "embeddings.jsonl"

    # Real disk_usage for the pre-flight check and the first chunk's check,
    # then simulate the disk filling up from the second chunk's check onward.
    real_disk_usage = shutil.disk_usage
    calls = {"n": 0}

    def flaky_disk_usage(path):
        calls["n"] += 1
        if calls["n"] <= 2:
            return real_disk_usage(path)
        return shutil._ntuple_diskusage(100, 99, 1)

    monkeypatch.setattr(shutil, "disk_usage", flaky_disk_usage)
    monkeypatch.setattr(embed_corpus_module, "ENCODE_CHUNK_SIZE", 2)

    with pytest.raises(embed_corpus_module.DiskSpaceError):
        embed_corpus_module.embed_corpus(str(config_path), str(out_path), min_free_gb=5.0)

    # Rows written before the check tripped (the first chunk of 2) are left in place.
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert 0 < len(lines) < 5


def test_embed_corpus_never_encodes_more_than_chunk_size_at_once(tmp_path, monkeypatch):
    """Regression test: a single model.encode() call over the whole corpus held every
    embedding vector in memory before any writing started, OOM-killing a real
    ~13.8M-sentence run. Encoding must happen in bounded chunks instead."""
    max_chunk_seen = {"n": 0}

    class _SpyModel(_FakeModel):
        def encode(self, sentences, normalize_embeddings=True, show_progress_bar=False):
            max_chunk_seen["n"] = max(max_chunk_seen["n"], len(sentences))
            return super().encode(sentences, normalize_embeddings, show_progress_bar)

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda out_dir: _SpyModel(out_dir)
    )
    sentences_path = tmp_path / "corpus.txt"
    sentences_path.write_text(
        "\n".join(f"sentence {i}" for i in range(25)) + "\n", encoding="utf-8"
    )
    config_path = _write_config(tmp_path, sentences_path)
    out_path = tmp_path / "out" / "embeddings.jsonl"

    monkeypatch.setattr(embed_corpus_module, "ENCODE_CHUNK_SIZE", 10)

    n = embed_corpus_module.embed_corpus(str(config_path), str(out_path))

    assert n == 25
    assert max_chunk_seen["n"] <= 10
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 25
