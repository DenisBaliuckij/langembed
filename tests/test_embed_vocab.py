import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "embed_vocab", Path(__file__).resolve().parent.parent / "scripts" / "embed_vocab.py"
)
assert _SPEC is not None and _SPEC.loader is not None
embed_vocab_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(embed_vocab_module)


class _FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class _FakeModel:
    def __init__(self, _out_dir: str, dim: int = 4) -> None:
        self._dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, items, normalize_embeddings=True, show_progress_bar=False):
        return [_FakeVector([0.0] * self._dim) for _ in items]


def _write_config(tmp_path: Path, sentences_path: Path) -> Path:
    config_path = tmp_path / "contrastive.yaml"
    config_path.write_text(
        f"simcse:\n  out_dir: {tmp_path / 'model'}\n  sentences_path: {sentences_path}\n",
        encoding="utf-8",
    )
    return config_path


def test_extract_vocab_dedupes_and_sorts(tmp_path):
    sentences_path = tmp_path / "corpus.txt"
    sentences_path.write_text("the cat sat\nthe dog sat\n", encoding="utf-8")

    words = embed_vocab_module.extract_vocab(str(sentences_path), lang="en")

    assert words == sorted({"the", "cat", "sat", "dog"})


def test_embed_vocab_direct_method_writes_one_line_per_unique_word(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda out_dir: _FakeModel(out_dir)
    )
    sentences_path = tmp_path / "corpus.txt"
    sentences_path.write_text("the cat sat\nthe dog sat\n", encoding="utf-8")
    config_path = _write_config(tmp_path, sentences_path)
    out_path = tmp_path / "out" / "vocab_embeddings.jsonl"

    n = embed_vocab_module.embed_vocab(str(config_path), str(out_path), lang="en")

    assert n == 4  # the, cat, sat, dog
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    row = json.loads(lines[0])
    assert "word" in row
    assert len(row["embedding"]) == 4


def test_embed_vocab_direct_method_much_smaller_than_sentence_count(tmp_path, monkeypatch):
    """The whole point of a vocab table: repeating words across many sentences must
    not multiply the output size the way per-sentence embedding does."""
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda out_dir: _FakeModel(out_dir)
    )
    sentences_path = tmp_path / "corpus.txt"
    sentences_path.write_text(
        "\n".join("the cat sat on the mat" for _ in range(1000)) + "\n", encoding="utf-8"
    )
    config_path = _write_config(tmp_path, sentences_path)
    out_path = tmp_path / "out" / "vocab_embeddings.jsonl"

    n = embed_vocab_module.embed_vocab(str(config_path), str(out_path), lang="en")

    assert n == 5  # the, cat, sat, on, mat -- not 1000 * 6


def test_embed_vocab_direct_method_binary_format(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda out_dir: _FakeModel(out_dir)
    )
    sentences_path = tmp_path / "corpus.txt"
    sentences_path.write_text("hello world\n", encoding="utf-8")
    config_path = _write_config(tmp_path, sentences_path)
    out_path = tmp_path / "out" / "vocab_embeddings.npy"

    n = embed_vocab_module.embed_vocab(str(config_path), str(out_path), lang="en", fmt="binary")

    assert n == 2
    import numpy as np

    arr = np.load(out_path)
    assert arr.shape == (2, 4)
    meta = json.loads(Path(str(out_path) + ".meta.json").read_text(encoding="utf-8"))
    assert meta["n"] == 2


def test_embed_vocab_cbow_method_does_not_touch_sentence_transformers(tmp_path, monkeypatch):
    """method='cbow' must not import/construct the SimCSE encoder at all -- it trains
    its own independent embedding matrix."""

    def _boom(*_a, **_k):
        raise AssertionError("SentenceTransformer should not be constructed for --method cbow")

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _boom)

    sentences_path = tmp_path / "corpus.txt"
    sentences_path.write_text(
        "\n".join("the cat sat on the mat" for _ in range(50)) + "\n", encoding="utf-8"
    )
    config_path = _write_config(tmp_path, sentences_path)
    out_path = tmp_path / "out" / "vocab_embeddings.jsonl"

    n = embed_vocab_module.embed_vocab(str(config_path), str(out_path), lang="en", method="cbow")

    assert n == 5
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    row = json.loads(lines[0])
    assert "word" in row and "embedding" in row


def test_embed_vocab_cbow_method_uses_cbow_config_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda out_dir: _FakeModel(out_dir)
    )
    sentences_path = tmp_path / "corpus.txt"
    sentences_path.write_text(
        "\n".join("the cat sat on the mat" for _ in range(50)) + "\n", encoding="utf-8"
    )
    config_path = tmp_path / "contrastive.yaml"
    config_path.write_text(
        f"simcse:\n  out_dir: {tmp_path / 'model'}\n  sentences_path: {sentences_path}\n"
        "cbow:\n  embedding_dim: 6\n  window: 1\n  vocab_size: 3\n  min_frequency: 1\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "out" / "vocab_embeddings.jsonl"

    n = embed_vocab_module.embed_vocab(str(config_path), str(out_path), lang="en", method="cbow")

    assert n == 3  # vocab_size cap from the config's cbow section
    row = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert len(row["embedding"]) == 6  # embedding_dim from the config's cbow section
