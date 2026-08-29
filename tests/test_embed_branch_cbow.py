import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "embed_branch_cbow", Path(__file__).resolve().parent.parent / "scripts" / "embed_branch_cbow.py"
)
assert _SPEC is not None and _SPEC.loader is not None
embed_branch_cbow_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(embed_branch_cbow_module)


def test_pool_sentence_vector_averages_in_vocab_words():
    word_to_vec = {"a": [1.0, 1.0], "b": [3.0, 3.0]}

    result = embed_branch_cbow_module.pool_sentence_vector(["a", "b"], word_to_vec, dim=2)

    assert result == [2.0, 2.0]


def test_pool_sentence_vector_skips_out_of_vocab_words():
    word_to_vec = {"a": [1.0, 1.0]}

    result = embed_branch_cbow_module.pool_sentence_vector(["a", "unknown"], word_to_vec, dim=2)

    assert result == [1.0, 1.0]


def test_pool_sentence_vector_all_out_of_vocab_returns_zero_vector():
    result = embed_branch_cbow_module.pool_sentence_vector(["nope"], {}, dim=3)

    assert result == [0.0, 0.0, 0.0]


def test_embed_branch_cbow_writes_pooled_sentence_vectors(tmp_path, monkeypatch):
    monkeypatch.setattr(embed_branch_cbow_module, "REPO_ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    corpus = tmp_path / "data" / "corpus_en.txt"
    corpus.write_text("the cat sat\nthe dog sat\nthe cat ran\n" * 30, encoding="utf-8")
    out_path = tmp_path / "out" / "embeddings_cbow.jsonl"

    n = embed_branch_cbow_module.embed_branch_cbow(
        "en",
        str(out_path),
        cbow_cfg={"embedding_dim": 6, "window": 1, "vocab_size": 10, "min_frequency": 1},
        embed_sample_size=5,
    )

    assert n == 5
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    row = json.loads(lines[0])
    assert row["text"] == "the cat sat"
    assert len(row["embedding"]) == 6


def test_embed_branch_cbow_uses_same_sentence_prefix_as_embed_corpus(tmp_path, monkeypatch):
    """Branches A/B/C (via embed_corpus.py) embed the first N non-empty lines of the
    corpus file, in file order -- CBOW must select the exact same sentences so the
    four outputs are directly comparable on identical text."""
    monkeypatch.setattr(embed_branch_cbow_module, "REPO_ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    corpus = tmp_path / "data" / "corpus_en.txt"
    lines = [f"sentence number {i} here" for i in range(20)]
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_path = tmp_path / "out" / "embeddings_cbow.jsonl"

    embed_branch_cbow_module.embed_branch_cbow(
        "en",
        str(out_path),
        cbow_cfg={"embedding_dim": 4, "window": 1, "vocab_size": 20, "min_frequency": 1},
        embed_sample_size=5,
    )

    written_texts = [
        json.loads(line)["text"]
        for line in out_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert written_texts == lines[:5]
