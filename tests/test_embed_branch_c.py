import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "embed_branch_c", Path(__file__).resolve().parent.parent / "scripts" / "embed_branch_c.py"
)
assert _SPEC is not None and _SPEC.loader is not None
embed_branch_c_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(embed_branch_c_module)


def test_build_llm_embed_config_uses_given_base_model_and_paths(tmp_path):
    triplets_path = tmp_path / "data" / "triplets_en_svd.jsonl"
    out_dir = tmp_path / "artifacts" / "embed_en_c_llm"

    cfg = embed_branch_c_module.build_llm_embed_config(
        "en", triplets_path, "Qwen/Qwen3-Embedding-0.6B", out_dir
    )

    assert cfg["base_model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert cfg["train"]["triplets_path"] == str(triplets_path)
    assert cfg["train"]["out_dir"] == str(out_dir)
    assert cfg["mode"] == "ready_embedder"
    assert "en sentence" in cfg["instruction"]


def test_embed_branch_c_raises_clearly_when_triplets_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(embed_branch_c_module, "REPO_ROOT", tmp_path)
    (tmp_path / "data").mkdir()

    with pytest.raises(SystemExit, match="triplets_en_svd.jsonl"):
        embed_branch_c_module.embed_branch_c("en", "svd")


class _FakeVector:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeModel:
    def __init__(self, _out_dir, dim=4):
        self._dim = dim

    def get_sentence_embedding_dimension(self):
        return self._dim

    def encode(self, items, normalize_embeddings=True, show_progress_bar=False):
        return [_FakeVector([0.0] * self._dim) for _ in items]


def test_embed_branch_c_trains_and_embeds_when_triplets_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(embed_branch_c_module, "REPO_ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    triplets_path = tmp_path / "data" / "triplets_en_svd.jsonl"
    triplets_path.write_text(
        '{"anchor": "a", "positive": "b", "negative": "c"}\n', encoding="utf-8"
    )
    corpus = tmp_path / "data" / "corpus_en.txt"
    corpus.write_text("hello world\nfoo bar\n", encoding="utf-8")

    from langembed.llm_embed import train_lora as train_lora_module

    seen_train_cfg = {}
    monkeypatch.setattr(train_lora_module, "train_lora", lambda cfg: seen_train_cfg.update(cfg))
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda out_dir: _FakeModel(out_dir)
    )

    n = embed_branch_c_module.embed_branch_c("en", "svd", embed_sample_size=2)

    assert n == 2
    assert seen_train_cfg["train"]["triplets_path"] == str(triplets_path)
    out_path = tmp_path / "output" / "en" / "embeddings_c_llm.jsonl"
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row["text"] == "hello world"
    assert len(row["embedding"]) == 4
