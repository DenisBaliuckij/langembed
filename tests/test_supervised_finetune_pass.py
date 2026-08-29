import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "supervised_finetune_pass",
    Path(__file__).resolve().parent.parent / "scripts" / "supervised_finetune_pass.py",
)
assert _SPEC is not None and _SPEC.loader is not None
supervised_finetune_pass = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(supervised_finetune_pass)


def test_get_triplets_native_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(supervised_finetune_pass, "REPO_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="native_triplets_ru.jsonl"):
        supervised_finetune_pass.get_triplets("ru", "native", n_labels=60, n_components=100)


def test_get_triplets_native_returns_existing_path(monkeypatch, tmp_path):
    monkeypatch.setattr(supervised_finetune_pass, "REPO_ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    native_path = tmp_path / "data" / "native_triplets_ru.jsonl"
    native_path.write_text('{"anchor": "a", "positive": "b", "negative": "c"}\n', encoding="utf-8")

    result = supervised_finetune_pass.get_triplets("ru", "native", n_labels=60, n_components=100)

    assert result == native_path


def test_get_triplets_svd_generates_and_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(supervised_finetune_pass, "REPO_ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "corpus_ru.txt").write_text(
        "\n".join(f"sentence {i}" for i in range(10)), encoding="utf-8"
    )

    from langembed.annotation import svd_label

    # Alternating high/low scores with distinct sentence pairs, not a single
    # identical pair repeated: build_triplets_from_pairs now drops degenerate
    # triplets where positive == negative (see triplets.py), which an all-identical
    # mock would produce once pos_cutoff == neg_cutoff.
    monkeypatch.setattr(
        svd_label,
        "build_svd_sts_pairs",
        lambda sentences, n, n_components, seed: [
            (f"a{i}", f"b{i}", 4.8 if i % 2 == 0 else 0.3) for i in range(n)
        ],
    )

    result = supervised_finetune_pass.get_triplets("ru", "svd", n_labels=6, n_components=3)

    assert result == tmp_path / "data" / "triplets_ru_svd.jsonl"
    lines = result.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0
    row = json.loads(lines[0])
    assert set(row.keys()) == {"anchor", "positive", "negative"}


def test_run_supervised_finetune_pass_calls_train_supervised_and_embed_corpus(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(supervised_finetune_pass, "REPO_ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    native_path = tmp_path / "data" / "native_triplets_ru.jsonl"
    native_path.write_text('{"anchor": "a", "positive": "b", "negative": "c"}\n', encoding="utf-8")

    from langembed.contrastive import train_supervised as train_supervised_module

    seen_cfg = {}

    def fake_train_supervised(cfg):
        seen_cfg.update(cfg)

    monkeypatch.setattr(train_supervised_module, "train_supervised", fake_train_supervised)

    seen_subprocess_args = {}

    def fake_run(args, **kwargs):
        seen_subprocess_args["args"] = args
        seen_subprocess_args["kwargs"] = kwargs

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(supervised_finetune_pass.subprocess, "run", fake_run)

    supervised_finetune_pass.run_supervised_finetune_pass(
        "ru", "native", n_labels=60, n_components=100
    )

    assert seen_cfg["supervised"]["triplets_path"] == str(native_path)
    assert seen_cfg["supervised"]["in_dir"] == str(tmp_path / "artifacts" / "simcse_ru")
    assert seen_cfg["supervised"]["out_dir"] == str(tmp_path / "artifacts" / "embed_ru_native")
    assert any("scripts/embed_corpus.py" in str(a) for a in seen_subprocess_args["args"])
    assert "--out" in seen_subprocess_args["args"]


def test_run_supervised_finetune_pass_with_base_model_and_out_tag(monkeypatch, tmp_path):
    """Branch B: a HuggingFace multilingual model id as base_model, with a distinct
    out_tag so it doesn't collide with Branch A's own artifacts/embed_<lang>_<method>."""
    monkeypatch.setattr(supervised_finetune_pass, "REPO_ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    native_path = tmp_path / "data" / "native_triplets_ru.jsonl"
    native_path.write_text('{"anchor": "a", "positive": "b", "negative": "c"}\n', encoding="utf-8")

    from langembed.contrastive import train_supervised as train_supervised_module

    seen_cfg = {}
    monkeypatch.setattr(
        train_supervised_module, "train_supervised", lambda cfg: seen_cfg.update(cfg)
    )

    seen_subprocess_args = {}

    def fake_run(args, **kwargs):
        seen_subprocess_args["args"] = args

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(supervised_finetune_pass.subprocess, "run", fake_run)

    supervised_finetune_pass.run_supervised_finetune_pass(
        "ru",
        "native",
        n_labels=60,
        n_components=100,
        base_model="sentence-transformers/LaBSE",
        out_tag="b_mling",
    )

    assert seen_cfg["supervised"]["in_dir"] == "sentence-transformers/LaBSE"
    assert seen_cfg["supervised"]["out_dir"] == str(tmp_path / "artifacts" / "embed_ru_b_mling")
    out_arg_index = seen_subprocess_args["args"].index("--out")
    assert seen_subprocess_args["args"][out_arg_index + 1] == str(
        tmp_path / "output" / "ru" / "embeddings_b_mling.jsonl"
    )
