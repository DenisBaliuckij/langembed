"""End-to-end smoke test: full pipeline on English fixture data, CPU only.

Proves the code path from raw text to Spearman output works on CPU without
real Gujarati data or a GPU. Does NOT validate model quality — Spearman after
50 MLM steps is expected to be low.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("datasets")

pytestmark = pytest.mark.e2e

_FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory):
    """Run all pipeline steps once per module and return output paths."""
    base = tmp_path_factory.mktemp("e2e")
    corpus_file = base / "corpus.txt"
    tokenizer_dir = base / "tokenizer"
    encoder_dir = base / "encoder"
    simcse_dir = base / "simcse"
    metrics_file = base / "metrics.json"

    # ── Step 1: build corpus ──────────────────────────────────────────────
    from langembed.data.build_corpus import build_corpus

    n = build_corpus([str(_FIXTURES / "en_corpus.txt")], str(corpus_file), set())
    assert n > 0, "build_corpus wrote zero lines"
    assert corpus_file.exists()

    # ── Step 2: train tokenizer ───────────────────────────────────────────
    from langembed.preprocess import normalize
    from langembed.tokenizer.train_tokenizer import diagnose, train_tokenizer

    tok = train_tokenizer(str(corpus_file), str(tokenizer_dir), vocab_size=500, min_frequency=1)
    stats = diagnose(tok, str(corpus_file))
    assert stats["unk_rate"] < 0.05, f"unk_rate={stats['unk_rate']:.4f}"
    for line in corpus_file.read_text(encoding="utf-8").splitlines()[:5]:
        normed = normalize(line.strip())
        if not normed:
            continue
        ids = tok(normed)["input_ids"]
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert decoded.replace(" ", "") == normed.replace(" ", "")

    # ── Step 3: MLM pre-training (smoke: 50 steps, tiny model, CPU) ──────
    from langembed.pretrain.train_mlm import train_mlm

    mlm_cfg: dict = {
        "seed": 42,
        "tokenizer_dir": str(tokenizer_dir),
        "corpus_path": str(corpus_file),
        "report_to": [],
        "model": {
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 256,
            "max_position_embeddings": 514,
            "max_seq_length": 64,
        },
        "training": {
            "mlm_probability": 0.15,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-4,
            "weight_decay": 0.01,
            "warmup_steps": 5,
            "max_steps": 200,
            "fp16": False,
            "save_steps": 100,
            "logging_steps": 10,
        },
        "smoke": {"max_steps": 50},
        "out_dir": str(encoder_dir),
    }
    train_mlm(mlm_cfg, smoke=True)
    assert (encoder_dir / "config.json").exists(), "encoder config.json missing"
    weight_files = list(encoder_dir.glob("*.safetensors")) + list(
        encoder_dir.glob("pytorch_model.bin")
    )
    assert weight_files, "no weight files in encoder_dir"

    # ── Step 4: SimCSE (smoke: 20 sentences, 1 epoch) ────────────────────
    from langembed.contrastive.train_simcse import train_simcse

    simcse_cfg: dict = {
        "encoder_dir": str(encoder_dir),
        "simcse": {
            "sentences_path": str(corpus_file),
            "max_seq_length": 64,
            "batch_size": 8,
            "epochs": 1,
            "warmup_steps": 2,
            "out_dir": str(simcse_dir),
        },
    }
    train_simcse(simcse_cfg, smoke=True)
    assert simcse_dir.exists(), "simcse output directory not created"

    return {
        "corpus_file": corpus_file,
        "tokenizer_dir": tokenizer_dir,
        "encoder_dir": encoder_dir,
        "simcse_dir": simcse_dir,
        "metrics_file": metrics_file,
    }


def test_corpus_written(pipeline: dict) -> None:
    assert pipeline["corpus_file"].exists()
    lines = pipeline["corpus_file"].read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0


def test_encoder_config_present(pipeline: dict) -> None:
    cfg_file = pipeline["encoder_dir"] / "config.json"
    assert cfg_file.exists()
    cfg = json.loads(cfg_file.read_text())
    assert cfg.get("hidden_size") == 128


def test_simcse_model_saved(pipeline: dict) -> None:
    assert pipeline["simcse_dir"].exists()
    assert any(pipeline["simcse_dir"].iterdir()), "simcse_dir is empty"


def test_embeddings_l2_normalised(pipeline: dict) -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(pipeline["simcse_dir"]))
    sents = [
        "The dog barked.",
        "The cat meowed.",
        "The computer processed data.",
        "The algorithm ran efficiently.",
    ]
    embs = model.encode(sents, normalize_embeddings=True)
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=0.01)


def test_cosine_in_valid_range(pipeline: dict) -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(pipeline["simcse_dir"]))
    sents = model.encode(
        [
            "The dog barked loudly.",
            "The dog made loud noises.",
            "She baked a chocolate cake.",
            "He wrote a computer program.",
        ],
        normalize_embeddings=True,
    )
    cos_similar = float(sents[0] @ sents[1])
    cos_dissimilar = float(sents[0] @ sents[3])
    assert -1.0 <= cos_similar <= 1.0
    assert -1.0 <= cos_dissimilar <= 1.0


def test_evaluate_runs_and_writes_metrics(pipeline: dict) -> None:
    from langembed.eval.evaluate import evaluate

    eval_cfg = {
        "test_path": str(_FIXTURES / "en_sts_test.jsonl"),
        "score_scale": 5.0,
        "retrieval_k": 5,
        "train_paths": [],
        "branches": {"en_smoke": str(pipeline["simcse_dir"])},
        "metrics_path": str(pipeline["metrics_file"]),
    }
    results = evaluate(eval_cfg)
    assert "spearman_en_smoke" in results
    assert "retrieval_recall@5_en_smoke" in results
    assert "retrieval_mrr@5_en_smoke" in results

    pipeline["metrics_file"].write_text(json.dumps(results, indent=2))
    assert pipeline["metrics_file"].exists()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(pipeline["simcse_dir"]))
    emb = model.encode(["test"])
    assert emb.shape[1] == 128
