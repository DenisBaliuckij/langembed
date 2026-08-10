import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch  # noqa: E402

from langembed.llm.train_gpt import warm_start_embeddings  # noqa: E402


def _make_encoder(vocab_size: int, hidden_size: int):
    from transformers import RobertaConfig, RobertaForMaskedLM

    config = RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=hidden_size * 2,
        max_position_embeddings=32,
        type_vocab_size=1,
    )
    return RobertaForMaskedLM(config)


def _make_gpt(vocab_size: int, hidden_size: int):
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=32,
        n_embd=hidden_size,
        n_layer=1,
        n_head=1,
    )
    return GPT2LMHeadModel(config)


def test_warm_start_embeddings_copies_weights():
    encoder = _make_encoder(vocab_size=50, hidden_size=16)
    gpt = _make_gpt(vocab_size=50, hidden_size=16)

    warm_start_embeddings(gpt, encoder)

    encoder_weights = encoder.roberta.embeddings.word_embeddings.weight
    gpt_weights = gpt.transformer.wte.weight
    assert torch.equal(gpt_weights, encoder_weights)


def test_warm_start_embeddings_raises_on_shape_mismatch():
    encoder = _make_encoder(vocab_size=50, hidden_size=16)
    gpt = _make_gpt(vocab_size=50, hidden_size=32)

    with pytest.raises(ValueError, match="shape"):
        warm_start_embeddings(gpt, encoder)
