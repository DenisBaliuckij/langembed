"""Continuous Bag-of-Words (CBOW): an alternative, from-scratch static word-embedding
method for the vocabulary table (selected via embed_vocab.py's `--method cbow`).

Unlike the default "direct" method (forward-passing each unique word through the
already-trained sentence encoder, i.e. distillation -- see embed_vocab.py), CBOW
trains its own small embedding matrix from the corpus by predicting each word from
its surrounding context window: the classic word2vec formulation (Mikolov et al.,
2013), independent of the sentence encoder.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from langembed.data.reservoir_sample import reservoir_sample

# Bounds CBOW's training corpus the same way train_simcse.py bounds SimCSE's: an
# unbounded training-sentence count reproduces the exact OOM pattern documented
# there (see MAX_TRAIN_EXAMPLES's docstring in contrastive/train_simcse.py).
MAX_TRAIN_SENTENCES = 500_000
SMOKE_SENTENCES = 256


def _tokenize(sentences: list[str], lang: str, spacy_model: str | None) -> list[list[str]]:
    from langembed.preprocess import normalize

    return [normalize(s, lang=lang, spacy_model=spacy_model).split(" ") for s in sentences]


def build_vocab(tokenized: list[list[str]], vocab_size: int, min_frequency: int) -> list[str]:
    """The most frequent `vocab_size` tokens with count >= min_frequency, sorted by
    descending frequency (ties broken alphabetically for determinism)."""
    counts = Counter(tok for sent in tokenized for tok in sent if tok)
    frequent = [(w, c) for w, c in counts.items() if c >= min_frequency]
    frequent.sort(key=lambda wc: (-wc[1], wc[0]))
    return [w for w, _ in frequent[:vocab_size]]


def build_training_pairs(
    tokenized: list[list[str]], word_to_id: dict[str, int], window: int
) -> list[tuple[list[int], int]]:
    """(context_ids, target_id) pairs. Out-of-vocabulary tokens are dropped from each
    sentence before windowing (rather than kept as an <unk> context token), and a
    position is only used as a target when it has a full window on both sides --
    the plain CBOW definition, no edge-padding."""
    pairs = []
    for sent in tokenized:
        ids = [word_to_id[w] for w in sent if w in word_to_id]
        for i in range(window, len(ids) - window):
            context = ids[i - window : i] + ids[i + 1 : i + window + 1]
            pairs.append((context, ids[i]))
    return pairs


def train_cbow(cfg: dict[str, Any], smoke: bool = False) -> tuple[list[str], list[list[float]]]:
    """Train a CBOW model on `cfg["cbow"]["sentences_path"]` and return the learned
    vocabulary and its embedding vectors (input-embedding matrix rows, one per word,
    in the same order). Returns ([], []) if the (possibly frequency-filtered)
    vocabulary ends up empty."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    c = cfg["cbow"]
    lang = cfg.get("language", "en")
    spacy_model = cfg.get("spacy_model")
    seed = cfg.get("seed", 42)
    window = c.get("window", 4)
    embedding_dim = c.get("embedding_dim", 128)
    vocab_size = c.get("vocab_size", 50_000)
    min_frequency = 1 if smoke else c.get("min_frequency", 5)
    batch_size = c.get("batch_size", 512)
    epochs = 1 if smoke else c.get("epochs", 3)

    torch.manual_seed(seed)

    n_target = SMOKE_SENTENCES if smoke else c.get("max_train_sentences", MAX_TRAIN_SENTENCES)
    sentences = reservoir_sample(c["sentences_path"], n_target, seed)
    tokenized = _tokenize(sentences, lang, spacy_model)

    vocab = build_vocab(tokenized, vocab_size, min_frequency)
    if not vocab:
        return [], []
    word_to_id = {w: i for i, w in enumerate(vocab)}

    pairs = build_training_pairs(tokenized, word_to_id, window)
    if not pairs:
        return vocab, [[0.0] * embedding_dim for _ in vocab]

    contexts = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    targets = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    loader = DataLoader(TensorDataset(contexts, targets), batch_size=batch_size, shuffle=True)

    class CBOWModel(nn.Module):
        def __init__(self, vocab_n: int, dim: int) -> None:
            super().__init__()
            self.in_embed = nn.Embedding(vocab_n, dim)
            self.out_proj = nn.Linear(dim, vocab_n)

        def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
            context_vecs = self.in_embed(context_ids)  # (batch, 2*window, dim)
            return self.out_proj(context_vecs.mean(dim=1))  # (batch, vocab_n)

    model = CBOWModel(len(vocab), embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=c.get("learning_rate", 0.003))
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _epoch in range(epochs):
        for context_batch, target_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(context_batch), target_batch)
            loss.backward()
            optimizer.step()

    vectors: list[list[float]] = model.in_embed.weight.detach().cpu().tolist()
    return vocab, vectors


def main() -> None:
    import argparse

    from langembed.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    vocab, vectors = train_cbow(load_config(args.config), smoke=args.smoke)
    dim = len(vectors[0]) if vectors else 0
    print(f"trained CBOW vocab: {len(vocab)} words, dim={dim}")


if __name__ == "__main__":
    main()
