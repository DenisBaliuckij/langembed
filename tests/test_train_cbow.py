from langembed.wordembed.train_cbow import build_training_pairs, build_vocab, train_cbow


def test_build_vocab_orders_by_frequency_then_alphabetically():
    tokenized = [
        ["the", "cat", "sat"],
        ["the", "dog", "sat"],
        ["the", "cat", "ran"],
    ]

    vocab = build_vocab(tokenized, vocab_size=10, min_frequency=1)

    # the:3, cat:2, sat:2, dog:1, ran:1 -- ties (cat/sat, dog/ran) alphabetical
    assert vocab == ["the", "cat", "sat", "dog", "ran"]


def test_build_vocab_respects_min_frequency():
    tokenized = [["a", "a", "a", "b"]]

    vocab = build_vocab(tokenized, vocab_size=10, min_frequency=2)

    assert vocab == ["a"]


def test_build_vocab_respects_vocab_size_cap():
    tokenized = [["a", "a", "a", "b", "b", "c"]]

    vocab = build_vocab(tokenized, vocab_size=2, min_frequency=1)

    assert vocab == ["a", "b"]


def test_build_training_pairs_full_window_both_sides():
    # ids: [0, 1, 2, 3, 4], window=1 -> targets at positions 1,2,3
    word_to_id = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
    tokenized = [["a", "b", "c", "d", "e"]]

    pairs = build_training_pairs(tokenized, word_to_id, window=1)

    assert pairs == [
        ([0, 2], 1),
        ([1, 3], 2),
        ([2, 4], 3),
    ]


def test_build_training_pairs_drops_out_of_vocab_tokens_before_windowing():
    word_to_id = {"a": 0, "b": 1, "c": 2}
    tokenized = [["a", "UNK", "b", "c"]]  # UNK dropped -> ids [a, b, c]

    pairs = build_training_pairs(tokenized, word_to_id, window=1)

    assert pairs == [([0, 2], 1)]


def test_build_training_pairs_skips_sentences_too_short_for_window():
    word_to_id = {"a": 0, "b": 1}
    tokenized = [["a", "b"]]

    pairs = build_training_pairs(tokenized, word_to_id, window=2)

    assert pairs == []


def test_train_cbow_end_to_end_smoke(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join("the cat sat on the mat" for _ in range(50)) + "\n", encoding="utf-8"
    )
    cfg = {
        "language": "en",
        "seed": 42,
        "cbow": {
            "sentences_path": str(corpus),
            "embedding_dim": 8,
            "window": 2,
            "vocab_size": 20,
            "min_frequency": 1,
            "batch_size": 16,
            "epochs": 1,
        },
    }

    words, vectors = train_cbow(cfg)

    assert set(words) == {"the", "cat", "sat", "on", "mat"}
    assert len(vectors) == len(words)
    assert all(len(v) == 8 for v in vectors)


def test_train_cbow_empty_vocab_returns_empty(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a b c\n", encoding="utf-8")
    cfg = {
        "language": "en",
        "cbow": {
            "sentences_path": str(corpus),
            "min_frequency": 100,  # nothing meets this bar
        },
    }

    words, vectors = train_cbow(cfg)

    assert words == []
    assert vectors == []
