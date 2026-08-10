import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "seed_sts_pairs",
        Path(__file__).resolve().parent.parent / "scripts" / "seed_sts_pairs.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_candidates_adjacent_plus_random():
    seed_sts_pairs = _load_module()

    sentences = [f"s{i}" for i in range(5)]
    pairs = seed_sts_pairs.build_candidates(sentences, n_random=3, seed=42)

    adjacent = [(f"s{i}", f"s{i + 1}") for i in range(4)]
    assert pairs[:4] == adjacent
    assert len(pairs) == 4 + 3


def test_seed_caps_corpus_via_reservoir_sample_before_building_candidates(monkeypatch, tmp_path):
    """A multi-million-line corpus must not turn into millions of adjacent pairs, each
    requiring a CPU encode call just to pick a handful of candidates for human review --
    seed() must bound the sentence pool before build_candidates() ever sees it."""
    seed_sts_pairs = _load_module()

    corpus_path = tmp_path / "corpus.txt"
    n_total = 20_000
    corpus_path.write_text("\n".join(f"sentence {i}" for i in range(n_total)), encoding="utf-8")

    config_path = tmp_path / "contrastive.yaml"
    config_path.write_text(
        f"simcse:\n  sentences_path: {corpus_path.as_posix()}\n  out_dir: unused\n",
        encoding="utf-8",
    )

    seen_candidate_pool_size = {}

    real_build_candidates = seed_sts_pairs.build_candidates

    def spying_build_candidates(sentences, n_random, seed):
        seen_candidate_pool_size["n"] = len(sentences)
        return real_build_candidates(sentences, n_random, seed)

    monkeypatch.setattr(seed_sts_pairs, "build_candidates", spying_build_candidates)

    class FakeModel:
        pass

    # seed() does `from sentence_transformers import SentenceTransformer` locally (per this
    # project's convention of keeping heavy ML imports inside functions), so the patch has
    # to land on the real module -- patching the seed_sts_pairs module attribute wouldn't be
    # seen by that fresh local import.
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", lambda path: FakeModel())
    monkeypatch.setattr(
        seed_sts_pairs,
        "uncertainty",
        lambda pairs, model: np.array([0.5] * len(pairs)),
    )

    written_items = []

    class FakeDB:
        def add(self, item):
            written_items.append(item)

        def commit(self):
            pass

        def close(self):
            pass

    def fake_get_db():
        yield FakeDB()

    monkeypatch.setattr(seed_sts_pairs, "get_db", fake_get_db)

    written = seed_sts_pairs.seed(str(config_path), n=10, max_candidate_sentences=500)

    assert seen_candidate_pool_size["n"] <= 500
    assert written == 10
