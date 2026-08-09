import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "train_simcse",
    Path(__file__).resolve().parent.parent
    / "src"
    / "langembed"
    / "contrastive"
    / "train_simcse.py",
)
assert _SPEC is not None and _SPEC.loader is not None
train_simcse_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train_simcse_module)

_reservoir_sample = train_simcse_module._reservoir_sample


def _write_lines(path: Path, n: int) -> None:
    path.write_text("\n".join(f"sentence {i}" for i in range(n)) + "\n", encoding="utf-8")


def test_reservoir_sample_returns_all_lines_when_file_smaller_than_k(tmp_path):
    p = tmp_path / "corpus.txt"
    _write_lines(p, 10)
    out = _reservoir_sample(str(p), k=100, seed=42)
    assert len(out) == 10
    assert out == [f"sentence {i}" for i in range(10)]


def test_reservoir_sample_caps_at_k_for_larger_file(tmp_path):
    p = tmp_path / "corpus.txt"
    _write_lines(p, 10_000)
    out = _reservoir_sample(str(p), k=100, seed=42)
    assert len(out) == 100
    # every sampled line really came from the source file
    assert set(out) <= {f"sentence {i}" for i in range(10_000)}


def test_reservoir_sample_is_deterministic_given_same_seed(tmp_path):
    p = tmp_path / "corpus.txt"
    _write_lines(p, 5_000)
    out1 = _reservoir_sample(str(p), k=50, seed=7)
    out2 = _reservoir_sample(str(p), k=50, seed=7)
    assert out1 == out2


def test_reservoir_sample_different_seeds_can_differ(tmp_path):
    p = tmp_path / "corpus.txt"
    _write_lines(p, 5_000)
    out1 = _reservoir_sample(str(p), k=50, seed=1)
    out2 = _reservoir_sample(str(p), k=50, seed=2)
    assert out1 != out2


def test_reservoir_sample_skips_blank_lines(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text("a\n\n  \nb\nc\n", encoding="utf-8")
    out = _reservoir_sample(str(p), k=100, seed=42)
    assert out == ["a", "b", "c"]
