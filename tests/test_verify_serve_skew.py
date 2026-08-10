import importlib.util
import json
from pathlib import Path


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "verify_serve_skew",
        Path(__file__).resolve().parent.parent / "scripts" / "verify_serve_skew.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_stops_reading_after_n_samples(monkeypatch, tmp_path):
    """embeddings.jsonl can be tens of GB for large corpora; verify() must stream and
    stop at n_samples rather than materializing every row, or it OOMs (as happened on
    the `or` pipeline run: SIGKILL after loading a 42GB file just to use the first 10 rows)."""
    verify_serve_skew = _load_module()

    n_total = 1000
    embeddings_path = tmp_path / "embeddings.jsonl"
    lines_read = {"count": 0}

    class CountingFile:
        def __init__(self, real_file):
            self._real_file = real_file

        def __iter__(self):
            for line in self._real_file:
                lines_read["count"] += 1
                yield line

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._real_file.close()

    with embeddings_path.open("w", encoding="utf-8") as f:
        for i in range(n_total):
            f.write(json.dumps({"text": f"sentence {i}", "embedding": [float(i), 0.0]}) + "\n")

    real_open = open

    def counting_open(path, *args, **kwargs):
        return CountingFile(real_open(path, *args, **kwargs))

    monkeypatch.setattr(verify_serve_skew, "open", counting_open, raising=False)

    class FakeResponse:
        def __init__(self, embeddings):
            self._embeddings = embeddings

        def raise_for_status(self):
            pass

        def json(self):
            return {"embeddings": self._embeddings}

    def fake_post(url, json, timeout):
        n = len(json["texts"])
        return FakeResponse([[float(i), 0.0] for i in range(n)])

    monkeypatch.setattr(verify_serve_skew.httpx, "post", fake_post)

    ok = verify_serve_skew.verify(
        str(embeddings_path), "http://localhost:8000", n_samples=10, atol=1e-5
    )

    assert ok is True
    assert lines_read["count"] == 10
