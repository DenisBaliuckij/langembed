"""Verify zero train/serve skew: /embed output must match batch embed_corpus output exactly
(Phase 7 acceptance check, ru track)."""

from __future__ import annotations

import argparse
import json

import httpx
import numpy as np


def verify(embeddings_path: str, serve_url: str, n_samples: int, atol: float) -> bool:
    rows = []
    with open(embeddings_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    sample = rows[:n_samples]
    texts = [r["text"] for r in sample]
    expected = np.array([r["embedding"] for r in sample])

    resp = httpx.post(f"{serve_url}/embed", json={"texts": texts}, timeout=60)
    resp.raise_for_status()
    actual = np.array(resp.json()["embeddings"])

    ok = bool(np.allclose(expected, actual, atol=atol))
    max_diff = float(np.max(np.abs(expected - actual)))
    print(f"skew check: match={ok} max_abs_diff={max_diff:.8f} n={len(sample)}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="artifacts/embeddings_ru/embeddings.jsonl")
    ap.add_argument("--serve-url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--atol", type=float, default=1e-5)
    args = ap.parse_args()
    ok = verify(args.embeddings, args.serve_url, args.n, args.atol)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
