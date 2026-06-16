"""Phase 6: evaluate branches A and B on the isolated STS test, with leakage guard."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langembed.config import load_config
from langembed.preprocess import normalize


def _h(s: str) -> str:
    return hashlib.sha1(normalize(s).encode("utf-8")).hexdigest()


def assert_no_leakage(test_path: str, train_paths: Sequence[str]) -> None:
    test_hashes: set[str] = set()
    for line in Path(test_path).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        test_hashes |= {_h(r["sentence_a"]), _h(r["sentence_b"])}
    for tp in train_paths:
        p = Path(tp)
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            if line.strip() and _h(line) in test_hashes:
                raise RuntimeError(f"Test leakage detected via {tp}")


def evaluate(cfg: dict[str, Any]) -> dict[str, float]:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

    assert_no_leakage(cfg["test_path"], cfg.get("train_paths", []))
    sa: list[str] = []
    sb: list[str] = []
    scores: list[float] = []
    for line in Path(cfg["test_path"]).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        sa.append(r["sentence_a"])
        sb.append(r["sentence_b"])
        scores.append(r["score"] / cfg["score_scale"])
    evaluator = EmbeddingSimilarityEvaluator(sa, sb, scores, name="gu-sts")
    results: dict[str, float] = {}
    for branch, path in cfg["branches"].items():
        if not Path(path).exists():
            print(f"skip branch {branch}: {path} missing")
            continue
        results[branch] = float(evaluator(SentenceTransformer(path)))
        print(f"branch {branch}: Spearman = {results[branch]:.4f}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    results = evaluate(cfg)
    metrics_path = Path(cfg.get("metrics_path", "metrics/eval.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({f"spearman_{k}": v for k, v in results.items()}, f, indent=2)
    print(f"metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
