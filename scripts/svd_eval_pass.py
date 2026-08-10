"""Standalone SVD-based re-evaluation pass for an already-completed language.

Reuses an already-trained SimCSE model and corpus to generate a second,
SVD-labeled STS test set and evaluate against it -- without re-running
corpus/tokenizer/pretrain/SimCSE, none of which depend on the auto-label
method. Produces a separate sts_test_<lang>_svd.jsonl and
metrics/eval_<lang>_svd.json alongside the existing back-translation-labeled
ones, so results from the two methods stay directly comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_svd_eval_pass(lang: str, n_labels: int, n_components: int) -> dict[str, float]:
    from langembed.annotation.auto_label import write_sts_pairs
    from langembed.annotation.svd_label import build_svd_sts_pairs
    from langembed.eval.evaluate import evaluate

    corpus_path = REPO_ROOT / f"data/corpus_{lang}.txt"
    sts_test_path = REPO_ROOT / f"data/sts_test_{lang}_svd.jsonl"

    print(f"=== [{lang}] SVD auto-label (offline) ===")
    with corpus_path.open(encoding="utf-8") as f:
        sentences = [ln.strip() for ln in f if ln.strip()]
    pairs = build_svd_sts_pairs(sentences, n=n_labels, n_components=n_components)
    n_written = write_sts_pairs(pairs, sts_test_path)
    print(f"  wrote {n_written} SVD-labeled STS pairs -> {sts_test_path}")

    eval_cfg: dict[str, Any] = {
        "language": lang,
        "spacy_model": None,
        "test_path": str(sts_test_path),
        "score_scale": 5.0,
        "retrieval_k": 5,
        "branches": {"A": f"artifacts/simcse_{lang}"},
        "train_paths": [],
        "metrics_path": f"metrics/eval_{lang}_svd.json",
        "label_source": "auto",
        "label_method": "svd",
    }
    eval_path = REPO_ROOT / "configs" / lang / "eval_svd.yaml"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(eval_cfg, f, allow_unicode=True, sort_keys=False)

    print(f"=== [{lang}] eval (SVD-labeled test set) ===")
    results = evaluate(eval_cfg)
    metrics_path = REPO_ROOT / eval_cfg["metrics_path"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  metrics written to {metrics_path}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True)
    ap.add_argument(
        "--n-labels",
        type=int,
        default=60,
        help="STS pairs to generate (matches run_pipeline.py's default)",
    )
    ap.add_argument("--svd-components", type=int, default=100)
    args = ap.parse_args()
    run_svd_eval_pass(args.lang, args.n_labels, args.svd_components)


if __name__ == "__main__":
    main()
