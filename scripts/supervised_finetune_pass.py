"""Per-method supervised fine-tuning pass: derives triplets for one label
method, fine-tunes the shared unsupervised SimCSE model on them, and produces
a method-specific final embeddings file.

See docs/superpowers/specs/2026-08-10-supervised-finetune-pass-design.md.

Usage:
    python scripts/supervised_finetune_pass.py --lang ru --label-method svd
    python scripts/supervised_finetune_pass.py --lang ru --label-method backtranslation
    python scripts/supervised_finetune_pass.py --lang ru --label-method native
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

LABEL_METHODS = ("svd", "backtranslation", "native")


def get_triplets(lang: str, label_method: str, n_labels: int, n_components: int) -> Path:
    """Return the path to a triplets JSONL file for `label_method`. Generates it
    first for svd/backtranslation; for native, locates the pre-existing
    data/native_triplets_<lang>.jsonl, raising FileNotFoundError if it doesn't
    exist yet (mirrors train_supervised.py's own "run Phase 5 and POST /export
    first" error).
    """
    if label_method == "native":
        native_path = REPO_ROOT / f"data/native_triplets_{lang}.jsonl"
        if not native_path.is_file() or native_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"No native triplets at {native_path}. Deploy the annotation service, "
                "have annotators label pairs, then POST /export first."
            )
        return native_path

    from langembed.annotation.triplets import build_triplets_from_pairs

    corpus_path = REPO_ROOT / f"data/corpus_{lang}.txt"
    with corpus_path.open(encoding="utf-8") as f:
        sentences = [ln.strip() for ln in f if ln.strip()]

    if label_method == "svd":
        from langembed.annotation.svd_label import build_svd_sts_pairs

        pairs = build_svd_sts_pairs(sentences, n=n_labels, n_components=n_components)
    elif label_method == "backtranslation":
        from langembed.annotation.auto_label import build_auto_sts_pairs

        cache_path = REPO_ROOT / f"data/backtranslation_cache_{lang}.jsonl"
        pairs = build_auto_sts_pairs(
            sentences,
            n=n_labels,
            providers=["google", "mymemory"],
            pivot_lang="en",
            source_lang=lang,
            cache_path=cache_path,
        )
    else:
        raise ValueError(f"unknown label_method: {label_method!r}")

    triplets = build_triplets_from_pairs(pairs)
    triplets_path = REPO_ROOT / f"data/triplets_{lang}_{label_method}.jsonl"
    triplets_path.parent.mkdir(parents=True, exist_ok=True)
    with triplets_path.open("w", encoding="utf-8") as f:
        for anchor, positive, negative in triplets:
            f.write(
                json.dumps(
                    {"anchor": anchor, "positive": positive, "negative": negative},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return triplets_path


def run_supervised_finetune_pass(
    lang: str, label_method: str, n_labels: int = 60, n_components: int = 100
) -> None:
    from langembed.contrastive.train_supervised import train_supervised

    print(f"=== [{lang}] supervised fine-tune ({label_method}) ===")
    triplets_path = get_triplets(lang, label_method, n_labels, n_components)
    print(f"  triplets: {triplets_path}")

    supervised_cfg: dict[str, Any] = {
        "seed": 42,
        "supervised": {
            "triplets_path": str(triplets_path),
            "in_dir": f"artifacts/simcse_{lang}",
            "out_dir": f"artifacts/embed_{lang}_{label_method}",
            "batch_size": 32,
            "epochs": 3,
            "warmup_steps": 100,
        },
    }
    supervised_path = REPO_ROOT / "configs" / lang / f"supervised_{label_method}.yaml"
    supervised_path.parent.mkdir(parents=True, exist_ok=True)
    with supervised_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(supervised_cfg, f, allow_unicode=True, sort_keys=False)

    train_supervised(supervised_cfg)
    print(f"  fine-tuned model -> artifacts/embed_{lang}_{label_method}")

    embed_cfg = {
        "simcse": {
            "out_dir": f"artifacts/embed_{lang}_{label_method}",
            "sentences_path": f"data/corpus_{lang}.txt",
        }
    }
    embed_cfg_path = REPO_ROOT / "configs" / lang / f"embed_{label_method}.yaml"
    with embed_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(embed_cfg, f, allow_unicode=True, sort_keys=False)

    out_path = REPO_ROOT / f"output/{lang}/embeddings_{label_method}.jsonl"
    subprocess.run(
        [
            sys.executable,
            "scripts/embed_corpus.py",
            "--config",
            str(embed_cfg_path),
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    print(f"  wrote embeddings -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lang", required=True)
    ap.add_argument("--label-method", required=True, choices=LABEL_METHODS)
    ap.add_argument("--n-labels", type=int, default=60)
    ap.add_argument("--svd-components", type=int, default=100)
    args = ap.parse_args()
    run_supervised_finetune_pass(args.lang, args.label_method, args.n_labels, args.svd_components)


if __name__ == "__main__":
    main()
