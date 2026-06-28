"""LLM2Vec stage 1 (optional): masked next-token prediction to adapt a decoder
LLM for bidirectional encoding before contrastive training.

Full MNTP + bidirectional conversion is non-trivial; the maintained reference
implementation is the `llm2vec` library. This wrapper delegates to it when
available and gives clear guidance otherwise.
"""

from __future__ import annotations

import argparse
from typing import Any

from langembed.config import load_config


def run_mntp(cfg: dict[str, Any]) -> None:
    m = cfg.get("mntp", {})
    if not m.get("enable"):
        raise SystemExit("mntp.enable is false — skip this stage (mode: ready_embedder).")
    try:
        import llm2vec  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "MNTP needs the `llm2vec` package: pip install llm2vec. "
            "See https://github.com/McGill-NLP/llm2vec for the MNTP recipe, then "
            f"save the adapted model to {m['out_dir']}."
        ) from e
    # With llm2vec installed, run its MNTP trainer on cfg['mntp']['corpus_path'];
    # left as an integration point to keep this scaffold dependency-light.
    raise SystemExit(
        "llm2vec is installed: wire its MNTP trainer here "
        f"(corpus={m['corpus_path']}, out={m['out_dir']})."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run_mntp(load_config(args.config))


if __name__ == "__main__":
    main()
