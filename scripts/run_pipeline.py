"""End-to-end pipeline runner: one or more PDF corpora -> trained embeddings, for any language.

Generalizes the ru language track (see docs/superpowers/plans/2026-07-01-russian-embeddings.md)
into a reusable tool: extraction -> corpus -> tokenizer -> bounded MLM pretrain -> unsupervised
SimCSE -> human-in-the-loop STS labeling -> eval -> final embeddings + serve skew check.

Every stage still runs through the existing per-stage modules via generated
configs/<lang>/*.yaml files (per the project's "no magic numbers outside configs/*.yaml"
convention) -- this script only templates those configs per language and orchestrates the
sequence, mirroring the Makefile targets exactly.

Usage:
    python scripts/run_pipeline.py --lang ru --input data/raw/book1.pdf data/raw/book2.pdf
    python scripts/run_pipeline.py --lang fr --input corpus.pdf --skip-eval
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parent.parent


def free_port(start: int) -> int:
    """Find the first available TCP port at or after `start` (avoids stale-container clashes)."""
    port = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def extract_inputs(lang: str, inputs: list[Path]) -> list[str]:
    from langembed.data.extract_text import extract_pdf_text, split_sentences

    raw_dir = REPO_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = []
    for pdf in inputs:
        print(f"  extracting {pdf.name} ...")
        sentences = split_sentences(extract_pdf_text(pdf))
        out_path = raw_dir / f"{lang}_{pdf.stem}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            for s in sentences:
                f.write(s + "\n")
        print(f"    {len(sentences)} sentences -> {out_path.relative_to(REPO_ROOT)}")
        raw_paths.append(str(out_path.relative_to(REPO_ROOT)))
    return raw_paths


def calibrate_pretrain_steps(config_path: Path, target_minutes: float) -> int:
    t0 = time.time()
    run(
        [
            sys.executable,
            "-m",
            "langembed.pretrain.train_mlm",
            "--config",
            str(config_path),
            "--smoke",
        ]
    )
    elapsed = time.time() - t0
    smoke_steps = yaml.safe_load(config_path.read_text(encoding="utf-8"))["smoke"]["max_steps"]
    return max(50, round(smoke_steps * (target_minutes * 60) / elapsed))


def start_server(module_app: str, port: int, env: dict[str, str] | None = None) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", module_app, "--port", str(port)],
        cwd=REPO_ROOT,
        env=env,
    )
    time.sleep(3)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lang", required=True, help="language code, e.g. ru, fr, de")
    ap.add_argument(
        "--input", nargs="+", default=None, type=Path, help="one or more PDF corpus files"
    )
    ap.add_argument(
        "--raw-input",
        nargs="+",
        default=None,
        type=Path,
        help=(
            "one or more pre-extracted plain-text files (one sentence per line) to use "
            "as the corpus directly, skipping PDF extraction. Exactly one of --input / "
            "--raw-input is required."
        ),
    )
    ap.add_argument(
        "--output", default=Path("output"), type=Path, help="final deliverable directory"
    )
    ap.add_argument(
        "--n-labels", type=int, default=60, help="STS candidate pairs to seed for labeling"
    )
    ap.add_argument(
        "--pretrain-minutes", type=float, default=25.0, help="target MLM pretrain wall time"
    )
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument(
        "--spacy-model",
        default=None,
        help=(
            "spaCy model for text preparation (lemmatize + POS-token "
            "substitution), e.g. ru_core_news_sm; omit to skip"
        ),
    )
    ap.add_argument(
        "--skip-eval",
        action="store_true",
        help="skip labeling/eval; only produce corpus, encoder, SimCSE model and embeddings",
    )
    ap.add_argument(
        "--auto-label",
        action="store_true",
        help=(
            "skip the manual /label step; generate silver STS pairs via "
            "back-translation instead (no human, no docker/postgres needed for this step); "
            "sends corpus sentences to external translation services (Google/MyMemory)"
        ),
    )
    ap.add_argument(
        "--translate-providers",
        nargs="+",
        default=["google", "mymemory"],
        help="free translation backends for back-translation (deep-translator provider names)",
    )
    ap.add_argument(
        "--pivot-lang", default="en", help="pivot language for the back-translation round-trip"
    )
    ap.add_argument(
        "--translate-rpm",
        type=float,
        default=20.0,
        help="max back-translation requests/minute (politeness limit for free MT APIs)",
    )
    return ap


def _resolve_repo_path(path: str) -> Path:
    """Resolve `path` against REPO_ROOT unless it's already absolute, matching the
    convention `run()`/`extract_inputs()` get for free from `cwd=REPO_ROOT`."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_raw_text_inputs(paths: list[Path]) -> list[str]:
    """Validate pre-extracted plain-text corpus files and return absolute path strings
    for direct use as `build_corpus`'s `raw_paths` (one sentence per line, no PDF
    extraction needed). Raises FileNotFoundError naming the first missing path."""
    resolved = []
    for p in paths:
        rp = p if p.is_absolute() else REPO_ROOT / p
        if not rp.is_file():
            raise FileNotFoundError(f"--raw-input file not found: {rp}")
        resolved.append(str(rp))
    return resolved


def generate_auto_sts(
    corpus_path: str,
    sts_test_path: str,
    lang: str,
    providers: list[str],
    pivot_lang: str,
    requests_per_minute: float,
    n_labels: int,
) -> int:
    """Auto-label branch of pipeline step 5: build silver STS pairs via back-translation
    and write them to `sts_test_path`. Returns the number of pairs written. Unlike the
    manual-labeling branch, this has no docker/server/human-input dependency.

    `corpus_path`, `sts_test_path`, and the derived cache path are resolved against
    REPO_ROOT (like every other path in this file) so the pipeline behaves the same
    regardless of the process's current working directory.
    """
    from langembed.annotation.auto_label import (
        ADJACENT_SCORE,
        PARAPHRASE_SCORE,
        RANDOM_SCORE,
        build_auto_sts_pairs,
        write_sts_pairs,
    )

    print("  (sends corpus sentences to external MT services: " + ", ".join(providers) + ")")

    corpus_abs = _resolve_repo_path(corpus_path)
    sts_test_abs = _resolve_repo_path(sts_test_path)
    cache_path = _resolve_repo_path(f"data/backtranslation_cache_{lang}.jsonl")

    with corpus_abs.open(encoding="utf-8") as f:
        sentences = [ln.strip() for ln in f if ln.strip()]
    pairs = build_auto_sts_pairs(
        sentences,
        n=n_labels,
        providers=providers,
        pivot_lang=pivot_lang,
        source_lang=lang,
        cache_path=cache_path,
        requests_per_minute=requests_per_minute,
    )
    n_para = sum(1 for _, _, score in pairs if score == PARAPHRASE_SCORE)
    n_adj = sum(1 for _, _, score in pairs if score == ADJACENT_SCORE)
    n_rand = sum(1 for _, _, score in pairs if score == RANDOM_SCORE)
    print(f"  tiers: paraphrase={n_para} adjacent={n_adj} random={n_rand} -> {sts_test_abs}")
    return write_sts_pairs(pairs, sts_test_abs)


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    if bool(args.input) == bool(args.raw_input):
        ap.error("exactly one of --input or --raw-input is required")

    lang = args.lang
    cfg_dir = REPO_ROOT / "configs" / lang
    corpus_path = f"data/corpus_{lang}.txt"
    sts_test_path = f"data/sts_test_{lang}.jsonl"

    if args.raw_input:
        n_files = len(args.raw_input)
        print(f"=== [{lang}] 1/6 raw text input ({n_files} file(s), no PDF extraction) ===")
        raw_paths = resolve_raw_text_inputs(args.raw_input)
    else:
        print(f"=== [{lang}] 1/6 extraction ({len(args.input)} PDF(s)) ===")
        raw_paths = extract_inputs(lang, args.input)

    tokenizer_cfg = {
        "language": lang,
        "spacy_model": args.spacy_model,
        "data": {"raw_paths": raw_paths, "out_path": corpus_path, "test_path": sts_test_path},
        "tokenizer": {
            "vocab_size": args.vocab_size,
            "min_frequency": 2,
            "unk_rate_max": 0.01,
            "out_dir": f"artifacts/tokenizer_{lang}",
        },
    }
    tokenizer_path = cfg_dir / "tokenizer.yaml"
    write_yaml(tokenizer_path, tokenizer_cfg)

    print(f"=== [{lang}] 2/6 corpus + tokenizer ===")
    run([sys.executable, "-m", "langembed.data.build_corpus", "--config", str(tokenizer_path)])
    run(
        [
            sys.executable,
            "-m",
            "langembed.tokenizer.train_tokenizer",
            "--config",
            str(tokenizer_path),
        ]
    )

    print(f"=== [{lang}] 3/6 bounded MLM pretrain ===")
    pretrain_cfg: dict[str, Any] = {
        "seed": 42,
        "tokenizer_dir": f"artifacts/tokenizer_{lang}",
        "corpus_path": corpus_path,
        "out_dir": f"artifacts/encoder_{lang}",
        "model": {
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "intermediate_size": 1024,
            "max_position_embeddings": 130,
            "max_seq_length": 128,
        },
        "training": {
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.0005,
            "weight_decay": 0.01,
            "warmup_steps": 200,
            "max_steps": 1500,
            "fp16": False,
            "save_steps": 500,
            "logging_steps": 50,
            "mlm_probability": 0.15,
        },
        "smoke": {"max_steps": 50},
    }
    pretrain_path = cfg_dir / "pretrain.yaml"
    write_yaml(pretrain_path, pretrain_cfg)
    max_steps = calibrate_pretrain_steps(pretrain_path, args.pretrain_minutes)
    pretrain_cfg["training"]["max_steps"] = max_steps
    write_yaml(pretrain_path, pretrain_cfg)
    print(f"  calibrated max_steps={max_steps} (target ~{args.pretrain_minutes} min)")
    run([sys.executable, "-m", "langembed.pretrain.train_mlm", "--config", str(pretrain_path)])

    print(f"=== [{lang}] 4/6 unsupervised SimCSE ===")
    contrastive_cfg = {
        "seed": 42,
        "encoder_dir": f"artifacts/encoder_{lang}",
        "simcse": {
            "sentences_path": corpus_path,
            "out_dir": f"artifacts/simcse_{lang}",
            "batch_size": 32,
            "epochs": 1,
            "warmup_steps": 100,
            "max_seq_length": 128,
        },
        "supervised": {
            "triplets_path": f"data/native_triplets_{lang}.jsonl",
            "in_dir": f"artifacts/simcse_{lang}",
            "out_dir": f"artifacts/embed_{lang}_v1",
            "batch_size": 32,
            "epochs": 3,
            "warmup_steps": 100,
        },
    }
    contrastive_path = cfg_dir / "contrastive.yaml"
    write_yaml(contrastive_path, contrastive_cfg)
    run(
        [
            sys.executable,
            "-m",
            "langembed.contrastive.train_simcse",
            "--config",
            str(contrastive_path),
            "--smoke",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "langembed.contrastive.train_simcse",
            "--config",
            str(contrastive_path),
        ]
    )

    if not args.skip_eval:
        if args.auto_label:
            print(f"=== [{lang}] 5/6 auto-label STS pairs (back-translation, no human) ===")
            n_written = generate_auto_sts(
                corpus_path,
                sts_test_path,
                lang,
                args.translate_providers,
                args.pivot_lang,
                args.translate_rpm,
                args.n_labels,
            )
            print(f"  wrote {n_written} auto-labeled STS pairs -> {sts_test_path}")
        else:
            print(f"=== [{lang}] 5/6 human-in-the-loop STS labeling + eval ===")
            run(["docker", "compose", "up", "-d", "postgres"])
            run(
                [
                    sys.executable,
                    "scripts/seed_sts_pairs.py",
                    "--config",
                    str(contrastive_path),
                    "--n",
                    str(args.n_labels),
                ]
            )

            label_port = free_port(8001)
            server = start_server("langembed.annotation.api:app", label_port)
            try:
                input(
                    f"\nLabel pairs at http://localhost:{label_port}/label (rate 1-5), "
                    "then press Enter here to continue...\n"
                )
                import httpx

                resp = httpx.get(
                    f"http://localhost:{label_port}/export-sts",
                    params={"out_path": sts_test_path},
                    timeout=30,
                )
                resp.raise_for_status()
                print(" ", resp.json())
            finally:
                stop_server(server)

        eval_cfg = {
            "language": lang,
            "spacy_model": args.spacy_model,
            "test_path": sts_test_path,
            "score_scale": 5.0,
            "retrieval_k": 5,
            "branches": {"A": f"artifacts/simcse_{lang}"},
            # STS candidates are corpus sentences by construction (active-learning sampled in
            # seed_sts_pairs.py), so they overlap the training corpus by design -- see
            # docs/ru-embeddings-report.pdf, section 3, for why train_paths must stay empty.
            "train_paths": [],
            "metrics_path": f"metrics/eval_{lang}.json",
            # Auto-labeled silver pairs have only three discrete score values (4.8/2.0/0.3),
            # a very different distribution from human 1-5 aggregate scores -- record which
            # mode produced this eval config so metrics from the two aren't compared blind.
            "label_source": "auto" if args.auto_label else "manual",
        }
        eval_path = cfg_dir / "eval.yaml"
        write_yaml(eval_path, eval_cfg)
        run([sys.executable, "-m", "langembed.eval.evaluate", "--config", str(eval_path)])

    print(f"=== [{lang}] 6/6 final embeddings + serve skew check ===")
    out_dir = args.output / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = out_dir / "embeddings.jsonl"
    run(
        [
            sys.executable,
            "scripts/embed_corpus.py",
            "--config",
            str(contrastive_path),
            "--out",
            str(embeddings_path),
        ]
    )

    serve_port = free_port(8000)
    serve_env = {
        **os.environ,
        "LANGEMBED_MODEL_DIR": f"artifacts/simcse_{lang}",
        "LANGEMBED_LANG": lang,
    }
    if args.spacy_model:
        serve_env["LANGEMBED_SPACY_MODEL"] = args.spacy_model
    server = start_server("langembed.serving.serve:app", serve_port, env=serve_env)
    try:
        run(
            [
                sys.executable,
                "scripts/verify_serve_skew.py",
                "--embeddings",
                str(embeddings_path),
                "--serve-url",
                f"http://localhost:{serve_port}",
                "--n",
                "10",
            ]
        )
    finally:
        stop_server(server)

    metrics_src = REPO_ROOT / "metrics" / f"eval_{lang}.json"
    if metrics_src.exists():
        shutil.copy(metrics_src, out_dir / "eval.json")

    print(f"\nDone. Final embeddings for '{lang}': {embeddings_path}")


if __name__ == "__main__":
    main()
