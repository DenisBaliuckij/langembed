"""Per-language orchestrator: runs all 4 embedding methods back to back -- Branch A
(from-scratch encoder + supervised fine-tune, via run_pipeline.py +
supervised_finetune_pass.py), Branch B (multilingual LaBSE fine-tune, also via
supervised_finetune_pass.py), Branch C (LoRA Qwen3 embedder, embed_branch_c.py), and
CBOW (mean-pooled word vectors, embed_branch_cbow.py) -- so one invocation produces
every comparable output/<lang>/embeddings_*.jsonl file for one language.

By default also removes any previous output/<lang>/ directory before starting: the
2026-08-10 run left 40-220GB embeddings.jsonl files per language (before
embed_corpus.py's --limit fix) sitting on disk, and this orchestrator is the natural
place to make sure that never silently accumulates again on a re-run. Pass
--no-clean to keep existing output.

All 4 branches' supervised fine-tunes share one --label-method's triplets (default
svd: fully offline, and the only auto-label method that succeeded for every language
in the 2026-08-10 run) so the 4 outputs are trained on identical signal.

Usage:
    python scripts/run_all_branches.py --lang mr --raw-input data/raw/mr_nllb.txt \
        --label-method svd
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# output/<lang> is meant to be a symlink into fast NVMe storage rather than a plain
# directory on the (slow, spinning) root disk -- matches the pre-existing convention
# for every language deployed before 2026-08-13, and keeps output visible to the
# nginx download page, which only mounts this NVMe path (not output/ itself).
NVME_OUTPUT_ROOT = Path("/mnt/nvme-mssql/langembed_deploy/langembed/output")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def clean_output_dir(lang: str) -> None:
    out_dir = REPO_ROOT / "output" / lang
    if out_dir.is_symlink():
        # shutil.rmtree refuses to operate on a symlink at all (raises OSError), so
        # it must be unlinked instead; the NVMe target's own contents are cleared
        # separately if it's a real directory (see ensure_output_symlink).
        target = out_dir.resolve()
        if target.exists():
            size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            print(f"removing previous output/{lang}/ ({size / 1024**3:.2f} GB, on NVMe)")
            shutil.rmtree(target)
        out_dir.unlink()
    elif out_dir.exists():
        size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
        print(f"removing previous output/{lang}/ ({size / 1024**3:.2f} GB)")
        shutil.rmtree(out_dir)


def ensure_output_symlink(lang: str) -> None:
    """Create output/<lang> as a symlink to NVME_OUTPUT_ROOT/<lang> if it doesn't
    already exist as something (symlink or real directory). Without this, a new
    language with no pre-existing symlink fell through to run_pipeline.py's own
    `out_dir.mkdir(parents=True, exist_ok=True)`, which silently created a plain
    directory on the slow root disk -- invisible to the nginx download page and,
    at scale, one more contributor to that disk's I/O pressure."""
    out_dir = REPO_ROOT / "output" / lang
    if out_dir.is_symlink() or out_dir.exists():
        return
    nvme_target = NVME_OUTPUT_ROOT / lang
    nvme_target.mkdir(parents=True, exist_ok=True)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.symlink_to(nvme_target)
    print(f"output/{lang}/ -> {nvme_target}")


def run_all_branches(
    lang: str,
    raw_input: list[str],
    label_method: str = "svd",
    clean: bool = True,
    embed_sample_size: int = 200,
) -> None:
    if clean:
        clean_output_dir(lang)
    ensure_output_symlink(lang)

    print(f"=== [{lang}] Branch A: from-scratch pipeline ===")
    run(
        [
            sys.executable,
            "scripts/run_pipeline.py",
            "--lang",
            lang,
            "--raw-input",
            *raw_input,
            "--auto-label",
            "--auto-label-method",
            label_method,
            "--embed-sample-size",
            str(embed_sample_size),
        ]
    )

    print(f"=== [{lang}] Branch A: supervised fine-tune ({label_method}) ===")
    run(
        [
            sys.executable,
            "scripts/supervised_finetune_pass.py",
            "--lang",
            lang,
            "--label-method",
            label_method,
        ]
    )

    print(f"=== [{lang}] Branch B: multilingual (LaBSE) fine-tune ===")
    run(
        [
            sys.executable,
            "scripts/supervised_finetune_pass.py",
            "--lang",
            lang,
            "--label-method",
            label_method,
            "--base-model",
            "sentence-transformers/LaBSE",
            "--out-tag",
            "b_mling",
        ]
    )

    print(f"=== [{lang}] Branch C: LoRA LLM embedder ===")
    run(
        [
            sys.executable,
            "scripts/embed_branch_c.py",
            "--lang",
            lang,
            "--label-method",
            label_method,
            "--embed-sample-size",
            str(embed_sample_size),
        ]
    )

    print(f"=== [{lang}] CBOW ===")
    run(
        [
            sys.executable,
            "scripts/embed_branch_cbow.py",
            "--lang",
            lang,
            "--embed-sample-size",
            str(embed_sample_size),
        ]
    )

    print(f"=== [{lang}] all 4 branches complete ===")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lang", required=True)
    ap.add_argument(
        "--raw-input",
        nargs="+",
        required=True,
        help="pre-extracted plain-text corpus file(s), forwarded to run_pipeline.py --raw-input",
    )
    ap.add_argument(
        "--label-method",
        default="svd",
        choices=["svd", "backtranslation", "native"],
        help="triplet source shared by Branches A/B/C's supervised fine-tune (default: svd)",
    )
    ap.add_argument(
        "--embed-sample-size",
        type=int,
        default=200,
        help="sentences embedded per branch (forwarded as each script's own --embed-sample-size)",
    )
    ap.add_argument(
        "--no-clean",
        action="store_true",
        help="keep any existing output/<lang>/ instead of removing it first",
    )
    args = ap.parse_args()
    run_all_branches(
        args.lang,
        args.raw_input,
        label_method=args.label_method,
        clean=not args.no_clean,
        embed_sample_size=args.embed_sample_size,
    )


if __name__ == "__main__":
    main()
