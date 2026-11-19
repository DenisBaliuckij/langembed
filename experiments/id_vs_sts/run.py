# ruff: noqa: E501 -- report/table strings
"""ID vs STS across languages (2026-10-11): does the intrinsic dimension of a model's embeddings
relate to its STS quality? See docs/intrinsic-dimension.md (experiment 2) and README.md here.

For every language x model found in artifacts/ (A0 = simcse_<l>, A = embed_<l>_svd,
B = embed_<l>_b_mling, BT = embed_<l>_backtranslation; plus all-mpnet-base-v2 for English):
  1. embed a fixed sample of corpus sentences (data/corpus_<l>.txt, the same sample for all
     models of a language) and estimate the intrinsic dimension (5 seeds, real subsampling);
  2. STS-auto: Spearman on the project's 60-pair auto-labelled test (data/sts_test_<l>[_svd].jsonl),
     with a bootstrap 95% interval;
  3. STS-human: Spearman on SemEval-2024 Task 1 (SemRel) human relatedness test sets for
     hi, mr, pa, te, en (data/semrel2024/<iso3>_test.parquet), with a bootstrap 95% interval.
One JSON line per language x model is appended to results.jsonl (finished pairs are skipped on
re-run). Run in langembed-ml on CPU:
  python experiments/id_vs_sts/run.py [--langs hi mr] [--items 5000] [--out experiments/id_vs_sts]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from langembed.eval.intrinsic_dim import anisotropy, estimate_all

LANGS = ["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "my", "or", "pa", "ta", "te", "ur"]
SEMREL = {"hi": "hin", "mr": "mar", "pa": "pan", "te": "tel", "en": "eng"}
MODELS = {
    "A0": "artifacts/simcse_{l}",
    "A": "artifacts/embed_{l}_svd",
    "B": "artifacts/embed_{l}_b_mling",
    "BT": "artifacts/embed_{l}_backtranslation",
}
EXTERNAL = {("en", "mpnet"): "data/colleague_articles_20261011/models/all-mpnet-base-v2"}


def sample_items(lang: str, n: int, out: Path, seed: int = 0) -> list[str]:
    """Reservoir sample of n distinct corpus lines of 20-300 characters, cached per language."""
    cache = out / "items" / f"{lang}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8").split("\n")
    rng = random.Random(seed)
    seen: set[str] = set()
    sample: list[str] = []
    k = 0
    with open(f"data/corpus_{lang}.txt", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = " ".join(line.split())
            if not 20 <= len(s) <= 300 or s in seen:
                continue
            seen.add(s)
            k += 1
            if len(sample) < n:
                sample.append(s)
            else:
                j = rng.randrange(k)
                if j < n:
                    sample[j] = s
            if len(seen) > 2_000_000:  # bound memory on huge corpora
                seen.clear()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(sample), encoding="utf-8")
    return sample


def sts_pairs(lang: str) -> dict[str, tuple[list[str], list[str], np.ndarray]]:
    sets = {}
    for path in (Path(f"data/sts_test_{lang}.jsonl"), Path(f"data/sts_test_{lang}_svd.jsonl")):
        if path.exists():
            rows = [
                json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()
            ]
            sets["auto"] = (
                [r["sentence_a"] for r in rows],
                [r["sentence_b"] for r in rows],
                np.array([float(r["score"]) for r in rows]),
            )
            break
    if lang in SEMREL:
        import pandas as pd

        d = pd.read_parquet(f"data/semrel2024/{SEMREL[lang]}_test.parquet")
        strip = lambda s: str(s).strip().strip('"')  # noqa: E731
        sets["human"] = (
            [strip(s) for s in d.sentence1],
            [strip(s) for s in d.sentence2],
            d.label.to_numpy(dtype=float),
        )
    return sets


def spearman_ci(pred: np.ndarray, gold: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    rho = float(spearmanr(pred, gold).statistic)
    rng = np.random.default_rng(seed)
    n = len(gold)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = spearmanr(pred[idx], gold[idx]).statistic
        if np.isfinite(r):
            boots.append(r)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"rho": rho, "ci_low": float(lo), "ci_high": float(hi), "n_pairs": n}


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a * b, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=LANGS)
    ap.add_argument("--items", type=int, default=5000)
    ap.add_argument("--id-sample", type=int, default=4000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="experiments/id_vs_sts")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["lang"], r["variant"]))

    for lang in args.langs:
        jobs = [(v, p.format(l=lang)) for v, p in MODELS.items() if Path(p.format(l=lang)).exists()]
        jobs += [(v, p) for (lg, v), p in EXTERNAL.items() if lg == lang and Path(p).exists()]
        jobs = [(v, p) for v, p in jobs if (lang, v) not in done]
        if not jobs:
            continue
        items = sample_items(lang, args.items, out)
        sets = sts_pairs(lang)
        for variant, model_dir in jobs:
            t0 = time.time()
            model = SentenceTransformer(model_dir, device="cpu")
            emb = model.encode(items, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
            row = {
                "lang": lang,
                "variant": variant,
                "model": model_dir,
                "n_items": len(items),
                "dim_ambient": int(emb.shape[1]),
                "anisotropy": anisotropy(emb),
            }
            ids = [
                estimate_all(
                    emb, prep="unit", sample_size=args.id_sample, n_sizes=8, alphas=(1.0,), seed=s
                )
                for s in args.seeds
            ]
            for key in ("twonn", "mle", "schweinhart_a1", "participation_ratio", "pca90"):
                vals = np.array([r[key] for r in ids], dtype=float)
                row[f"{key}_mean"] = float(vals.mean())
                row[f"{key}_std"] = float(vals.std(ddof=1))
            for name, (sa, sb, gold) in sets.items():
                ea = model.encode(sa, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
                eb = model.encode(sb, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
                for k, v in spearman_ci(cosine(ea, eb), gold).items():
                    row[f"sts_{name}_{k}"] = v
            row["seconds"] = round(time.time() - t0)
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"{lang} {variant}: ID twonn {row['twonn_mean']:.1f} mle {row['mle_mean']:.1f} "
                f"mst {row['schweinhart_a1_mean']:.1f} | STS auto {row.get('sts_auto_rho', float('nan')):+.2f} "
                f"human {row.get('sts_human_rho', float('nan')):+.2f} ({row['seconds']}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
