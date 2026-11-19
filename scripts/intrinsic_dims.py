"""Compare embedding variants by intrinsic dimension (see docs/intrinsic-dimension.md).

Usage:
  python scripts/intrinsic_dims.py --emb A0=emb/A0.npy --emb mpnet=emb/mpnet.jsonl \\
      --out-dir output/intrinsic_dims/pilot [--preps unit centered] [--seeds 0 1 2 3 4] \\
      [--sample-size 10000] [--n-sizes 10] [--alphas 0.5 1 2]

Each --emb is NAME=PATH: a .npy matrix (rows = items) or a .jsonl file with an "embedding"
field per line. For a fair comparison all variants should embed the same items in the same
order (rows are matched by position; use --align to check the row counts are equal).

Writes to --out-dir:
  results.csv   one row per variant x prep x seed with every estimator
  summary.csv   mean and standard deviation over the seeds
  summary.md    the same as a Markdown table
  meta.json     arguments, anisotropy per variant, run time
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from langembed.eval.intrinsic_dim import PREPS, anisotropy, estimate_all

METRICS = (
    "twonn",
    "mle",
    "schweinhart_a0.5",
    "schweinhart_a1",
    "schweinhart_a2",
    "pca90",
    "pca95",
    "participation_ratio",
)


def load_embeddings(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line)["embedding"])
    return np.asarray(rows, dtype=np.float64)


def summarize(rows: list[dict], metrics: list[str]) -> list[dict]:
    out = []
    keys = sorted({(r["variant"], r["prep"]) for r in rows}, key=lambda k: (k[1], k[0]))
    for variant, prep in keys:
        group = [r for r in rows if r["variant"] == variant and r["prep"] == prep]
        s = {
            "variant": variant,
            "prep": prep,
            "seeds": len(group),
            "n_points": group[0]["n_points"],
            "dim_ambient": group[0]["dim_ambient"],
        }
        for m in metrics:
            vals = np.array([r[m] for r in group], dtype=np.float64)
            s[f"{m}_mean"] = float(vals.mean())
            s[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out.append(s)
    return out


def markdown(summary: list[dict], metrics: list[str], aniso: dict[str, float]) -> str:
    head = ["variant", "prep", "ambient dim", "anisotropy"] + metrics
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in summary:
        cells = [s["variant"], s["prep"], str(s["dim_ambient"]), f"{aniso[s['variant']]:.2f}"]
        for m in metrics:
            mean, std = s[f"{m}_mean"], s[f"{m}_std"]
            cells.append(f"{mean:.1f} ± {std:.1f}" if not m.startswith("pca") else f"{mean:.0f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--emb", action="append", required=True, help="NAME=PATH (.npy or .jsonl); repeatable"
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--preps", nargs="+", default=list(PREPS), choices=PREPS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument(
        "--sample-size", type=int, default=10000, help="points per seed (exact MST: O(n^2) memory)"
    )
    ap.add_argument(
        "--n-sizes", type=int, default=10, help="nested subsample sizes for the MST estimator"
    )
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument(
        "--align", action="store_true", help="fail unless every variant has the same row count"
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants: dict[str, np.ndarray] = {}
    for spec in args.emb:
        name, _, path = spec.partition("=")
        variants[name] = load_embeddings(Path(path))
        print(f"{name}: {variants[name].shape} from {path}", flush=True)
    if args.align and len({v.shape[0] for v in variants.values()}) > 1:
        raise SystemExit("--align: variants have different row counts")

    started = time.time()
    rows, aniso = [], {}
    for name, vectors in variants.items():
        aniso[name] = anisotropy(vectors)
        for prep in args.preps:
            for seed in args.seeds:
                t0 = time.time()
                res = estimate_all(
                    vectors,
                    prep=prep,
                    sample_size=args.sample_size,
                    n_sizes=args.n_sizes,
                    alphas=tuple(args.alphas),
                    seed=seed,
                )
                rows.append({"variant": name, **res})
                print(
                    f"{name} {prep} seed {seed}: twonn {res['twonn']:.1f}, mle {res['mle']:.1f}, "
                    f"mst(a=1) {res.get('schweinhart_a1', float('nan')):.1f}, pca90 {res['pca90']} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )

    metrics = [m for m in METRICS if m in rows[0]] + [
        k for k in rows[0] if k.startswith("schweinhart_") and k not in METRICS
    ]
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, metrics)
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    (out_dir / "summary.md").write_text(markdown(summary, metrics, aniso), encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "anisotropy": aniso,
                "seconds": round(time.time() - started),
                "shapes": {k: list(v.shape) for k, v in variants.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print((out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
