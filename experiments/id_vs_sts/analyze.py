# ruff: noqa: E501 -- report/table strings
"""Analysis of experiments/id_vs_sts/results.jsonl -> report.md (tables + correlations).

Two views, because intrinsic dimension (ID) across languages is confounded by the item set
(each language has its own sentences):
  - pooled: Spearman correlation between ID and STS over all language x model points, with a
    permutation p-value and a bootstrap interval over points;
  - within language: for every pair of models of the same language, does the model with the
    higher ID also have the higher STS? (sign agreement, two-sided sign test).
Run: python experiments/id_vs_sts/analyze.py [--dir experiments/id_vs_sts]
"""

from __future__ import annotations

import argparse
import itertools
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ID_KEYS = ("mle_mean", "twonn_mean", "schweinhart_a1_mean")


def pooled(
    points: list[dict], id_key: str, sts_key: str, n_perm: int = 10000, seed: int = 0
) -> dict:
    x = np.array([p[id_key] for p in points])
    y = np.array([p[sts_key] for p in points])
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    perm = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(n_perm)])
    p = float((np.sum(np.abs(perm) >= abs(rho)) + 1) / (n_perm + 1))
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(x), len(x))
        r = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(r):
            boots.append(r)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan"))
    return {"n": len(points), "rho": rho, "p_perm": p, "ci_low": float(lo), "ci_high": float(hi)}


def within_language(rows: list[dict], id_key: str, sts_key: str) -> dict:
    agree = disagree = 0
    pairs = []
    by_lang: dict[str, list[dict]] = {}
    for r in rows:
        if sts_key in r:
            by_lang.setdefault(r["lang"], []).append(r)
    for lang, rs in by_lang.items():
        for a, b in itertools.combinations(rs, 2):
            d_id = a[id_key] - b[id_key]
            d_sts = a[sts_key] - b[sts_key]
            if d_id == 0 or d_sts == 0:
                continue
            same = (d_id > 0) == (d_sts > 0)
            agree += same
            disagree += not same
            pairs.append(
                f"{lang} {a['variant']}-{b['variant']}: ΔID {d_id:+.1f}, ΔSTS {d_sts:+.2f}"
            )
    n = agree + disagree
    k = min(agree, disagree)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else float("nan")
    return {"pairs": n, "agree": agree, "p_sign": p, "detail": pairs}


def fmt_ci(r: dict, prefix: str) -> str:
    if f"{prefix}_rho" not in r:
        return "—"
    return (
        f"{r[f'{prefix}_rho']:+.2f} [{r[f'{prefix}_ci_low']:+.2f}, {r[f'{prefix}_ci_high']:+.2f}]"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/id_vs_sts")
    args = ap.parse_args()
    d = Path(args.dir)
    rows = [
        json.loads(x)
        for x in (d / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    rows.sort(key=lambda r: (r["lang"], r["variant"]))

    lines = [
        "# ID vs STS across languages — results",
        "",
        f"{len(rows)} language × model points, {len({r['lang'] for r in rows})} languages. "
        "ID = mean over 5 seeds (4,000 of 5,000 corpus sentences each); STS = Spearman with a "
        "bootstrap 95% interval. STS-auto: the project's 60-pair auto-labelled tests; STS-human: "
        "SemEval-2024 Task 1 (SemRel) test sets.",
        "",
        "| Lang | Model | Dim | Anisotropy | ID MLE | ID TwoNN | ID MST | PR | STS-auto (60 pairs) | STS-human (SemRel) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['lang']} | {r['variant']} | {r['dim_ambient']} | {r['anisotropy']:.2f} | "
            f"{r['mle_mean']:.1f} ± {r['mle_std']:.1f} | {r['twonn_mean']:.1f} ± {r['twonn_std']:.1f} | "
            f"{r['schweinhart_a1_mean']:.1f} ± {r['schweinhart_a1_std']:.1f} | {r['participation_ratio_mean']:.0f} | "
            f"{fmt_ci(r, 'sts_auto')} | {fmt_ci(r, 'sts_human')} |"
        )

    lines += [
        "",
        "## Pooled correlation (ID vs STS over all points)",
        "",
        "| ID estimator | STS | Points | Spearman ρ | 95% interval | Permutation p |",
        "|---|---|---|---|---|---|",
    ]
    summary = {}
    for sts_key, label in (("sts_human_rho", "human"), ("sts_auto_rho", "auto")):
        pts = [r for r in rows if sts_key in r]
        for id_key in ID_KEYS:
            if len(pts) < 4:
                continue
            res = pooled(pts, id_key, sts_key)
            summary[f"{id_key}~{label}"] = res
            lines.append(
                f"| {id_key.replace('_mean', '')} | {label} | {res['n']} | {res['rho']:+.2f} | "
                f"[{res['ci_low']:+.2f}, {res['ci_high']:+.2f}] | {res['p_perm']:.3f} |"
            )

    both = [r for r in rows if "sts_human_rho" in r and "sts_auto_rho" in r]
    if len(both) >= 4:
        res = pooled(both, "sts_auto_rho", "sts_human_rho")
        summary["sts_auto~sts_human"] = res
        lines += [
            "",
            "## Do the auto-labelled tests agree with human labels?",
            "",
            f"Spearman between STS-auto and STS-human over the {res['n']} points that have both: "
            f"ρ = {res['rho']:+.2f} [{res['ci_low']:+.2f}, {res['ci_high']:+.2f}], permutation p = {res['p_perm']:.3f}.",
        ]

    lines += [
        "",
        "## Within-language pairs (higher ID ⇒ higher STS?)",
        "",
        "| ID estimator | STS | Pairs | Same direction | Sign-test p |",
        "|---|---|---|---|---|",
    ]
    details = {}
    for sts_key, label in (("sts_human_rho", "human"), ("sts_auto_rho", "auto")):
        for id_key in ID_KEYS:
            w = within_language(rows, id_key, sts_key)
            details[f"{id_key}~{label}"] = w["detail"]
            if w["pairs"]:
                lines.append(
                    f"| {id_key.replace('_mean', '')} | {label} | {w['pairs']} | {w['agree']} | {w['p_sign']:.3f} |"
                )
    lines += ["", "Pairs (MLE ID, human STS):", ""] + [
        f"- {p}" for p in details.get("mle_mean~human", [])
    ]
    (d / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "summary.json").write_text(
        json.dumps({"pooled": summary, "within": details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
