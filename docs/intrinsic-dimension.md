# Intrinsic dimension of embeddings: manual

How to measure the **intrinsic dimension** (ID) of embedding sets with `langembed`, and how to read the numbers. The code is `src/langembed/eval/intrinsic_dim.py` (estimators) and `scripts/intrinsic_dims.py` (command line). Tests are in `tests/test_intrinsic_dim.py`.

## 1. What it measures and why

An embedding model maps items into vectors of a fixed length, such as 256 for branch A, 768 for B and mpnet, 1024 for C and 128 for CBOW. The points do not fill that space. They lie near a much lower-dimensional set, and the dimension of that set is the **intrinsic dimension**.

ID is useful here because it needs **no labels**. We have no gold STS data for most languages and only a few hundred triplets for fine-tuning, so we cannot rank branches by accuracy. We can, however, compare how much structure each variant uses on the same items, follow it through training, across languages and against anisotropy, and relate it to STS where STS exists.

The approach follows [whiteroomlz/intrinsic-dimensions-computation](https://github.com/whiteroomlz/intrinsic-dimensions-computation), which estimates the ID of n-gram embedding sets from minimum spanning trees on a Slurm cluster. This module reimplements its MST-length estimator, with these changes:

| Original repository | Here |
|---|---|
| MSTs from the C++ `mst-clustering` package, Slurm jobs of up to 720 h | Exact Prim MST in NumPy on one machine (about 10–25 s per run for 10k points) |
| Min-max scaling of every column, Euclidean distance | L2-normalized vectors (Euclidean between unit vectors equals cosine geometry), optionally mean-centred first |
| No random seed | Fixed seeds; every result is mean ± std over seeds |
| BQY estimator with reference tables for dimensions 2–17 only | Not ported: our IDs can exceed 17. TwoNN and MLE are added as independent cross-checks instead |
| Results sent by e-mail as `.xlsx` | CSV and Markdown files in an output folder |

## 2. Estimators

All estimators run on the same random sample of distinct, prepared vectors.

| Column | Estimator | Idea | Notes |
|---|---|---|---|
| `schweinhart_a{α}` | MST length (Schweinhart) | For n points from a d-dimensional set, `E_α(n) = Σ |e|^α` over MST edges grows like `n^((d-α)/d)`. A log-log fit over nested subsamples (10% → 100% of the sample) gives slope `s`, and `d = α / (1 - s)`. | Valid for α < d. Compare several α; agreement means a stable estimate. |
| `twonn` | TwoNN (Facco et al., 2017) | Ratio `μ = r₂/r₁` of second to first neighbour distance; `-log(1 - F(μ)) = d·log μ`. | Local and robust to density changes; the top 10% of ratios are discarded. |
| `mle` | Levina–Bickel MLE | Maximum likelihood from the k nearest neighbours, inverse-averaged (MacKay–Ghahramani), averaged over k = 10…20. | Local; tends to underestimate high IDs at small n. |
| `pca90`, `pca95` | PCA | Number of components explaining 90% / 95% of the variance. | A linear dimension, usually much larger than the ID of a curved set. Treat it as an upper reference. |
| `participation_ratio` | PCA | `(Σλ)² / Σλ²` | A smooth linear dimension. |
| anisotropy (`meta.json`, Markdown table) | Mean pairwise cosine of the raw vectors | Near 0 = isotropic; high = all vectors point one way. | If it is high, compare `unit` with `centered`. |

When the three nonlinear estimators (`schweinhart`, `twonn`, `mle`) agree within a few units, the ID is well defined. If they disagree strongly, the set is multi-scale (for example clusters of different dimension), so report the spread rather than one number.

## 3. Inputs

Each variant is a matrix with one row per item:
- a `.npy` file of shape `(n_items, dim)`, or
- a `.jsonl` file with an `embedding` field per line, such as the output of `scripts/embed_article_units_cpu.py`.

Rows must be **the same items in the same order** across variants. Pass `--align` to check that row counts match. Use **distinct items**: identical strings give identical vectors and zero distances. Duplicate vectors are removed automatically, and `n_unique_available` shows how many remained.

## 4. Running

On the host, in the `langembed-ml` image (CPU only, no GPU needed):

```bash
cd /home/s939/langembed_deploy/langembed
bash scripts/docker_run_watchdog.sh id-run 240 false scripts/intrinsic_dims.py \
    --emb A0=experiments/intrinsic_pilot/emb/A0.npy \
    --emb mpnet=experiments/intrinsic_pilot/emb/mpnet.jsonl \
    --align --out-dir experiments/intrinsic_pilot/results
```

Options:

| Option | Default | Meaning |
|---|---|---|
| `--emb NAME=PATH` | (required, repeatable) | An embedding variant. |
| `--preps` | `unit centered` | `unit`: L2-normalize. `centered`: subtract the mean vector, then L2-normalize. |
| `--seeds` | `0 1 2 3 4` | One random sample and subsample order per seed. |
| `--sample-size` | `10000` | Points per seed. Memory ≈ 3 × 8 × n² bytes (about 2.4 GB at 10k, 9.6 GB at 20k). |
| `--n-sizes` | `10` | Nested subsample sizes for the MST estimator. |
| `--alphas` | `0.5 1 2` | Exponents of the MST estimator. |
| `--align` | off | Fail if variants have different row counts. |

Locally, without Docker: `PYTHONPATH=src python scripts/intrinsic_dims.py ...` (needs numpy).

Run time is about 15–30 s per variant × prep × seed at 10k points on corpus-host. The estimators are pure NumPy and single process.

## 5. Output

`--out-dir` contains:
- `results.csv`: one row per variant × prep × seed, with every estimator.
- `summary.csv`: mean and standard deviation over seeds.
- `summary.md`: the summary as a Markdown table.
- `meta.json`: arguments, anisotropy, shapes and run time.

The seed standard deviation is the noise floor. Differences between variants smaller than about two standard deviations are not meaningful.

## 6. Caveats

- **Normalization:** projecting onto the unit sphere can lower the ID by at most 1. It is the right geometry for cosine-trained models.
- **Anisotropy:** with a high mean cosine (SimCSE `A0` ≈ 0.7), unit-normalized distances are dominated by the common direction. Compare `centered`.
- **Finite samples:** all estimators underestimate when the true ID is high and n is small. Compare variants at the **same** n, never absolute numbers across different n.
- **α for the MST estimator:** a result much larger than α is fine. If `schweinhart_a2` ≫ `schweinhart_a0.5`, the scaling regime is not reached; increase n.
- **Item set:** the ID describes the item set as much as the model. Phrase units, sentences and vocabulary give different numbers, so compare models on identical items.

## 7. Experiments this enables

1. **Branch comparison without labels:** A0/A/B/C/CBOW plus external models on the same items (pilot below).
2. **ID against STS across languages:** for languages with `eval.json` (hi, ta, gu, kn, mr, or, pa, ur), test whether ID relates to Spearman.
3. **n-gram replication of the original repository:** n-gram matrices built from CBOW or vocabulary vectors (n = 1–3) per language.
4. **Training dynamics:** ID at each stage (masked-language pretraining → SimCSE → supervised).
5. **Anisotropy:** `unit` vs `centered`, and whitening, related to semantic-graph cluster quality.
6. **Choosing vector sizes:** PCA truncation to about the ID, then measure the STS drop.
7. **MST clustering:** cutting the longest MST edges (Zahn) as an alternative to the k-NN + Louvain semantic-graph builder.

## 8. Pilot (2026-10-11)

10,243 distinct English phrase units from the 10-article branch comparison (`experiments/branches10`): A0 (`simcse_en`), A (`embed_en_svd`), B, C, CBOW, and `sentence-transformers/all-mpnet-base-v2`. 5 seeds, `unit` and `centered`. The inputs are in `experiments/intrinsic_pilot/` on corpus-host (`pilot_prepare.py` builds the aligned matrices).

Results, as mean ± std over 5 seeds at `unit` normalization. The `centered` numbers differ by less than 1 for the nonlinear estimators. The full table is in `experiments/intrinsic_pilot/results/summary.md`.

| Variant | Ambient dim | Anisotropy | TwoNN | MLE | MST α=0.5 | MST α=1 | MST α=2 | PCA 90% | Participation ratio |
|---|---|---|---|---|---|---|---|---|---|
| A0 (`simcse_en`) | 256 | 0.52 | 13.0 | 14.1 | 13.5 ± 0.9 | 13.5 ± 0.8 | 13.6 ± 0.7 | 66 | 25.2 (centered 13.6) |
| A (`embed_en_svd`) | 256 | 0.65 | 13.4 | 14.3 | 13.9 ± 0.8 | 13.9 ± 0.8 | 13.9 ± 0.7 | 65 | 15.9 |
| B (multilingual base) | 768 | 0.25 | 6.4 | 8.7 | 8.2 ± 0.1 | 8.6 ± 0.1 | 9.1 ± 0.1 | 128 | 64.7 |
| C (LLM embedder) | 1024 | 0.83 | 7.2 | 11.3 | 10.2 ± 0.4 | 10.6 ± 0.4 | 11.3 ± 0.4 | 312 | 51.4 |
| CBOW (mean word vectors) | 128 | 0.03 | 6.9 | 10.3 | 9.6 ± 1.5 | 9.9 ± 1.6 | 10.5 ± 2.0 | 106 | 98.7 |
| mpnet (`all-mpnet-base-v2`) | 768 | 0.13 | 6.1 | 7.6 | 7.9 ± 0.1 | 8.3 ± 0.2 | 9.0 ± 0.2 | 258 | 123.5 |

What the pilot shows:

- **The estimators agree.** For every variant, TwoNN, MLE and the MST estimator fall within about 3 of each other, and the MST result barely depends on α. The phrase sets have a well-defined ID of about 6–14, far below the vector lengths of 128–1024.
- **A0 and A are a group of their own, at about 13–14.** Every other variant, including the external mpnet model, sits at about 6–11. Branch A's SVD fine-tuning did not change the ID of the SimCSE base (13.5 → 13.9).
- **Linear and nonlinear structure differ completely.** A0 and A are almost flat (participation ratio 16–25, close to their ID). mpnet spreads over about 258 linear directions (90% variance) but has an ID of only about 7–8, i.e. a strongly curved low-dimensional structure. CBOW is nearly isotropic (anisotropy 0.03, participation ratio 99) with an ID of about 10.
- **Centering** matters for the linear numbers of the anisotropic variants (A0 participation ratio 25.2 → 13.6) but not for the nonlinear ID, which uses local distances only.
- **No ranking.** This is not a quality ranking. Whether a lower or higher ID goes with better STS is exactly what experiment 2 (ID against STS across languages) has to establish. That mpnet, a strong general model, has the lowest ID is a hypothesis to test, not a conclusion.

Limits of the pilot:

- There are only 10,243 distinct phrases, so each seed samples 10,000 of nearly the same points. The seed std therefore mainly reflects the MST subsample order, and TwoNN/MLE show ±0.0. Use `--sample-size` well below the item count, or more items, to get a real noise floor.
- It covers English phrase units only, one domain (10 scientific articles) and one item type.

## 9. Experiment 2: ID vs STS across languages (2026-10-11)

32 language × model points in 14 languages (A0/A/B/BT, plus mpnet for English). ID was measured on 5,000 corpus sentences per language; STS on the project's auto-labelled tests and on the human-labelled SemEval-2024 SemRel test sets (hi, mr, pa, te, en). Details and the full table: [experiments/id_vs_sts/README.md](../experiments/id_vs_sts/README.md).

- **ID does not predict STS quality.** Pooled over the human-labelled points: ρ = −0.29 [−0.82, +0.41], p = 0.36. Within languages the direction is random (5 of 11 pairs).
- **The auto-labelled 60-pair STS tests do not agree with human labels** (ρ = −0.15, p = 0.64). Use SemRel where it exists.
- **ID, participation ratio and anisotropy detect collapsed models:** `bn` A0 (anisotropy 0.98, participation ratio 3) and `mr` A0 (ID 16.7, participation ratio 10, human STS +0.23, repaired by A: ID 32.9, STS +0.64). Use them as a health check, not a quality score.
