# ID vs STS across languages (2026-10-11)

Does the intrinsic dimension (ID) of a model's embeddings relate to its semantic-similarity quality? This is experiment 2 of [docs/intrinsic-dimension.md](../../docs/intrinsic-dimension.md).

## Setup

- **Models:** every model found in `artifacts/` for 14 languages (as, bn, en, gu, hi, kn, ml, mr, my, or, pa, ta, te, ur):
  - A0 = `simcse_<l>`, A = `embed_<l>_svd`, B = `embed_<l>_b_mling`, BT = `embed_<l>_backtranslation`;
  - plus `all-mpnet-base-v2` for English.
  - 32 language × model points in total. Branch C (LLM embedder) was skipped; it is too heavy for CPU.
- **ID:** 5,000 distinct corpus sentences per language (`data/corpus_<l>.txt`, 20–300 characters), the same sentences for all models of a language. ID is the mean ± std over 5 seeds, each using 4,000 of the 5,000 sentences. MLE, TwoNN and the MST estimator (α = 1) are reported.
- **STS-auto:** the project's own tests (`data/sts_test_<l>[_svd].jsonl`), 60 pairs per language, **auto-labelled** (SVD).
- **STS-human:** SemEval-2024 Task 1 (SemRel) human relatedness test sets: hi 968, mr 298, pa 634, te 297 and en 2,600 pairs (`data/semrel2024/`).
- **Statistics:** Spearman with a bootstrap 95% interval for every STS score. For ID vs STS: the pooled Spearman (permutation p, bootstrap interval), and within-language model pairs (sign test). The within-language view matters because each language has its own sentences, which confounds ID across languages.

Files: `run.py` (embedding, ID, STS) and `analyze.py` (tables, correlations). Results are in `results/`: `results.jsonl`, `report.md` (the full table) and `summary.json`. Run time was about 40 minutes on CPU on corpus-host.

```bash
bash scripts/docker_run_watchdog.sh id-sts 300 false experiments/id_vs_sts/run.py      # resumable
bash scripts/docker_run_watchdog.sh id-sts-analyze 20 false experiments/id_vs_sts/analyze.py
```

## Findings

1. **ID does not predict STS quality.**
   - Pooled over the 12 points with human labels: ρ = −0.29 [−0.82, +0.41], p = 0.36 (MLE). The other estimators give the same.
   - Pooled over the 32 points with the auto tests: ρ ≈ −0.05, p ≈ 0.8.
   - Within languages, a higher ID went with a higher human STS in 5 of 11 model pairs (p = 1.0).
2. **The project's auto-labelled STS tests do not measure quality.**
   - Over the 12 points that have both: ρ(auto, human) = −0.15 [−0.83, +0.60], p = 0.64.
   - Examples: Punjabi A0 scores +0.77 on the auto test but −0.20 on human labels; mpnet scores −0.03 on the auto test and +0.81 (the best) on human labels.
   - With 60 pairs the 95% interval is about ±0.25, so most auto scores cannot be told apart from 0.
   - The earlier `eval.json` numbers (−0.33…+0.12) should not be used to compare models. The SemRel sets give human labels for hi, mr, pa, te and en.
3. **ID, the participation ratio and anisotropy do detect collapsed models.**
   - `bn` A0 is degenerate: anisotropy 0.98, so all vectors are almost the same, and participation ratio 3.
   - `mr` A0 has a low ID (16.7), participation ratio 10 and poor human STS (+0.23). SVD fine-tuning (A) repaired it: ID 32.9, participation ratio 127, human STS +0.64. That is the largest STS gain in the experiment, and it came with the largest ID change.
   - So these numbers are a cheap health check after training, not a quality score.
4. **What else the table shows:**
   - In most languages A and A0 are nearly identical in both ID (difference < 1) and STS. SVD fine-tuning changed little except for mr and en.
   - In gu, BT ≈ A0 in every metric.
   - The multilingual branch B has a lower ID than A/A0 in the same language (as, or, te, by about 4–8) and, where human labels exist, a better human STS (te +0.76 vs +0.65; en +0.81 vs +0.61/+0.66).
   - English shows an ID of 20–30 on corpus sentences, against 6–14 on phrase units in the pilot. ID describes the item set as much as the model.

## Limits

- Only 12 points have human labels (5 languages), so the pooled interval is wide. A weak relationship cannot be ruled out; a strong one can.
- Only CPU-sized models were tested; branch C is missing.
- Corpus sentences differ in domain and length per language.
