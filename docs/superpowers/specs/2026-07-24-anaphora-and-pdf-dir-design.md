# Anaphora (DET) Substitution + Directory PDF Input — Design

**Date:** 2026-07-24
**Status:** Approved

## Problem

Two gaps found while reviewing the text-preparation feature:

1. **Anaphora/pronoun substitution is incomplete.** `preprocess._prepare_tokens()`
   substitutes `PRON`-tagged tokens with `pron1`, but Universal Dependencies
   splits Russian pronouns across two POS tags: substantive pronouns (он,
   она, это, мы, себя, кто, который) are `PRON`, while possessive and
   demonstrative pronouns (его, её, их, свой, этот) — traditional Russian
   grammar's "местоимения-прилагательные" — are tagged `DET`. Verified
   empirically against `ru_core_news_sm`
   (`docs/superpowers/specs/2026-07-23-text-preparation-design.md`'s sibling
   probe technique). These `DET` words are exactly the ones that do the work
   in anaphora ("его книга" = "*his* book", referring back to an earlier
   mentioned person) and are currently just lemmatized as ordinary words,
   never substituted.

2. **`run_pipeline.py --input` only accepts explicit file paths**, not a
   directory of PDFs — the user wants to point at a folder and have every
   PDF inside it (recursively) combined into one corpus for that language.

## Solution

### 1. `src/langembed/preprocess.py`

Add `"DET": "pron1"` to `_POS_TOKENS` (currently
`{"PROPN": "person1", "PRON": "pron1", "NUM": "ordinal1"}`). Same
substitution mechanism, same `_RESERVED_TOKENS` idempotency fixed-point
(`pron1` is already reserved) — a one-line map extension, not new logic.

### 2. `scripts/run_pipeline.py`

New pure helper:

```python
def resolve_pdf_inputs(paths: list[Path]) -> list[Path]:
    """Expand any directory in `paths` into the *.pdf files found inside it
    (recursively, case-insensitive extension); files pass through unchanged.
    """
```

- A directory yields `sorted({p for p in dir.rglob("*") if p.suffix.lower() == ".pdf"})`
  (case-insensitive extension, deterministic order).
- A directory that resolves to zero PDFs raises `ValueError` with a clear
  message (silently producing an empty corpus is worse than failing fast).
- **Collision guard**: after resolving, if two resolved paths share the same
  `.stem` (e.g. `foo/book.pdf` and `bar/book.pdf`), raise `ValueError` naming
  both paths — `extract_inputs()` in the same file names its extracted
  `.txt` output `{lang}_{pdf.stem}.txt`, so a stem collision would silently
  overwrite one book's extracted text with another's. This is a real risk
  the recursive-glob feature introduces (flat `--input a.pdf b.pdf` calls
  couldn't collide before, since the caller wrote each path by hand), so the
  guard is part of this fix, not scope creep.
- `main()` calls `args.input = resolve_pdf_inputs(args.input)` immediately
  after `ap.parse_args()`, before anything else uses `args.input`.

### 3. `README.md`

New instructions showing directory-based invocation, e.g.:

```bash
python scripts/run_pipeline.py --lang ru --input data/raw/my_book_collection --spacy-model ru_core_news_sm
```

with a note that every PDF found (recursively) under that directory is
combined into one corpus for the given language, mirroring the existing
`--input file1.pdf file2.pdf` flow.

## Testing

- `tests/test_preprocess.py`: extend with `DET`-tagged cases (его/её/их/свой/этот),
  verified against the real `ru_core_news_sm` model per the existing
  `requires_ru_model` pattern; confirm idempotency still holds for a
  sentence mixing PRON and DET pronouns.
- `tests/test_run_pipeline.py` (new file): unit tests for `resolve_pdf_inputs`
  using `tmp_path` — directory expansion, file passthrough, mixed
  file+directory input, recursive discovery, empty-directory error, and the
  stem-collision guard. No real PDF content needed (empty files with a
  `.pdf` name are enough, since the function only inspects paths).

## Out of scope

- Full coreference resolution (linking a pronoun to its specific antecedent
  entity) — the docx's manual pipeline and this project's existing
  `pron1`/`person1` design only ever substitute a *category* placeholder,
  never resolve *which* entity a pronoun refers to. Not requested and not
  what "anaphora" means in the docx's "Предобработка" step.
- Retraining `artifacts/{tokenizer,encoder,simcse}_ru` again — this is the
  second change to `normalize()`'s output since the last retrain; whether to
  retrain again is a follow-up decision after this lands, not part of this
  spec's acceptance.
