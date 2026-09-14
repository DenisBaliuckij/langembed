# Design: manually-triggered full-pipeline Airflow DAG (corpus → embeddings → LLM training)

Status: approved for implementation planning
Date: 2026-08-31
Related: [[project-langembed]], [[project-sciparse-pdf-conversion]] (memory), `docs/IMPLEMENTATION_PLAN.md`

## Purpose

A single, manually-triggered Airflow DAG on `corpus-host` (172.21.128.103, existing
`apache-airflow` stack) that runs the full path from a text/document corpus to trained
embeddings and, optionally, an LLM (branch C's LoRA fine-tune) for one language per run.
The operator picks the exact combination of technologies/methods via the DAG's trigger
parameters — which corpus source, which conversion technique, which embedding branches,
which labeling method — instead of editing scripts or SSHing in by hand.

This is the first of two sub-projects. Documentation (IEEE-standard, in Russian, with
screenshots) is an explicit follow-on spec, written only after this DAG has been built,
deployed, and exercised end-to-end (see Testing plan). It is out of scope here.

## Explicitly out of scope

- The twirpx/glottolog/elp/grammarwatch/lsp scraper stack. Per explicit decision during
  design: that track exists to collect a separate, specialized grammar/dictionary corpus
  and is **not** wired into this DAG as a corpus source.
- Modifying `run_all_branches.py` or its nightly-queue caller (`all_branches_queue.sh`).
  Both stay exactly as they are for their existing use case; this DAG calls the same
  underlying per-branch scripts directly as independent Airflow tasks instead (see
  Task graph).
- Modifying `sciparse`'s own pipeline internals or the existing `pdf_conversion` DAG.
  This design adds a normalization/bridging layer in front of them.

## Architecture overview

```
resolve_corpus  (branch on `source_mode`)
 ├─ existing_text      → wait_for_corpus_size
 └─ convert_documents  → normalize_to_pdf (per source doc, per format)
                           → extract_text (conversion_method: fast | sciparse)
                              [sciparse path: register_for_conversion → wait_for_sciparse
                               → tex_to_plaintext]
                           → concatenate_to_raw_text
        ↓ (join, trigger_rule=none_failed_min_one_success)
 corpus_ready
        ↓
 shared_corpus_prep   (run_pipeline.py — always runs if >=1 branch selected)
        ↓ (fan-out; each task runs only if selected in `branches`)
 ├─ branch_a_finetune   (supervised_finetune_pass.py, default base model)
 ├─ branch_b_finetune   (supervised_finetune_pass.py --base-model <base_model_b>)
 ├─ branch_c_lora       (embed_branch_c.py)
 └─ branch_cbow         (embed_branch_cbow.py)
```

Every compute task (`shared_corpus_prep`, the 4 branch tasks, and the two conversion
paths) executes the same way: `SSHOperator` from the Airflow worker to the host itself,
running a parameterized `docker run` against the existing `langembed-ml`/`langembed-base`
images, wrapped in the same watchdog/retry/timeout shell logic already proven in
`/home/s939/all_branches_queue.sh`. No DockerOperator, no docker.sock mounted into the
worker container — see "Why SSH, not DockerOperator" below.

### Why SSH, not DockerOperator

The `airflow-worker` container currently has neither GPU passthrough nor a mounted Docker
socket (confirmed via `docker-compose.yaml` inspection), so it cannot launch GPU containers
directly. Two ways to fix that were considered:

- **Mount `/var/run/docker.sock` into `airflow-worker`, use `DockerOperator`.** More
  "Airflow-native" (logs surface directly in the UI), but grants the shared, always-on
  worker container — which also runs ~20 unrelated DAGs — root-equivalent control over the
  host's Docker daemon. Given this host has had three separate containerd/GPU-driver
  crashes in the past month ([[infra-ops-corpus-host]]), widening that blast radius wasn't
  worth it.
- **SSH out to the host, reuse the existing launch pattern (chosen).** No new daemon-level
  access is granted to the worker container. The exact `docker run --name <unique> -v ...
  langembed-ml:latest python scripts/<script>.py ...` invocation, wrapped in the
  already-hardened watchdog (kills the container if free disk/available memory drops below
  a threshold), container-name-conflict retry, and an outer `timeout`, is reused verbatim
  from `all_branches_queue.sh` — just parameterized per DAG run instead of hardcoded to
  "all 4 branches, svd only, this fixed language list."

## Parameters (`dag_run.conf`)

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `lang` | string | required | target language code |
| `source_mode` | enum | required | `existing_text` \| `convert_documents` |
| `raw_text_path` | string | `data/raw/<lang>_nllb.txt` | used when `source_mode=existing_text` |
| `source_documents` | list[string] | `[]` | file paths, used when `source_mode=convert_documents`; format is detected from the file extension (`.pdf`, `.djvu`, `.doc`/`.docx`, `.epub`/`.fb2`) |
| `conversion_method` | enum | `fast` | `fast` (langembed's own `extract_pdf_text()`) \| `sciparse` (full layout/OCR/formula/figure pipeline) — applies after format normalization, see below |
| `min_corpus_size_mb` | int | `50` | threshold for `wait_for_corpus_size` |
| `label_method` | enum | `svd` | `svd` \| `backtranslation` \| `native` |
| `branches` | list[enum] | required, ≥1 | subset of `A`, `B`, `C`, `CBOW` |
| `embed_sample_size` | int | `200` | forwarded to each branch script |
| `base_model_b` | string | `sentence-transformers/LaBSE` | branch B's `--base-model` override |
| `use_gpu` | bool | `true` | controls whether `--gpus all` is passed to every `docker run`; `false` matches `all_branches_queue.sh`'s existing precedent of deliberately going CPU-only during host incidents |
| `no_clean` | bool | `false` | forwarded as `run_pipeline.py`'s clean-output behavior |
| `timeout_conversion_minutes` | int | `60` | per document, format-normalization + text-extraction |
| `timeout_corpus_prep_minutes` | int | `240` | `shared_corpus_prep` |
| `timeout_branch_minutes` | int | `480` | each branch task (matches the existing 28800s/8h default) |

Validation (DAG-level, at trigger time): `branches` must be non-empty; `source_documents`
must be non-empty when `source_mode=convert_documents`.

## Corpus stage detail

### `existing_text` mode
`wait_for_corpus_size` polls `raw_text_path` on the host until it exists and is
`>= min_corpus_size_mb`, or times out with a clear message. No new code.

### `convert_documents` mode — two-stage conversion

**Stage 1: format normalization (source format → PDF).** New, small, independent
converters, one per format:

| Format | Tool | Status |
|---|---|---|
| PDF | — | passthrough, no-op |
| DjVu | `djvulibre` (`ddjvu`) | new |
| DOC/DOCX | LibreOffice headless (`soffice --headless --convert-to pdf`) | new — the practical Linux-server equivalent of a "print to PDF" driver; no CUPS-PDF/print-queue setup needed |
| EPUB/FB2 | `pandoc` or calibre's `ebook-convert` | new — reflowable e-book formats don't print-to-PDF cleanly, so this stays a dedicated ebook tool rather than the print-driver route |

**Stage 2: text extraction (PDF → plain text), selected by `conversion_method`:**

- `fast`: langembed's existing `extract_pdf_text()` (`src/langembed/data/extract_text.py`)
  — quick, no GPU/Ollama dependency.
- `sciparse`: registers the normalized PDF in sciparse's existing MSSQL conversion queue
  (`dbo.PdfDocuments`, `NeedsLatexConversion`), waits for the existing `pdf_conversion`
  Airflow DAG to process it, then a **new** `.tex`-to-plaintext stripper (doesn't exist
  yet — `sciparse` and `extract_text.py` are both untouched by this design) extracts clean
  prose from the resulting `.tex`. Exact stored-procedure mechanics for registering a
  document (schema/proc names) are implementation-plan detail, not resolved here.

Per-document outputs are concatenated into one raw-text file per language before
`corpus_ready`, in the same format `run_pipeline.py --raw-input` already expects.

## Branch task detail

Confirmed via CLI inspection: `supervised_finetune_pass.py`, `embed_branch_c.py`, and
`embed_branch_cbow.py` all take only `--lang` (plus their own specific params) — none take
raw text directly. They read artifacts that only exist after `run_pipeline.py` has already
run for that language. So `shared_corpus_prep` (`run_pipeline.py --lang <lang> --raw-input
<concatenated corpus> --auto-label --auto-label-method <label_method> --embed-sample-size
<embed_sample_size>`) is a hard prerequisite whenever **any** branch is selected, even if
branch A's own output isn't one of the ones requested.

Each selected branch then runs as its own SSH-launched task:
- `branch_a_finetune`: `supervised_finetune_pass.py --lang <lang> --label-method <label_method>`
- `branch_b_finetune`: same + `--base-model <base_model_b> --out-tag b_mling`
- `branch_c_lora`: `embed_branch_c.py --lang <lang> --label-method <label_method> --embed-sample-size <embed_sample_size>`
- `branch_cbow`: `embed_branch_cbow.py --lang <lang> --embed-sample-size <embed_sample_size>`

Each is independently skipped (Airflow `skipped` state, not `failed`) if not present in
`branches`.

## Execution mechanism (all compute tasks)

`SSHOperator` (new Airflow SSH provider connection, reusing the existing host automation
SSH key) runs, per task:

```
CONTAINER=<dag_run_id>-<task_id>-<attempt>   # unique per attempt, avoids the
                                               # name-conflict class all_branches_queue.sh
                                               # already had to work around
(watchdog subshell: poll free disk / available memory every 20s;
 `docker kill $CONTAINER` and exit non-zero if either drops below threshold)
timeout <task-specific minutes>m docker run --name "$CONTAINER" \
  --gpus all   # omitted entirely if use_gpu=false
  -v <repo>:/app -v /mnt/nvme-mssql:/mnt/nvme-mssql -w /app \
  langembed-ml:latest \
  python scripts/<script>.py <args>
docker rm -f "$CONTAINER"
```

Airflow-level `retries=1` with a sane `retry_delay` sits on top for transient failures
(SSH hiccup, MSSQL connection blip) — separate from the watchdog, which is for real
resource danger and is not retried; a watchdog-triggered kill surfaces as a distinct
failure reason in the task log rather than a generic non-zero exit.

`max_active_runs=1` on the DAG itself — matches the standing operational rule "never run
two embedding-pipeline jobs at once" ([[project-langembed]]).

Note (pre-existing gap, not introduced by this design): `langembed-ml:latest`, the image
`all_branches_queue.sh` and this design both depend on, does not currently exist on
`corpus-host` (only `langembed-base:latest` does — confirmed via `docker images`). It needs
rebuilding before any task can actually run; this is implementation-plan work, not a design
change.

## Testing plan (gates the documentation follow-on project)

1. **Unit tests** for the new Python pieces (format normalizers, the `.tex`-to-plaintext
   stripper) — this repo's existing `make test` / TDD convention.
2. **Cheap DAG smoke test**: `source_mode=existing_text` + a truncated tiny raw-text
   fixture + `branches=[CBOW]` only + `use_gpu=false` — validates the whole graph
   (branching, SSH execution, watchdog, skip logic, cleanup) in minutes, not hours.
3. **Per-format conversion smoke test**: one small sample file per format (PDF/DjVu/DOCX/
   EPUB) run through both `conversion_method` options, confirming each produces usable
   plain text.
4. **Full end-to-end run**: one real language, all 4 branches, both source modes exercised
   at least once.
5. **Concurrency guard check**: confirm a second trigger while a run is active is correctly
   blocked/queued by `max_active_runs=1`.

Only after all five pass does the documentation sub-project (IEEE-standard, Russian,
screenshots) begin, per explicit user decision.
