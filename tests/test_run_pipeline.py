import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_pipeline", Path(__file__).resolve().parent.parent / "scripts" / "run_pipeline.py"
)
assert _SPEC is not None and _SPEC.loader is not None
run_pipeline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_pipeline)


def test_generate_auto_sts_writes_pairs(monkeypatch, tmp_path):
    from langembed.annotation import auto_label

    monkeypatch.setattr(auto_label, "back_translate", lambda *a, **k: "PARA:" + a[0])

    corpus = tmp_path / "corpus_ru.txt"
    corpus.write_text("\n".join(f"sentence {i}" for i in range(12)), encoding="utf-8")
    sts_out = tmp_path / "sts_test_ru.jsonl"

    n = run_pipeline.generate_auto_sts(str(corpus), str(sts_out), "ru", ["google"], "en", 1000.0, 9)

    assert n == 9
    lines = sts_out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 9
    row = json.loads(lines[0])
    assert set(row.keys()) == {"sentence_a", "sentence_b", "score"}


def test_generate_auto_sts_uses_per_language_cache_path(monkeypatch, tmp_path):
    from langembed.annotation import auto_label

    seen_cache_paths = []

    def fake_build(*args, **kwargs):
        seen_cache_paths.append(kwargs["cache_path"])
        return []

    monkeypatch.setattr(auto_label, "build_auto_sts_pairs", fake_build)

    corpus = tmp_path / "corpus_fr.txt"
    corpus.write_text("a\nb\n", encoding="utf-8")
    sts_out = tmp_path / "sts_test_fr.jsonl"

    run_pipeline.generate_auto_sts(str(corpus), str(sts_out), "fr", ["google"], "en", 20.0, 6)

    # Cache path is anchored to REPO_ROOT, like every other path in this file, so the
    # pipeline behaves the same regardless of the process's CWD.
    assert seen_cache_paths == [run_pipeline.REPO_ROOT / "data" / "backtranslation_cache_fr.jsonl"]


def test_generate_auto_sts_resolves_relative_paths_against_repo_root(monkeypatch, tmp_path):
    """corpus_path/sts_test_path passed as relative strings must resolve against
    REPO_ROOT, not the process's current working directory -- otherwise the pipeline
    breaks when invoked from a different CWD. Fakes REPO_ROOT itself (a fresh tmp_path)
    so this can't touch the real repository, then runs from an unrelated CWD to prove
    resolution doesn't depend on it."""
    from langembed.annotation import auto_label

    monkeypatch.setattr(auto_label, "back_translate", lambda *a, **k: "PARA:" + a[0])

    fake_repo_root = tmp_path / "fake_repo"
    (fake_repo_root / "data").mkdir(parents=True)
    (fake_repo_root / "data" / "corpus_de.txt").write_text(
        "\n".join(f"sentence {i}" for i in range(12)), encoding="utf-8"
    )
    monkeypatch.setattr(run_pipeline, "REPO_ROOT", fake_repo_root)

    other_cwd = tmp_path / "somewhere_else"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    n = run_pipeline.generate_auto_sts(
        "data/corpus_de.txt", "data/sts_test_de.jsonl", "de", ["google"], "en", 1000.0, 9
    )

    assert n == 9
    assert (fake_repo_root / "data" / "sts_test_de.jsonl").exists()


def test_generate_auto_sts_reports_tier_counts(monkeypatch, tmp_path, capsys):
    from langembed.annotation import auto_label

    monkeypatch.setattr(auto_label, "back_translate", lambda *a, **k: "PARA:" + a[0])

    corpus = tmp_path / "corpus_ru.txt"
    corpus.write_text("\n".join(f"sentence {i}" for i in range(12)), encoding="utf-8")
    sts_out = tmp_path / "sts_test_ru.jsonl"

    run_pipeline.generate_auto_sts(str(corpus), str(sts_out), "ru", ["google"], "en", 1000.0, 9)

    out = capsys.readouterr().out
    assert "tiers: paraphrase=" in out
    assert "adjacent=" in out
    assert "random=" in out


def test_generate_auto_sts_discloses_external_mt_services(monkeypatch, tmp_path, capsys):
    from langembed.annotation import auto_label

    monkeypatch.setattr(auto_label, "back_translate", lambda *a, **k: "PARA:" + a[0])

    corpus = tmp_path / "corpus_ru.txt"
    corpus.write_text("\n".join(f"sentence {i}" for i in range(12)), encoding="utf-8")
    sts_out = tmp_path / "sts_test_ru.jsonl"

    run_pipeline.generate_auto_sts(
        str(corpus), str(sts_out), "ru", ["google", "mymemory"], "en", 1000.0, 9
    )

    out = capsys.readouterr().out
    assert "external MT services" in out
    assert "google" in out and "mymemory" in out


def test_generate_svd_sts_writes_pairs(tmp_path):
    corpus = tmp_path / "corpus_ru.txt"
    corpus.write_text(
        "\n".join(f"sentence about topic {i % 4} number {i}" for i in range(20)), encoding="utf-8"
    )
    sts_out = tmp_path / "sts_test_ru.jsonl"

    n = run_pipeline.generate_svd_sts(str(corpus), str(sts_out), 3, 9)

    assert n == 9
    lines = sts_out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 9
    row = json.loads(lines[0])
    assert set(row.keys()) == {"sentence_a", "sentence_b", "score"}


def test_generate_svd_sts_resolves_relative_paths_against_repo_root(monkeypatch, tmp_path):
    fake_repo_root = tmp_path / "fake_repo"
    (fake_repo_root / "data").mkdir(parents=True)
    (fake_repo_root / "data" / "corpus_de.txt").write_text(
        "\n".join(f"sentence about topic {i % 4} number {i}" for i in range(20)), encoding="utf-8"
    )
    monkeypatch.setattr(run_pipeline, "REPO_ROOT", fake_repo_root)

    other_cwd = tmp_path / "somewhere_else"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    n = run_pipeline.generate_svd_sts("data/corpus_de.txt", "data/sts_test_de.jsonl", 3, 9)

    assert n == 9
    assert (fake_repo_root / "data" / "sts_test_de.jsonl").exists()


def test_svd_label_cli_flag_defaults():
    ap = run_pipeline.build_arg_parser()
    args = ap.parse_args(["--lang", "ru", "--input", "book.pdf"])

    assert args.auto_label_method == "backtranslation"
    assert args.svd_components == 100


def test_svd_label_cli_flag_set():
    ap = run_pipeline.build_arg_parser()
    args = ap.parse_args(
        [
            "--lang",
            "ru",
            "--input",
            "book.pdf",
            "--auto-label",
            "--auto-label-method",
            "svd",
            "--svd-components",
            "50",
        ]
    )

    assert args.auto_label_method == "svd"
    assert args.svd_components == 50


def test_auto_label_method_rejects_unknown_value():
    ap = run_pipeline.build_arg_parser()
    import pytest

    with pytest.raises(SystemExit):
        ap.parse_args(["--lang", "ru", "--input", "book.pdf", "--auto-label-method", "bogus"])


def test_eval_cfg_records_label_method():
    """Same source-text-check approach as test_eval_cfg_records_label_source (main() runs
    a long unmocked subprocess pipeline with no test coverage by design)."""
    source = (Path(__file__).resolve().parent.parent / "scripts" / "run_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert '"label_method": args.auto_label_method if args.auto_label else None' in source


def test_main_branches_to_svd_when_method_is_svd():
    """Same source-text-check approach as test_eval_cfg_records_label_source."""
    source = (Path(__file__).resolve().parent.parent / "scripts" / "run_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert 'args.auto_label and args.auto_label_method == "svd"' in source
    assert "generate_svd_sts(" in source


def test_auto_label_help_text_discloses_external_services():
    ap = run_pipeline.build_arg_parser()
    assert "translation services" in ap.format_help()


def test_eval_cfg_records_label_source():
    """`main()` runs a long unmocked subprocess pipeline with no test coverage by design
    (see docs/superpowers/plans/2026-08-05-automated-sts-labeling.md); this checks the
    source text directly for the label_source field rather than invoking main()."""
    source = (Path(__file__).resolve().parent.parent / "scripts" / "run_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert '"label_source": "auto" if args.auto_label else "manual"' in source


def test_auto_label_cli_flag_defaults():
    ap = run_pipeline.build_arg_parser()
    args = ap.parse_args(["--lang", "ru", "--input", "book.pdf"])

    assert args.auto_label is False
    assert args.translate_providers == ["google", "mymemory"]
    assert args.pivot_lang == "en"
    assert args.translate_rpm == 20.0


def test_auto_label_cli_flag_set():
    ap = run_pipeline.build_arg_parser()
    args = ap.parse_args(
        ["--lang", "ru", "--input", "book.pdf", "--auto-label", "--pivot-lang", "hi"]
    )

    assert args.auto_label is True
    assert args.pivot_lang == "hi"


def test_resolve_raw_text_inputs_absolute_paths(tmp_path):
    f1 = tmp_path / "corpus1.txt"
    f2 = tmp_path / "corpus2.txt"
    f1.write_text("a\n", encoding="utf-8")
    f2.write_text("b\n", encoding="utf-8")

    resolved = run_pipeline.resolve_raw_text_inputs([f1, f2])

    assert resolved == [str(f1), str(f2)]


def test_resolve_raw_text_inputs_relative_path_resolves_against_repo_root(monkeypatch, tmp_path):
    fake_repo_root = tmp_path / "fake_repo"
    (fake_repo_root / "data" / "raw").mkdir(parents=True)
    corpus = fake_repo_root / "data" / "raw" / "corpus.txt"
    corpus.write_text("a\n", encoding="utf-8")
    monkeypatch.setattr(run_pipeline, "REPO_ROOT", fake_repo_root)

    resolved = run_pipeline.resolve_raw_text_inputs([Path("data/raw/corpus.txt")])

    assert resolved == [str(corpus)]


def test_resolve_raw_text_inputs_missing_file_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="does_not_exist.txt"):
        run_pipeline.resolve_raw_text_inputs([tmp_path / "does_not_exist.txt"])


def test_input_and_raw_input_are_mutually_exclusive_and_required():
    ap = run_pipeline.build_arg_parser()

    # neither given -> args parse fine (argparse level), the exactly-one check lives in main()
    args_neither = ap.parse_args(["--lang", "ru"])
    assert args_neither.input is None
    assert args_neither.raw_input is None

    args_raw = ap.parse_args(["--lang", "ru", "--raw-input", "corpus.txt"])
    assert args_raw.raw_input == [Path("corpus.txt")]
    assert args_raw.input is None
