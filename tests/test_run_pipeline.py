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

    assert seen_cache_paths == ["data/backtranslation_cache_fr.jsonl"]


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
