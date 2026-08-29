import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_all_branches", Path(__file__).resolve().parent.parent / "scripts" / "run_all_branches.py"
)
assert _SPEC is not None and _SPEC.loader is not None
run_all_branches_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_all_branches_module)


def test_clean_output_dir_removes_existing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all_branches_module, "REPO_ROOT", tmp_path)
    out_dir = tmp_path / "output" / "en"
    out_dir.mkdir(parents=True)
    (out_dir / "embeddings.jsonl").write_text("stale data\n", encoding="utf-8")

    run_all_branches_module.clean_output_dir("en")

    assert not out_dir.exists()


def test_clean_output_dir_no_op_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all_branches_module, "REPO_ROOT", tmp_path)

    run_all_branches_module.clean_output_dir("en")  # must not raise


def test_run_all_branches_calls_all_four_branch_scripts_in_order(monkeypatch, tmp_path):
    monkeypatch.setattr(run_all_branches_module, "REPO_ROOT", tmp_path)

    calls = []
    monkeypatch.setattr(run_all_branches_module, "run", lambda cmd: calls.append(cmd))
    cleaned = []
    monkeypatch.setattr(
        run_all_branches_module, "clean_output_dir", lambda lang: cleaned.append(lang)
    )

    run_all_branches_module.run_all_branches("mr", ["data/raw/mr_nllb.txt"], label_method="svd")

    assert cleaned == ["mr"]
    assert len(calls) == 5

    # Branch A: run_pipeline.py
    assert "scripts/run_pipeline.py" in calls[0]
    assert "--raw-input" in calls[0] and "data/raw/mr_nllb.txt" in calls[0]
    assert "--auto-label" in calls[0]

    # Branch A: supervised fine-tune (no --base-model / --out-tag)
    assert "scripts/supervised_finetune_pass.py" in calls[1]
    assert "--label-method" in calls[1] and "svd" in calls[1]
    assert "--base-model" not in calls[1]

    # Branch B: same script, with a multilingual base model + distinct out-tag
    assert "scripts/supervised_finetune_pass.py" in calls[2]
    assert "--base-model" in calls[2] and "sentence-transformers/LaBSE" in calls[2]
    assert "--out-tag" in calls[2] and "b_mling" in calls[2]

    # Branch C
    assert "scripts/embed_branch_c.py" in calls[3]

    # CBOW
    assert "scripts/embed_branch_cbow.py" in calls[4]


def test_run_all_branches_no_clean_skips_directory_removal(monkeypatch, tmp_path):
    monkeypatch.setattr(run_all_branches_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_all_branches_module, "run", lambda cmd: None)
    cleaned = []
    monkeypatch.setattr(
        run_all_branches_module, "clean_output_dir", lambda lang: cleaned.append(lang)
    )

    run_all_branches_module.run_all_branches("mr", ["data/raw/mr_nllb.txt"], clean=False)

    assert cleaned == []


def test_cli_no_clean_flag_forwards_clean_false(monkeypatch):
    old_argv = sys.argv
    seen = {}

    def fake_run_all_branches(
        lang, raw_input, label_method="svd", clean=True, embed_sample_size=200
    ):
        seen.update(
            lang=lang,
            raw_input=raw_input,
            label_method=label_method,
            clean=clean,
            embed_sample_size=embed_sample_size,
        )

    monkeypatch.setattr(run_all_branches_module, "run_all_branches", fake_run_all_branches)
    try:
        sys.argv = ["run_all_branches.py", "--lang", "mr", "--raw-input", "x.txt", "--no-clean"]
        run_all_branches_module.main()
    finally:
        sys.argv = old_argv

    assert seen["clean"] is False
    assert seen["lang"] == "mr"
    assert seen["raw_input"] == ["x.txt"]
