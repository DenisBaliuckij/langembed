import json


def test_run_svd_eval_pass_writes_sts_and_metrics(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "svd_eval_pass", Path(__file__).resolve().parent.parent / "scripts" / "svd_eval_pass.py"
    )
    assert spec is not None and spec.loader is not None
    svd_eval_pass = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(svd_eval_pass)

    monkeypatch.setattr(svd_eval_pass, "REPO_ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "corpus_ru.txt").write_text(
        "\n".join(f"sentence {i}" for i in range(10)), encoding="utf-8"
    )

    from langembed.annotation import svd_label

    monkeypatch.setattr(
        svd_label, "build_svd_sts_pairs", lambda sentences, n, n_components: [("a", "b", 4.2)] * n
    )

    from langembed.eval import evaluate as evaluate_module

    seen_cfg = {}

    def fake_evaluate(cfg):
        seen_cfg.update(cfg)
        return {"spearman_A": 0.5, "retrieval_recall@5_A": 0.1, "retrieval_mrr@5_A": 0.2}

    monkeypatch.setattr(evaluate_module, "evaluate", fake_evaluate)

    results = svd_eval_pass.run_svd_eval_pass("ru", n_labels=5, n_components=3)

    assert results == {"spearman_A": 0.5, "retrieval_recall@5_A": 0.1, "retrieval_mrr@5_A": 0.2}
    assert seen_cfg["label_source"] == "auto"
    assert seen_cfg["label_method"] == "svd"
    assert seen_cfg["metrics_path"] == "metrics/eval_ru_svd.json"

    sts_path = tmp_path / "data" / "sts_test_ru_svd.jsonl"
    lines = sts_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    row = json.loads(lines[0])
    assert row == {"sentence_a": "a", "sentence_b": "b", "score": 4.2}

    metrics_path = tmp_path / "metrics" / "eval_ru_svd.json"
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == results

    eval_yaml_path = tmp_path / "configs" / "ru" / "eval_svd.yaml"
    assert eval_yaml_path.exists()
