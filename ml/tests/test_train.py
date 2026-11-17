"""End-to-end smoke tests on synthetic data: both models train, log and beat random."""

import json

import mlflow
import pytest

from ml.src import train


@pytest.fixture
def isolated_mlflow(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.chdir(tmp_path)  # keep MLflow artifacts out of the repo
    return tmp_path


def test_baseline_trains_and_beats_random(isolated_mlflow):
    metrics = train.main(["--model", "logreg", "--synthetic", "--experiment", "pytest"])
    # Random PR-AUC equals the fraud rate (~3.5%); the synthetic signal should beat it clearly.
    assert metrics["test_pr_auc"] > 3 * metrics["test_fraud_rate"]
    assert 0.0 <= metrics["threshold"] <= 1.0


def test_xgb_trains_registers_and_serves(isolated_mlflow):
    metrics = train.main(["--model", "xgb", "--synthetic", "--experiment", "pytest"])
    assert metrics["test_pr_auc"] > 3 * metrics["test_fraud_rate"]
    assert "val_test_pr_auc_drop" in metrics

    model = mlflow.pyfunc.load_model(f"models:/{train.REGISTERED_MODEL}@{train.CHAMPION_ALIAS}")
    example = model.input_example
    out = model.predict(example, params={"top_k": 2})
    assert list(out.columns) == ["score", "is_fraud", "reasons"]
    assert out["score"].between(0, 1).all()
    for reasons in out["reasons"].map(json.loads):
        assert len(reasons) <= 2
        assert all({"feature", "value", "contribution"} <= set(r) for r in reasons)


def test_params_file_overrides_defaults(isolated_mlflow):
    params = isolated_mlflow / "p.json"
    params.write_text(json.dumps({"max_depth": 3, "learning_rate": 0.2}))
    train.main(
        [
            "--model",
            "xgb",
            "--synthetic",
            "--no-register",
            "--params",
            str(params),
            "--experiment",
            "pytest",
        ]
    )
    run = mlflow.search_runs(experiment_names=["pytest"]).iloc[0]
    assert run["params.max_depth"] == "3"
    assert run["params.learning_rate"] == "0.2"
