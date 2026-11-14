"""End-to-end smoke test: the baseline trains on synthetic data and beats random."""

from ml.src import train


def test_baseline_trains_and_beats_random(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.chdir(tmp_path)  # keep MLflow artifacts out of the repo
    metrics = train.main(["--synthetic", "--experiment", "pytest"])
    # Random PR-AUC equals the fraud rate (~3.5%); the synthetic signal should beat it clearly.
    assert metrics["test_pr_auc"] > 3 * metrics["test_fraud_rate"]
    assert 0.0 <= metrics["threshold"] <= 1.0
