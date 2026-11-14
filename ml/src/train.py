"""Train a model and log it to MLflow.

Week 1: logistic regression baseline. Week 2 adds ``--model xgb``.

    python -m ml.src.train --model logreg                 # full IEEE-CIS data
    python -m ml.src.train --model logreg --nrows 100000  # quicker run
    python -m ml.src.train --model logreg --synthetic     # no Kaggle data needed
"""

from __future__ import annotations

import argparse
import os
import time

import mlflow
import mlflow.sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.src.data import TARGET, load_ieee, make_synthetic, numeric_features, time_split
from ml.src.evaluate import best_cost_threshold, evaluate


def build_logreg() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a FraudShield model")
    p.add_argument("--model", choices=["logreg"], default="logreg")
    p.add_argument("--data-dir", default=os.getenv("DATA_DIR", "data/raw"))
    p.add_argument("--nrows", type=int, default=None, help="read only the first N rows")
    p.add_argument("--synthetic", action="store_true", help="use generated data (smoke test)")
    p.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT", "fraudshield"))
    return p.parse_args(argv)


def main(argv=None) -> dict[str, float]:
    args = parse_args(argv)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(args.experiment)

    df = make_synthetic(20_000) if args.synthetic else load_ieee(args.data_dir, args.nrows)
    split = time_split(df)
    print(split.summary().to_string(index=False))

    features = numeric_features(split.train)

    # float64 everywhere so the logged model schema accepts NaN at serving time
    # (an int column in the schema would reject a missing value from the API).
    def xy(part):
        return part[features].astype("float64"), part[TARGET]

    x_train, y_train = xy(split.train)
    x_val, y_val = xy(split.val)
    x_test, y_test = xy(split.test)

    model = build_logreg()
    with mlflow.start_run(run_name=f"{args.model}-baseline"):
        mlflow.log_params(
            {
                "model": args.model,
                "data": "synthetic" if args.synthetic else "ieee-cis",
                "nrows": args.nrows or len(df),
                "n_features": len(features),
                "split": "time 70/15/15",
                "class_weight": "balanced",
            }
        )

        start = time.perf_counter()
        model.fit(x_train, y_train)
        mlflow.log_metric("train_seconds", time.perf_counter() - start)

        # Threshold is chosen on validation only; test is touched once, at the end.
        val_score = model.predict_proba(x_val)[:, 1]
        threshold, _ = best_cost_threshold(y_val, val_score)
        test_score = model.predict_proba(x_test)[:, 1]

        metrics = {
            "threshold": threshold,
            **evaluate(y_val, val_score, threshold, prefix="val_"),
            **evaluate(y_test, test_score, threshold, prefix="test_"),
        }
        mlflow.log_metrics(metrics)
        # skops is MLflow's safe (non-pickle) format; numpy dtypes inside the pipeline
        # must be marked trusted explicitly.
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=x_val.head(5),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
            skops_trusted_types=["numpy.dtype"],
        )

    print(f"\nThreshold (cost-optimal on val): {threshold:.3f}")
    for k in ("val_pr_auc", "test_pr_auc", "test_recall_at_p90", "test_precision", "test_recall"):
        print(f"{k:>20}: {metrics[k]:.4f}")
    return metrics


if __name__ == "__main__":
    main()
