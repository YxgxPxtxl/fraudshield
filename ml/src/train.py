"""Train a model and log it to MLflow.

    python -m ml.src.train --model xgb                       # full IEEE-CIS data
    python -m ml.src.train --model xgb --params ml/params/xgb_best.json
    python -m ml.src.train --model logreg                    # week 1 baseline
    python -m ml.src.train --model xgb --synthetic           # no Kaggle data needed

Rules that keep the numbers honest:
- split by time 70/15/15; the threshold and early stopping use validation only;
- test is scored once, at the end, and never used to choose anything;
- the champion model is picked by validation PR-AUC, not test.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ml.src.data import (  # noqa: E402
    TARGET,
    Split,
    load_ieee,
    make_synthetic,
    numeric_features,
    time_split,
)
from ml.src.evaluate import best_cost_threshold, evaluate  # noqa: E402
from ml.src.features import FeatureBuilder, build_features  # noqa: E402
from ml.src.model import FraudScorer, FraudScorerPyfunc, pyfunc_output  # noqa: E402

REGISTERED_MODEL = "fraudshield"
CHAMPION_ALIAS = "champion"

DEFAULT_XGB_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
    "scale_pos_weight": 1.0,
}


def build_logreg() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]
    )


def make_xgb(params: dict, n_estimators: int = 3000, n_jobs: int = -1) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=1,
        eval_metric="aucpr",
        early_stopping_rounds=100,
        n_jobs=n_jobs,
        random_state=42,
        **params,
    )


def serving_frame(df: pd.DataFrame, builder: FeatureBuilder) -> pd.DataFrame:
    """What a caller sends the model: feature columns, numbers as float, categories as text."""
    x = df.reindex(columns=builder.columns).copy()
    for c in builder.columns:
        if c in builder.categories:
            x[c] = x[c].astype(object).where(x[c].notna(), None)
        else:
            x[c] = x[c].astype("float64")
    return x


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a FraudShield model")
    p.add_argument("--model", choices=["logreg", "xgb"], default="xgb")
    p.add_argument("--params", help="JSON file of XGBoost params (e.g. from tune.py)")
    p.add_argument("--data-dir", default=os.getenv("DATA_DIR", "data/raw"))
    p.add_argument("--nrows", type=int, default=None, help="read only the first N rows")
    p.add_argument("--synthetic", action="store_true", help="use generated data (smoke test)")
    p.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT", "fraudshield"))
    p.add_argument("--no-register", action="store_true", help="skip the MLflow model registry")
    p.add_argument(
        "--figures-dir", default="docs/figures", help="where to save plots for the README"
    )
    return p.parse_args(argv)


def load_frame(args) -> pd.DataFrame:
    return make_synthetic(20_000) if args.synthetic else load_ieee(args.data_dir, args.nrows)


def score_and_log(y_val, val_score, y_test, test_score) -> dict[str, float]:
    """Pick the threshold on validation, then report val and test with it."""
    threshold, _ = best_cost_threshold(y_val, val_score)
    metrics = {
        "threshold": threshold,
        **evaluate(y_val, val_score, threshold, prefix="val_"),
        **evaluate(y_test, test_score, threshold, prefix="test_"),
    }
    # How much worse the model gets one period later: the drift headline number.
    metrics["val_test_pr_auc_drop"] = 1 - metrics["test_pr_auc"] / metrics["val_pr_auc"]
    mlflow.log_metrics(metrics)
    return metrics


# ---------------------------------------------------------------------- logreg
def run_logreg(df: pd.DataFrame, args) -> dict[str, float]:
    split = time_split(df)
    print(split.summary().to_string(index=False))
    features = numeric_features(split.train)

    def xy(part):
        return part[features].astype("float64"), part[TARGET]

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = map(
        xy, (split.train, split.val, split.test)
    )
    model = build_logreg()
    with mlflow.start_run(run_name="logreg-baseline"):
        mlflow.log_params(
            {
                "model": "logreg",
                "data": "synthetic" if args.synthetic else "ieee-cis",
                "nrows": args.nrows or len(df),
                "n_features": len(features),
                "split": "time 70/15/15",
            }
        )
        start = time.perf_counter()
        model.fit(x_train, y_train)
        mlflow.log_metric("train_seconds", time.perf_counter() - start)
        val_score = model.predict_proba(x_val)[:, 1]
        test_score = model.predict_proba(x_test)[:, 1]
        metrics = score_and_log(y_val, val_score, y_test, test_score)
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(x_val.head(5), val_score[:5]),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
            skops_trusted_types=["numpy.dtype"],
        )
    return metrics


# ------------------------------------------------------------------------- xgb
def importance_figure(
    scorer: FraudScorer, sample: pd.DataFrame, path: Path, top: int = 20
) -> pd.DataFrame:
    """Mean |SHAP| per feature on a validation sample, as a bar chart and a table."""
    dm = xgb.DMatrix(scorer.builder.transform(sample), enable_categorical=True)
    contribs = scorer.booster.predict(dm, pred_contribs=True)[:, :-1]
    imp = (
        pd.DataFrame(
            {"feature": scorer.builder.columns, "mean_abs_shap": np.abs(contribs).mean(axis=0)}
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    head = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * top + 1))
    ax.barh(head["feature"], head["mean_abs_shap"], color="#3b6fb6")
    ax.set_xlabel("mean |SHAP value| (impact on log-odds of fraud)")
    ax.set_title(f"Top {top} features — XGBoost, validation sample")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return imp


def promote_if_better(client: MlflowClient, version: str, val_pr_auc: float) -> bool:
    """Point the 'champion' alias at this version if it beats the current one on validation."""
    try:
        current = client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS)
        best = client.get_run(current.run_id).data.metrics.get("val_pr_auc", -1.0)
    except Exception:  # no champion yet
        best = -1.0
    if val_pr_auc > best:
        client.set_registered_model_alias(REGISTERED_MODEL, CHAMPION_ALIAS, version)
        return True
    return False


def run_xgb(df: pd.DataFrame, args) -> dict[str, float]:
    params = dict(DEFAULT_XGB_PARAMS)
    if args.params:
        params.update(json.loads(Path(args.params).read_text()))

    build_features(df, inplace=True)
    split: Split = time_split(df)
    del df
    print(split.summary().to_string(index=False))

    builder = FeatureBuilder().fit(split.train)
    x_train, x_val, x_test = (builder.transform(p) for p in (split.train, split.val, split.test))
    y_train, y_val, y_test = split.train[TARGET], split.val[TARGET], split.test[TARGET]
    example = serving_frame(split.val.head(5), builder)

    with mlflow.start_run(run_name="xgb") as run:
        mlflow.log_params(
            {
                "model": "xgb",
                "data": "synthetic" if args.synthetic else "ieee-cis",
                "nrows": args.nrows or len(split.train) + len(split.val) + len(split.test),
                "n_features": len(builder.columns),
                "n_categorical": len(builder.categorical_columns),
                "split": "time 70/15/15",
                **params,
            }
        )
        start = time.perf_counter()
        model = make_xgb(params)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        mlflow.log_metrics(
            {"train_seconds": time.perf_counter() - start, "best_iteration": model.best_iteration}
        )

        booster = model.get_booster()
        booster = booster[: model.best_iteration + 1]  # keep only the trees early stopping chose
        val_score = booster.predict(xgb.DMatrix(x_val, enable_categorical=True))
        test_score = booster.predict(xgb.DMatrix(x_test, enable_categorical=True))
        metrics = score_and_log(y_val, val_score, y_test, test_score)

        scorer = FraudScorer(booster, builder, metrics["threshold"])

        # Explanations: global importance (plot + table) from a validation sample.
        sample = split.val.sample(min(20_000, len(split.val)), random_state=0)
        fig_path = Path(args.figures_dir) / "feature_importance.png"
        if args.synthetic:
            fig_path = Path(tempfile.mkdtemp()) / "feature_importance.png"
        imp = importance_figure(scorer, sample, fig_path)
        mlflow.log_artifact(str(fig_path), "explanations")
        with tempfile.TemporaryDirectory() as tmp:
            imp_path = Path(tmp) / "feature_importance.csv"
            imp.to_csv(imp_path, index=False)
            mlflow.log_artifact(str(imp_path), "explanations")

            scorer_dir = scorer.save(Path(tmp) / "scorer")
            preds = pyfunc_output(scorer, example)
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=FraudScorerPyfunc(),
                artifacts={"scorer": str(scorer_dir)},
                code_paths=[str(Path(__file__).resolve().parents[1])],  # the `ml` package
                signature=infer_signature(example, preds, params={"top_k": 3}),
                input_example=example,
            )

        registered = None
        if not args.no_register:
            client = MlflowClient()
            mv = mlflow.register_model(info.model_uri, REGISTERED_MODEL)
            promoted = promote_if_better(client, mv.version, metrics["val_pr_auc"])
            registered = f"{REGISTERED_MODEL} v{mv.version}" + (" -> champion" if promoted else "")
            mlflow.set_tag("registered_as", registered)

    print(f"\nRun {run.info.run_id}  trees={model.best_iteration + 1}")
    if registered:
        print(f"Registered: {registered}")
    print("Top features:", ", ".join(imp["feature"].head(8)))
    return metrics


def main(argv=None) -> dict[str, float]:
    args = parse_args(argv)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(args.experiment)

    df = load_frame(args)
    metrics = run_logreg(df, args) if args.model == "logreg" else run_xgb(df, args)

    print(f"\nThreshold (cost-optimal on val): {metrics['threshold']:.3f}")
    keys = (
        "val_pr_auc",
        "test_pr_auc",
        "val_test_pr_auc_drop",
        "test_recall_at_p90",
        "test_precision",
        "test_recall",
    )
    for k in keys:
        print(f"{k:>22}: {metrics[k]:.4f}")
    return metrics


if __name__ == "__main__":
    main()
