"""Tune XGBoost with Optuna, scoring each trial by validation PR-AUC.

    python -m ml.src.tune --trials 30 --timeout-min 60
    python -m ml.src.train --model xgb --params ml/params/xgb_best.json

Trials use a faster learning rate (0.1) so a laptop can try many settings; the
final model is then retrained by train.py at 0.05 with the winning settings.
Test data is never looked at here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import optuna
import pandas as pd

from ml.src.data import TARGET, load_ieee, make_synthetic, time_split
from ml.src.evaluate import pr_auc
from ml.src.features import FeatureBuilder, build_features
from ml.src.train import make_xgb

TUNE_LR = 0.1


def search_space(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 30, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.9),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 30.0, log=True),
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune XGBoost for FraudShield")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--timeout-min", type=float, default=60, help="stop after this many minutes")
    p.add_argument("--data-dir", default=os.getenv("DATA_DIR", "data/raw"))
    p.add_argument("--synthetic", action="store_true")
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.5,
        help="tune on the most recent fraction of the training period (faster trials)",
    )
    p.add_argument("--out", default="ml/params/xgb_best.json")
    p.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT", "fraudshield"))
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(args.experiment)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = make_synthetic(20_000) if args.synthetic else load_ieee(args.data_dir)
    build_features(df, inplace=True)
    split = time_split(df)
    del df
    builder = FeatureBuilder().fit(split.train)
    # Rows are in time order, so the tail is the most recent part of the training period.
    recent = split.train.iloc[int(len(split.train) * (1 - args.train_frac)) :]
    x_train, x_val = builder.transform(recent), builder.transform(split.val)
    y_train, y_val = recent[TARGET], split.val[TARGET]
    del split, recent
    print(
        f"Tuning on {len(x_train):,} recent training rows, scored on {len(x_val):,} validation rows"
    )

    def objective(trial: optuna.Trial) -> float:
        params = {**search_space(trial), "learning_rate": TUNE_LR}
        model = make_xgb(params, n_estimators=2000)
        model.set_params(early_stopping_rounds=50)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        score = pr_auc(
            y_val, model.predict_proba(x_val, iteration_range=(0, model.best_iteration + 1))[:, 1]
        )
        trial.set_user_attr("trees", model.best_iteration + 1)
        with mlflow.start_run(run_name=f"tune-{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics({"val_pr_auc": score, "trees": model.best_iteration + 1})
        print(
            f"trial {trial.number:3d}  val_pr_auc={score:.4f}  trees={model.best_iteration + 1}",
            flush=True,
        )
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    with mlflow.start_run(run_name="xgb-tuning"):
        study.optimize(objective, n_trials=args.trials, timeout=args.timeout_min * 60)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metrics({"best_val_pr_auc": study.best_value, "n_trials": len(study.trials)})
        trials = study.trials_dataframe()
        out_csv = Path(args.out).with_suffix(".trials.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        trials.to_csv(out_csv, index=False)
        mlflow.log_artifact(str(out_csv))

    best = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in study.best_params.items()}
    Path(args.out).write_text(json.dumps(best, indent=2) + "\n")
    print(
        f"\nBest val PR-AUC {study.best_value:.4f} after {len(study.trials)} trials -> {args.out}"
    )
    print(json.dumps(best, indent=2))
    return {"best_value": study.best_value, "params": best, "trials": pd.DataFrame(trials)}


if __name__ == "__main__":
    main()
