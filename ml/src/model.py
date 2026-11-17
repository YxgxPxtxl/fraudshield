"""The deployable FraudShield model: features in, score + decision + reasons out.

Wraps the XGBoost booster together with the fitted FeatureBuilder and the decision
threshold, so whatever loads it (the week 3 API, a notebook) gets one object that
does everything. Saved to MLflow as a pyfunc model.

Explanations are TreeSHAP values computed by XGBoost itself (``pred_contribs``):
the same numbers the ``shap`` library produces for tree models, but fast enough to
return with every prediction.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
import xgboost as xgb

from ml.src.features import FeatureBuilder


class FraudScorer:
    def __init__(self, booster: xgb.Booster, builder: FeatureBuilder, threshold: float):
        self.booster = booster
        self.builder = builder
        self.threshold = float(threshold)

    # ------------------------------------------------------------------ scoring
    def _matrix(self, features: pd.DataFrame) -> xgb.DMatrix:
        return xgb.DMatrix(self.builder.transform(features), enable_categorical=True)

    def score(self, features: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self._matrix(features))

    def explain(self, features: pd.DataFrame, top_k: int = 3) -> list[list[dict]]:
        """Top-k features pushing each transaction towards fraud (positive SHAP values)."""
        contribs = self.booster.predict(self._matrix(features), pred_contribs=True)[:, :-1]
        names = np.array(self.builder.columns)
        x = self.builder.transform(features)
        reasons = []
        for i, row in enumerate(contribs):
            top = np.argsort(row)[::-1][:top_k]
            reasons.append(
                [
                    {
                        "feature": str(names[j]),
                        "value": None if pd.isna(x.iat[i, j]) else _plain(x.iat[i, j]),
                        "contribution": round(float(row[j]), 4),
                    }
                    for j in top
                    if row[j] > 0
                ]
            )
        return reasons

    def predict(self, features: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
        scores = self.score(features)
        return pd.DataFrame(
            {
                "score": scores,
                "is_fraud": scores >= self.threshold,
                "reasons": self.explain(features, top_k),
            },
            index=features.index,
        )

    # -------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(d / "booster.json")
        meta = {"threshold": self.threshold, "builder": asdict(self.builder)}
        (d / "scorer.json").write_text(json.dumps(meta, indent=2))
        return d

    @classmethod
    def load(cls, directory: str | Path) -> FraudScorer:
        d = Path(directory)
        booster = xgb.Booster()
        booster.load_model(d / "booster.json")
        meta = json.loads((d / "scorer.json").read_text())
        return cls(booster, FeatureBuilder(**meta["builder"]), meta["threshold"])


def _plain(v):
    v = v.item() if hasattr(v, "item") else v
    return round(v, 4) if isinstance(v, float) else v


def pyfunc_output(scorer: FraudScorer, model_input: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Prediction frame with reasons as JSON text, which MLflow's schema can describe."""
    out = scorer.predict(model_input, top_k=top_k)
    out["reasons"] = out["reasons"].map(json.dumps)
    return out


class FraudScorerPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow wrapper: ``mlflow.pyfunc.load_model(uri).predict(features_df)``."""

    def load_context(self, context):
        self.scorer = FraudScorer.load(context.artifacts["scorer"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        return pyfunc_output(self.scorer, model_input, int((params or {}).get("top_k", 3)))
