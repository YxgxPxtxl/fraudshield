"""Evaluation for imbalanced fraud detection.

Accuracy is useless here: a model that always says "not fraud" is ~96% accurate on
IEEE-CIS. We report PR-AUC, recall at a fixed precision, and a cost-based threshold.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Assumed business costs (tune later): a missed fraud costs far more than a false alarm.
DEFAULT_FN_COST = 100.0  # chargeback + lost goods
DEFAULT_FP_COST = 5.0  # analyst review time + customer friction


def pr_auc(y_true, y_score) -> float:
    """Area under the precision-recall curve (average precision)."""
    return float(average_precision_score(y_true, y_score))


def recall_at_precision(y_true, y_score, min_precision: float = 0.9) -> float:
    """Best recall achievable while keeping precision >= min_precision (0 if never reached)."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def expected_cost(y_true, y_pred, fn_cost=DEFAULT_FN_COST, fp_cost=DEFAULT_FP_COST) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fn * fn_cost + fp * fp_cost)


def best_cost_threshold(
    y_true, y_score, fn_cost=DEFAULT_FN_COST, fp_cost=DEFAULT_FP_COST, grid: int = 201
) -> tuple[float, float]:
    """Threshold that minimises total cost on (validation) data. Returns (threshold, cost)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    best_t, best_c = 0.5, float("inf")
    for t in np.linspace(0.0, 1.0, grid):
        c = expected_cost(y_true, (y_score >= t).astype(int), fn_cost, fp_cost)
        if c < best_c:
            best_t, best_c = float(t), c
    return best_t, best_c


def evaluate(y_true, y_score, threshold: float, prefix: str = "") -> dict[str, float]:
    """All headline metrics at a given threshold, keyed for MLflow logging."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    metrics = {
        "pr_auc": pr_auc(y_true, y_score),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "recall_at_p90": recall_at_precision(y_true, y_score, 0.9),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "cost": expected_cost(y_true, y_pred),
        "fraud_rate": float(y_true.mean()),
    }
    return {f"{prefix}{k}": v for k, v in metrics.items()}
