import numpy as np
import pytest

from ml.src.evaluate import (
    best_cost_threshold,
    evaluate,
    expected_cost,
    pr_auc,
    recall_at_precision,
)


def test_perfect_scores_give_perfect_metrics():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert pr_auc(y, s) == pytest.approx(1.0)
    assert recall_at_precision(y, s, 0.9) == pytest.approx(1.0)


def test_random_scores_pr_auc_near_base_rate():
    rng = np.random.default_rng(0)
    y = (rng.random(50_000) < 0.035).astype(int)
    assert pr_auc(y, rng.random(50_000)) == pytest.approx(0.035, abs=0.01)


def test_recall_at_precision_zero_when_unreachable():
    y = np.array([1, 0, 0, 0])
    s = np.array([0.1, 0.9, 0.8, 0.7])  # the fraud is ranked last
    assert recall_at_precision(y, s, 0.9) == 0.0


def test_expected_cost_counts_fn_and_fp():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0])  # 1 FN, 1 FP
    assert expected_cost(y_true, y_pred, fn_cost=100, fp_cost=5) == 105


def test_cost_threshold_flags_more_when_misses_are_expensive():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.05).astype(int)
    s = np.clip(0.3 * y + rng.normal(0.3, 0.15, 5000), 0, 1)
    t_cheap_miss, _ = best_cost_threshold(y, s, fn_cost=1, fp_cost=5)
    t_costly_miss, _ = best_cost_threshold(y, s, fn_cost=100, fp_cost=5)
    assert t_costly_miss < t_cheap_miss


def test_evaluate_prefixes_keys():
    y = np.array([0, 1, 0, 1])
    m = evaluate(y, np.array([0.1, 0.9, 0.4, 0.6]), threshold=0.5, prefix="val_")
    assert set(m) >= {"val_pr_auc", "val_recall_at_p90", "val_cost", "val_precision"}
    assert all(k.startswith("val_") for k in m)
