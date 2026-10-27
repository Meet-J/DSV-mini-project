"""Tests for evaluation helpers, on tiny cases where the right answer is known by hand."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation import (
    best_f1_threshold, bootstrap_ci, calibration_table, classification_metrics,
    paired_bootstrap_diff, select_by_one_standard_error, threshold_table, wilson_interval,
)


def test_confusion_counts_and_metrics_by_hand():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])          # at 0.5: predictions 1, 0, 1, 0
    m = classification_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == m["recall"] == m["accuracy"] == 0.5
    assert m["flagged_share"] == 0.5


def test_threshold_is_inclusive():
    m = classification_metrics(np.array([1, 0]), np.array([0.5, 0.2]), 0.5)
    assert m["tp"] == 1                           # probability exactly equal to the threshold counts as flagged


def test_best_f1_threshold_on_separable_data():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = best_f1_threshold(y, p)
    assert 0.3 < t <= 0.7
    assert classification_metrics(y, p, t)["f1"] == 1.0


def test_threshold_table_flagged_share_never_increases_with_threshold():
    rng = np.random.default_rng(0)
    p = rng.random(500); y = (rng.random(500) < p).astype(int)
    tbl = threshold_table(y, p)
    assert tbl["customers_flagged_pct"].is_monotonic_decreasing
    assert tbl["recall"].is_monotonic_decreasing  # raising the bar can only lose churners


def test_one_standard_error_rule_prefers_simpler_model_inside_the_band():
    res = pd.DataFrame({"model": ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"],
                        "cv_mean": [0.846, 0.830, 0.842, 0.849], "cv_se": [0.005, 0.005, 0.005, 0.005]})
    chosen, why = select_by_one_standard_error(res, ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"])
    assert chosen == "Logistic Regression" and "Gradient Boosting" in why   # 0.846 >= 0.849 - 0.005


def test_one_standard_error_rule_keeps_complex_model_when_clearly_better():
    res = pd.DataFrame({"model": ["Logistic Regression", "Gradient Boosting"], "cv_mean": [0.80, 0.86], "cv_se": [0.005, 0.005]})
    chosen, _ = select_by_one_standard_error(res, ["Logistic Regression", "Gradient Boosting"])
    assert chosen == "Gradient Boosting"


def test_bootstrap_ci_is_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400); p = np.clip(y * 0.3 + rng.random(400) * 0.7, 0, 1)
    a = bootstrap_ci(y, p, roc_auc_score, n_boot=200, seed=7)
    b = bootstrap_ci(y, p, roc_auc_score, n_boot=200, seed=7)
    assert a == b and a[0] < roc_auc_score(y, p) < a[1]


def test_calibration_table_counts_and_perfectly_calibrated_case():
    rng = np.random.default_rng(2)
    p = rng.random(20000); y = (rng.random(20000) < p).astype(int)
    t = calibration_table(y, p)
    assert t["customers"].sum() == 20000
    assert np.allclose(t["mean_predicted"], t["observed_rate"], atol=0.03)


def test_paired_bootstrap_detects_a_real_difference_and_a_null_one():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 1500)
    good = np.clip(y * 0.4 + rng.random(1500) * 0.6, 0, 1)
    bad = rng.random(1500)                                          # pure noise
    mean, lo, hi = paired_bootstrap_diff(y, good, bad, roc_auc_score, n_boot=300, seed=1)
    assert lo > 0 and mean > 0.2                                    # clearly better: interval excludes 0
    mean0, lo0, hi0 = paired_bootstrap_diff(y, good, good, roc_auc_score, n_boot=300, seed=1)
    assert mean0 == lo0 == hi0 == 0.0                               # identical models: no difference


def test_precision_is_undefined_not_zero_when_nobody_is_flagged():
    tbl = threshold_table(np.array([0, 1, 1]), np.array([0.2, 0.3, 0.4]), [0.5])
    assert tbl["customers_flagged_pct"].iloc[0] == 0 and np.isnan(tbl["precision"].iloc[0])


def test_wilson_interval_known_values_and_edges():
    lo, hi = wilson_interval(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60                 # textbook value for 50/100
    lo0, hi0 = wilson_interval(0, 20)
    assert lo0 == 0.0 and 0 < hi0 < 0.2                            # stays inside [0, 1] and is not degenerate at zero successes
    assert all(np.isnan(wilson_interval(0, 0)))
