"""Tests for scoring and segmentation."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src import config
from src.feature_engineering import build_pipeline, encode_target
from src.predict import (
    DATA_DICTIONARY, oof_scores, reporting_fields, retention_segment, risk_category,
    score_new_customers, value_segment,
)


# ---------------------------------------------------------------- segmentation rules
def test_risk_category_boundaries():
    p = pd.Series([0.0, 0.2999, 0.30, 0.50, 0.70, 0.7001, 1.0])
    assert risk_category(p).tolist() == ["Low", "Low", "Medium", "Medium", "Medium", "High", "High"]


def test_value_segment_and_retention_segment():
    charges = pd.Series([50.0, 90.0, 90.0], index=["a", "b", "c"])
    value = value_segment(charges, threshold=89.85)
    assert value.tolist() == ["Standard value", "High value", "High value"]
    risk = pd.Series(["High", "High", "Low"], index=charges.index)
    active = pd.Series([True, True, False], index=charges.index)
    seg = retention_segment(risk, value, active)
    assert seg.tolist() == ["High risk + standard value", "High risk + high value", "Already churned"]


def test_data_dictionary_has_unique_columns():
    names = [n for n, _ in DATA_DICTIONARY]
    assert len(names) == len(set(names)) and all(desc.strip() for _, desc in DATA_DICTIONARY)


def test_number_of_services_by_hand():
    base = {"PhoneService": "Yes", "InternetService": "Fiber optic", "MonthlyCharges": 80.0, "tenure": 5,
            "PaymentMethod": "Mailed check", "MultipleLines": "No",
            "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "No"}
    none = {**base, "PhoneService": "No", "InternetService": "No", "MultipleLines": "No phone service",
            **{c: "No internet service" for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]}}
    out = reporting_fields(pd.DataFrame([base, none], index=["x", "y"]))
    assert out["NumberOfServices"].tolist() == [4, 0]            # phone + internet + OnlineSecurity + StreamingTV; and nothing
    assert out["ProtectiveAddOnCount"].tolist() == [1, 0]
    assert out["AutoPay"].tolist() == ["No", "No"]


# ---------------------------------------------------------------- out-of-fold scoring
@pytest.fixture(scope="module")
def small_problem():
    pytest.importorskip("shap")
    if not config.CLEAN_DATA_PATH.exists():
        pytest.skip("Run `python -m src.preprocessing` first.")
    df = pd.read_csv(config.CLEAN_DATA_PATH).set_index(config.ID_COL).iloc[:1500]
    return df.drop(columns=[config.TARGET_COL]), encode_target(df[config.TARGET_COL])


def _forest():
    return build_pipeline(RandomForestClassifier(n_estimators=40, max_depth=6, min_samples_leaf=5, random_state=config.RANDOM_SEED), scale_numeric=False)


def test_every_customer_is_scored_once_with_valid_probabilities(small_problem):
    X, y = small_problem
    s = oof_scores(_forest(), X, y, n_splits=5)
    assert s["proba"].between(0, 1).all() and s["proba"].index.equals(X.index)
    assert sorted(s["fold"].unique()) == [0, 1, 2, 3, 4] and s["contributions"].index.equals(X.index)


def test_out_of_fold_scores_are_not_in_sample_scores(small_problem):
    """If 'out-of-fold' were just a label, OOF AUC would match the (inflated) in-sample AUC."""
    X, y = small_problem
    in_sample = roc_auc_score(y, _forest().fit(X, y).predict_proba(X)[:, 1])
    oof = roc_auc_score(y, oof_scores(_forest(), X, y)["proba"])
    assert in_sample - oof > 0.02


def test_oof_is_reproducible_for_a_fixed_seed(small_problem):
    X, y = small_problem
    a, b = oof_scores(_forest(), X, y)["proba"], oof_scores(_forest(), X, y)["proba"]
    pd.testing.assert_series_equal(a, b)


# ---------------------------------------------------------------- new customers
def test_scoring_new_customers_with_the_saved_model():
    if not config.MODEL_PATH.exists() or not config.CLEAN_DATA_PATH.exists():
        pytest.skip("Run training first.")
    raw = pd.read_csv(config.CLEAN_DATA_PATH).drop(columns=[config.TARGET_COL]).head(25)
    out = score_new_customers(raw)
    assert len(out) == 25 and out.CustomerID.is_unique and out.ChurnProbability.between(0, 1).all()
    assert (risk_category(out.ChurnProbability) == out.RiskCategory).all()
    assert {"TopRiskDriver1", "ValueSegment"} <= set(out.columns)
