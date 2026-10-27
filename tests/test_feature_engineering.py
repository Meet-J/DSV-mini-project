"""Tests for the feature pipeline. The leakage tests are the important ones."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import config
from src.feature_engineering import (
    add_engineered_features, build_pipeline, build_preprocessor,
    encode_target, make_train_test_split,
)


@pytest.fixture(scope="module")
def clean_df():
    if not config.CLEAN_DATA_PATH.exists():
        pytest.skip("Run `python -m src.preprocessing` first.")
    return pd.read_csv(config.CLEAN_DATA_PATH)


def _toy_row(**overrides):
    row = {
        "tenure": 5, "PaymentMethod": "Electronic check", "MultipleLines": "No phone service",
        "OnlineSecurity": "No internet service", "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service", "TechSupport": "No internet service",
        "StreamingTV": "No internet service", "StreamingMovies": "No internet service",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------- stateless features
def test_collapses_redundant_levels_and_counts_protective_addons():
    df = pd.DataFrame([
        _toy_row(),                                               # no phone, no internet
        _toy_row(OnlineSecurity="Yes", TechSupport="Yes", StreamingTV="Yes",
                 OnlineBackup="No", DeviceProtection="No", StreamingMovies="No"),
    ])
    out = add_engineered_features(df)
    assert out["MultipleLines"].tolist() == ["No", "No"]
    assert (out["OnlineSecurity"] != "No internet service").all()
    # Streaming services must NOT count as protective add-ons.
    assert out["ProtectiveAddOnCount"].tolist() == [0, 2]


def test_autopay_flag():
    df = pd.DataFrame([_toy_row(PaymentMethod=p) for p in
                       ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]])
    assert add_engineered_features(df)["AutoPay"].tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize("tenure,group", [(0, "0-12"), (12, "0-12"), (13, "13-24"), (24, "13-24"),
                                          (25, "25-48"), (48, "25-48"), (49, "49+"), (72, "49+"), (80, "49+")])
def test_tenure_group_edges(tenure, group):
    out = add_engineered_features(pd.DataFrame([_toy_row(tenure=tenure)]))
    assert out["TenureGroup"].iloc[0] == group   # tenure 80 (outside training range) must not become NaN


def test_input_is_not_modified():
    df = pd.DataFrame([_toy_row()])
    before = df.copy()
    add_engineered_features(df)
    pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------------------- split
def test_split_is_stratified_reproducible_and_disjoint(clean_df):
    X_tr, X_te, y_tr, y_te = make_train_test_split(clean_df)
    X_tr2, X_te2, _, _ = make_train_test_split(clean_df)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert len(X_tr) + len(X_te) == len(clean_df)
    assert abs(y_tr.mean() - y_te.mean()) < 0.005          # stratified: same churn rate in both parts
    assert X_tr.index.equals(X_tr2.index) and X_te.index.equals(X_te2.index)   # same seed, same split


def test_features_contain_neither_target_nor_id(clean_df):
    X_tr, _, _, _ = make_train_test_split(clean_df)
    assert config.TARGET_COL not in X_tr.columns and config.ID_COL not in X_tr.columns
    assert X_tr.index.name == config.ID_COL     # the ID is kept only as the index, for scoring later


def test_target_encoding_rejects_unexpected_labels():
    assert encode_target(pd.Series(["Yes", "No", "No"])).tolist() == [1, 0, 0]
    with pytest.raises(ValueError):
        encode_target(pd.Series(["Yes", "Maybe"]))


# ---------------------------------------------------------------- leakage
def test_preprocessor_learns_only_from_training_rows(clean_df):
    """The scaler's mean must equal the TRAIN mean, not the full-data mean."""
    X_tr, X_te, _, _ = make_train_test_split(clean_df)
    pipe = build_pipeline(LogisticRegression())
    engineered_train = add_engineered_features(X_tr)
    pre = pipe.named_steps["preprocess"].fit(engineered_train)
    scaler = pre.named_transformers_["numeric"]

    assert scaler.n_samples_seen_ == len(X_tr)
    np.testing.assert_allclose(scaler.mean_[0], X_tr["tenure"].mean())
    # Guard against the test being vacuous: the full-data mean must actually differ.
    assert not np.isclose(X_tr["tenure"].mean(), clean_df["tenure"].mean())

    before = scaler.mean_.copy()
    pre.transform(add_engineered_features(X_te))            # transforming test data ...
    np.testing.assert_array_equal(scaler.mean_, before)      # ... must not change what was learned


def test_unseen_category_does_not_crash_and_encodes_as_zeros(clean_df):
    X_tr, X_te, y_tr, _ = make_train_test_split(clean_df)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    odd = X_te.head(3).copy()
    odd["PaymentMethod"] = "Cryptocurrency"                  # never seen in training
    odd["tenure"] = 80                                       # outside the training range
    with pytest.warns(UserWarning, match="unknown categories"):   # sklearn tells us it handled it
        proba = pipe.predict_proba(odd)[:, 1]
    assert np.isfinite(proba).all() and ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_output_columns_exclude_id_and_target(clean_df):
    X_tr, _, y_tr, _ = make_train_test_split(clean_df)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    assert not any(n.startswith(("Churn", "customerID")) for n in names)
    # Default feature set: no engineered features, TotalCharges and gender dropped (see notebook section 11).
    assert "ProtectiveAddOnCount" not in names and "AutoPay" not in names
    assert not any(n.startswith("TenureGroup") for n in names)
    assert "TotalCharges" not in names and "MonthlyCharges" not in names and not any(n.startswith("gender") for n in names)
    assert not any("No internet service" in n or "No phone service" in n for n in names)


def test_ablation_options_validate_names():
    with pytest.raises(ValueError):
        build_preprocessor(drop=["NotAColumn"])
    cols = build_preprocessor(drop=["TotalCharges"], engineered=())
    assert "TotalCharges" not in cols.transformers[0][2] and "AutoPay" not in cols.transformers[1][2]


def test_default_drop_is_applied_but_can_be_overridden(clean_df):
    X_tr, _, _, _ = make_train_test_split(clean_df)
    engineered = add_engineered_features(X_tr)
    default_cols = build_preprocessor().fit(engineered).get_feature_names_out()
    full_cols = build_preprocessor(drop=()).fit(engineered).get_feature_names_out()
    assert "TotalCharges" not in default_cols and "MonthlyCharges" not in default_cols
    assert "TotalCharges" in full_cols and "MonthlyCharges" in full_cols
    assert any(c.startswith("gender") for c in full_cols) and not any(c.startswith("gender") for c in default_cols)


def test_engineered_features_are_available_on_request(clean_df):
    X_tr, _, _, _ = make_train_test_split(clean_df)
    engineered = add_engineered_features(X_tr)
    names = list(build_preprocessor(engineered=("TenureGroup", "ProtectiveAddOnCount", "AutoPay")).fit(engineered).get_feature_names_out())
    assert "ProtectiveAddOnCount" in names and "AutoPay" in names and "TenureGroup_49+" in names
