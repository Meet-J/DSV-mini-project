"""Tests for the interpretation helpers. Additivity is the key one: explanations must reproduce the model."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import config
from src.feature_engineering import build_pipeline, make_train_test_split
from src.interpretation import (
    direction_agreement, feature_contributions, level_effects, odds_ratio_table,
    permutation_importance_table, top_drivers, tree_shap_contributions,
)


@pytest.fixture(scope="module")
def fitted():
    if not config.CLEAN_DATA_PATH.exists():
        pytest.skip("Run `python -m src.preprocessing` first.")
    X_tr, X_te, y_tr, y_te = make_train_test_split(pd.read_csv(config.CLEAN_DATA_PATH))
    pipe = build_pipeline(LogisticRegression(max_iter=2000, random_state=config.RANDOM_SEED, C=10)).fit(X_tr, y_tr)
    return pipe, X_tr, X_te, y_te


def test_contributions_add_up_to_the_models_log_odds(fitted):
    pipe, X_tr, X_te, _ = fitted
    contrib, baseline = feature_contributions(pipe, X_te, background=X_tr)
    np.testing.assert_allclose(baseline + contrib.sum(axis=1).to_numpy(), pipe.decision_function(X_te), atol=1e-8)


def test_contributions_cover_raw_features_only_and_exclude_dropped_columns(fitted):
    pipe, X_tr, X_te, _ = fitted
    contrib, _ = feature_contributions(pipe, X_te, background=X_tr)
    assert "Contract" in contrib.columns and "Contract_One year" not in contrib.columns   # one-hot columns grouped
    for dropped in ("TotalCharges", "gender", "MonthlyCharges"):
        assert dropped not in contrib.columns                                              # not in the model


def test_an_average_customer_gets_zero_contributions(fitted):
    pipe, X_tr, _, _ = fitted
    contrib, _ = feature_contributions(pipe, X_tr, background=X_tr)
    assert abs(contrib.sum().sum()) < 1e-6          # across the background, positives and negatives cancel exactly


def test_odds_ratio_for_a_level_comparison_matches_hand_calculation(fitted):
    pipe, *_ = fitted
    table = odds_ratio_table(pipe).set_index(["feature", "comparison"])
    names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    coef = pd.Series(pipe.named_steps["model"].coef_[0], index=names)
    expected = np.exp(coef["Contract_Two year"] - coef["Contract_Month-to-month"])
    assert np.isclose(table.loc[("Contract", "Two year vs Month-to-month"), "odds_ratio"], expected)
    assert (table["odds_ratio"] > 0).all() and not table["odds_ratio"].isna().any()


def test_monthly_charges_is_not_a_model_input_and_has_no_odds_ratio(fitted):
    pipe, *_ = fitted
    assert "MonthlyCharges" not in set(odds_ratio_table(pipe)["feature"])
    assert "MonthlyCharges" not in pipe.named_steps["preprocess"].get_feature_names_out()


def test_tenure_odds_ratio_is_expressed_per_twelve_months(fitted):
    pipe, X_tr, *_ = fitted
    sd = X_tr["tenure"].std(ddof=0)
    coef = pd.Series(pipe.named_steps["model"].coef_[0], index=pipe.named_steps["preprocess"].get_feature_names_out())
    expected = np.exp(coef["tenure"] / sd * 12)
    got = odds_ratio_table(pipe).set_index("feature").loc["tenure", "odds_ratio"]
    assert np.isclose(got, expected, rtol=1e-6)


def test_permutation_importance_is_exactly_zero_for_columns_the_pipeline_drops(fitted):
    pipe, _, X_te, y_te = fitted
    perm = permutation_importance_table(pipe, X_te, y_te, n_repeats=3).set_index("feature")
    for dropped in ("TotalCharges", "gender", "MonthlyCharges"):
        assert perm.loc[dropped, "importance_mean"] == 0
    assert perm["importance_mean"].idxmax() in {"tenure", "Contract", "InternetService"}


def test_top_drivers_lists_only_features_that_push_in_that_direction(fitted):
    pipe, X_tr, X_te, _ = fitted
    contrib, _ = feature_contributions(pipe, X_te, background=X_tr)
    risk = top_drivers(contrib, X_te, k=3, direction="risk")
    assert list(risk.columns) == ["top_risk_driver_1", "top_risk_driver_2", "top_risk_driver_3"]
    first = risk.iloc[0, 0]
    feature = first.split(" = ")[0]
    assert contrib.iloc[0][feature] == contrib.iloc[0].max() and contrib.iloc[0][feature] > 0
    protective = top_drivers(contrib, X_te, k=1, direction="protective")
    f2 = protective.iloc[0, 0].split(" = ")[0]
    assert contrib.iloc[0][f2] == contrib.iloc[0].min() and contrib.iloc[0][f2] < 0


# ---------------------------------------------------------------- tree ensembles (need shap)
@pytest.fixture(scope="module")
def fitted_forest():
    pytest.importorskip("shap")
    X_tr, X_te, y_tr, _ = make_train_test_split(pd.read_csv(config.CLEAN_DATA_PATH))
    forest = RandomForestClassifier(n_estimators=60, max_depth=5, min_samples_leaf=10, random_state=config.RANDOM_SEED)
    return build_pipeline(forest, scale_numeric=False).fit(X_tr, y_tr), X_te


def test_tree_shap_contributions_add_up_to_the_predicted_probability(fitted_forest):
    pipe, X_te = fitted_forest
    contrib, baseline = tree_shap_contributions(pipe, X_te)
    np.testing.assert_allclose(baseline + contrib.sum(axis=1).to_numpy(), pipe.predict_proba(X_te)[:, 1], atol=1e-6)


def test_tree_shap_is_grouped_to_raw_features_and_excludes_dropped_columns(fitted_forest):
    pipe, X_te = fitted_forest
    contrib, _ = tree_shap_contributions(pipe, X_te)
    assert "Contract" in contrib.columns and "Contract_One year" not in contrib.columns
    for dropped in ("TotalCharges", "gender", "MonthlyCharges"):
        assert dropped not in contrib.columns


def test_level_effects_cover_every_customer_once_per_feature(fitted_forest):
    pipe, X_te = fitted_forest
    contrib, _ = tree_shap_contributions(pipe, X_te)
    eff = level_effects(contrib, X_te, scale=100)
    for feature in ("Contract", "tenure (by band, months)", "SeniorCitizen"):
        assert eff[eff.feature == feature]["customers"].sum() == len(X_te)
    assert set(eff[eff.feature == "tenure (by band, months)"]["level"]) == {"0-12", "13-24", "25-48", "49+"}


def test_direction_agreement_on_a_hand_built_example():
    odds = pd.DataFrame({"feature": ["Contract", "TechSupport", "tenure"],
                         "comparison": ["Two year vs Month-to-month", "Yes vs No", "+12 months"],
                         "log_odds": [-1.3, -0.3, -0.4]})
    effects = pd.DataFrame({"feature": ["Contract", "Contract", "TechSupport", "TechSupport"],
                            "level": ["Two year", "Month-to-month", "Yes", "No"],
                            "mean_effect": [-9.0, 6.0, 1.0, 0.5]})       # Contract agrees; TechSupport has the wrong sign
    out = direction_agreement(odds, effects).set_index("feature")
    assert out.loc["Contract", "agree"] and not out.loc["TechSupport", "agree"]
    assert "tenure" not in out.index                                      # skipped on purpose (different units)


def test_senior_citizen_is_shown_as_yes_no_in_driver_labels():
    contrib = pd.DataFrame({"SeniorCitizen": [0.2], "Contract": [0.1]}, index=["c1"])
    X = pd.DataFrame({"SeniorCitizen": [1], "Contract": ["Month-to-month"]}, index=["c1"])
    out = top_drivers(contrib, X, k=2)
    assert out.iloc[0, 0] == "SeniorCitizen = Yes" and out.iloc[0, 1] == "Contract = Month-to-month"
