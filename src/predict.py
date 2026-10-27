"""Score customers: churn probability, risk tier, value segment and top drivers, shaped for Power BI.

Two entry points:

  python -m src.predict
      Scores all existing customers with OUT-OF-FOLD predictions: the data is split into 5 folds, and each
      customer is scored (and explained) by a model fitted on the other 4 folds, so no customer is ever scored
      by a model that saw them. Writes output/customer_churn_predictions.csv plus a data dictionary and report.

  python -m src.predict --input new_customers.csv --output scored.csv
      Scores NEW customers (same columns as the original data, no Churn column) with the saved model
      models/churn_model.pkl (fitted on the 80% training split).

Why out-of-fold for existing customers: a model scoring the rows it was trained on reports overconfident
probabilities, which would distort every number built on them (risk tiers, revenue at risk).
"""
from __future__ import annotations

import argparse
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import config
from src.evaluation import ACCENT, MUTED, SLATE, _style, wilson_interval
from src.feature_engineering import add_engineered_features, encode_target, make_train_test_split
from src.interpretation import TREE_MODELS, feature_contributions, top_drivers, tree_shap_contributions
from src.preprocessing import clean_data
from src.reporting import md_table

RISK_ORDER = {"Low": 1, "Medium": 2, "High": 3}
TENURE_ORDER = {"0-12": 1, "13-24": 2, "25-48": 3, "49+": 4}
CHARGE_BINS, CHARGE_LABELS = [0, 35, 65, 95, 200], ["<=35", "35-65", "65-95", ">95"]   # same bands as the EDA notebook
SERVICE_COLUMNS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


# ----------------------------------------------------------------------------------------------
# Segmentation rules
# ----------------------------------------------------------------------------------------------
def risk_category(proba, low: float = config.RISK_LOW_THRESHOLD, high: float = config.RISK_HIGH_THRESHOLD) -> pd.Series:
    """Low: p < low. Medium: low <= p <= high. High: p > high."""
    p = pd.Series(np.asarray(proba, dtype=float), index=getattr(proba, "index", None))
    return pd.Series(np.where(p > high, "High", np.where(p >= low, "Medium", "Low")), index=p.index)


def high_value_threshold(monthly_charges: pd.Series, quantile: float = config.HIGH_VALUE_QUANTILE) -> float:
    return float(monthly_charges.quantile(quantile))


def value_segment(monthly_charges: pd.Series, threshold: float) -> pd.Series:
    return pd.Series(np.where(monthly_charges >= threshold, "High value", "Standard value"), index=monthly_charges.index)


def retention_segment(risk: pd.Series, value: pd.Series, active: pd.Series) -> pd.Series:
    """Risk x value for active customers; customers who already churned are not retention targets."""
    seg = risk.str.cat(value.str.lower(), sep=" risk + ")
    return seg.where(active, "Already churned")


# ----------------------------------------------------------------------------------------------
# Out-of-fold scoring
# ----------------------------------------------------------------------------------------------
def oof_scores(pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = config.CV_FOLDS, seed: int = config.RANDOM_SEED) -> dict:
    """Probability and explanation for every row, each from a model fitted WITHOUT that row.

    `pipeline` must be unfitted (use sklearn.base.clone). Contributions are in probability points for tree
    models (tree SHAP) and in log-odds for logistic regression; either way they rank features correctly.
    """
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = pd.Series(np.nan, index=X.index, name="ChurnProbability")
    fold_of = pd.Series(-1, index=X.index, name="Fold")
    parts = []
    is_tree = type(pipeline.named_steps["model"]).__name__ in TREE_MODELS
    for k, (tr, va) in enumerate(folds.split(X, y)):
        model = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
        proba.iloc[va] = model.predict_proba(X.iloc[va])[:, 1]
        fold_of.iloc[va] = k
        if is_tree:
            contrib, baseline = tree_shap_contributions(model, X.iloc[va])
            gap = np.abs(baseline + contrib.sum(axis=1).to_numpy() - proba.iloc[va].to_numpy()).max()
            assert gap < 1e-6, f"fold {k}: contributions do not add up to the probability (gap {gap})"
        else:
            contrib, _ = feature_contributions(model, X.iloc[va], background=X.iloc[tr])
        parts.append(contrib)
    assert (fold_of >= 0).all() and proba.notna().all(), "every customer must be scored exactly once"
    return {"proba": proba, "contributions": pd.concat(parts).loc[X.index], "fold": fold_of}


# ----------------------------------------------------------------------------------------------
# Power BI table
# ----------------------------------------------------------------------------------------------
def reporting_fields(X: pd.DataFrame) -> pd.DataFrame:
    """Descriptive columns for slicing in Power BI. None of these (except the raw columns) is a model input."""
    eng = add_engineered_features(X)
    n_services = ((X["PhoneService"] == "Yes").astype(int) + (X["InternetService"] != "No").astype(int)
                  + (X[SERVICE_COLUMNS] == "Yes").sum(axis=1))
    out = pd.DataFrame(index=X.index)
    out["TenureGroup"] = eng["TenureGroup"]
    out["TenureGroupOrder"] = out["TenureGroup"].map(TENURE_ORDER)
    out["ChargeBand"] = pd.cut(X["MonthlyCharges"], bins=CHARGE_BINS, labels=CHARGE_LABELS).astype(str)
    out["ChargeBandOrder"] = out["ChargeBand"].map({l: i + 1 for i, l in enumerate(CHARGE_LABELS)})
    out["AutoPay"] = eng["AutoPay"].map({1: "Yes", 0: "No"})
    out["ProtectiveAddOnCount"] = eng["ProtectiveAddOnCount"]
    out["NumberOfServices"] = n_services.astype(int)
    return out


def attach_drivers(table: pd.DataFrame, contributions: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    risk = top_drivers(contributions, X, k=3, direction="risk")
    protective = top_drivers(contributions, X, k=1, direction="protective")
    risk.columns = [f"TopRiskDriver{i}" for i in (1, 2, 3)]
    protective.columns = ["TopProtectiveDriver1"]
    return table.join(risk).join(protective)


DATA_DICTIONARY = [
    ("CustomerID", "Unique customer identifier (primary key)."),
    ("Churn", "Actual outcome in the dataset snapshot: Yes = customer had left, No = still a customer."),
    ("ChurnFlag", "1 if Churn = Yes else 0. AVERAGE(ChurnFlag) is the churn rate."),
    ("CustomerStatus", "'Churned' or 'Active'. Risk metrics and revenue at risk should use Active customers only."),
    ("ChurnProbability", "Predicted probability of churn from an out-of-fold model (the customer was not in that model's training data)."),
    ("RiskCategory", "Low (p < 0.30), Medium (0.30 <= p <= 0.70) or High (p > 0.70). Practical project thresholds; see output/risk_segmentation_report.md."),
    ("RiskOrder", "1/2/3 for Low/Medium/High. Use as the 'sort by column' for RiskCategory."),
    ("ValueSegment", "'High value' if MonthlyCharges is in the top quartile of all customers, else 'Standard value'. Revenue is the only value signal in the data."),
    ("RetentionSegment", "Risk tier + value segment for Active customers (e.g. 'High risk + high value'); 'Already churned' otherwise."),
    ("MonthlyCharges", "Current monthly charge (currency not stated in the dataset). Not a model input."),
    ("TotalCharges", "Cumulative charges. 0 for the 11 customers with tenure 0 (see data/processed/cleaning_log.md). Not a model input."),
    ("ExpectedMonthlyRevenueAtRisk", "ChurnProbability x MonthlyCharges for Active customers; 0 for Churned customers. Assumes calibrated probabilities."),
    ("tenure", "Months with the company."),
    ("TenureGroup", "0-12, 13-24, 25-48 or 49+ months."),
    ("TenureGroupOrder", "1-4; sort order for TenureGroup."),
    ("Contract", "Month-to-month, One year or Two year."),
    ("InternetService", "DSL, Fiber optic or No."),
    ("PaymentMethod", "Electronic check, Mailed check, Bank transfer (automatic) or Credit card (automatic)."),
    ("PaperlessBilling", "Yes or No."),
    ("AutoPay", "Yes if PaymentMethod is an automatic bank transfer or credit card."),
    ("NumberOfServices", "Count of services held: phone, internet, and each of the six add-ons set to Yes (0-8)."),
    ("ProtectiveAddOnCount", "How many of OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport are Yes (0-4)."),
    ("ChargeBand", "MonthlyCharges band: <=35, 35-65, 65-95, >95 (upper edge inclusive)."),
    ("ChargeBandOrder", "1-4; sort order for ChargeBand."),
    ("gender", "Male or Female. Not a model input."),
    ("SeniorCitizen", "Yes or No."),
    ("Partner", "Yes or No."),
    ("Dependents", "Yes or No."),
    ("PhoneService", "Yes or No."),
    ("MultipleLines", "Yes, No, or 'No phone service'."),
    ("OnlineSecurity", "Yes, No, or 'No internet service'."),
    ("OnlineBackup", "Yes, No, or 'No internet service'."),
    ("DeviceProtection", "Yes, No, or 'No internet service'."),
    ("TechSupport", "Yes, No, or 'No internet service'."),
    ("StreamingTV", "Yes, No, or 'No internet service'."),
    ("StreamingMovies", "Yes, No, or 'No internet service'."),
    ("TopRiskDriver1", "Feature pushing this customer's predicted churn probability up the most, as 'feature = value' (tree SHAP). Empty if none pushes up."),
    ("TopRiskDriver2", "Second-largest upward driver."),
    ("TopRiskDriver3", "Third-largest upward driver."),
    ("TopProtectiveDriver1", "Feature pushing the probability down the most. These describe the model's associations, not causes."),
    ("DataSplit", "'Train' or 'Test': which side of the 80/20 split used for model selection the customer fell on. Probabilities here are out-of-fold for everyone."),
]


def build_predictions_table(X: pd.DataFrame, y: pd.Series, scores: dict, split: pd.Series, value_cutoff: float) -> pd.DataFrame:
    proba = scores["proba"]
    active = y == 0
    value = value_segment(X["MonthlyCharges"], value_cutoff)
    risk = risk_category(proba)

    t = pd.DataFrame(index=X.index)
    t["Churn"] = np.where(y == 1, "Yes", "No")
    t["ChurnFlag"] = y.astype(int)
    t["CustomerStatus"] = np.where(active, "Active", "Churned")
    t["ChurnProbability"] = proba.round(6)
    t["RiskCategory"] = risk
    t["RiskOrder"] = risk.map(RISK_ORDER)
    t["ValueSegment"] = value
    t["RetentionSegment"] = retention_segment(risk, value, active)
    t["MonthlyCharges"] = X["MonthlyCharges"]
    t["TotalCharges"] = X["TotalCharges"]
    t["ExpectedMonthlyRevenueAtRisk"] = np.where(active, proba * X["MonthlyCharges"], 0.0).round(4)
    t["tenure"] = X["tenure"]
    t = t.join(reporting_fields(X))
    for col in ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling", "gender", "Partner", "Dependents",
                "PhoneService", "MultipleLines", *SERVICE_COLUMNS]:
        t[col] = X[col]
    t["SeniorCitizen"] = X["SeniorCitizen"].map({1: "Yes", 0: "No"})
    t = attach_drivers(t, scores["contributions"], X)
    t["DataSplit"] = split
    t.index.name = "CustomerID"
    t = t.reset_index()
    # column order follows the data dictionary
    return t[[name for name, _ in DATA_DICTIONARY]]


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------
def tier_calibration(table: pd.DataFrame) -> pd.DataFrame:
    bands = {"Low": (0.0, config.RISK_LOW_THRESHOLD), "Medium": (config.RISK_LOW_THRESHOLD, config.RISK_HIGH_THRESHOLD),
             "High": (config.RISK_HIGH_THRESHOLD, 1.0)}
    rows = []
    for tier, (lo, hi) in bands.items():
        g = table[table.RiskCategory == tier]
        k, n = int(g.ChurnFlag.sum()), len(g)
        ci_lo, ci_hi = wilson_interval(k, n)
        observed = k / n if n else np.nan
        rows.append({"tier": tier, "band": f"{lo:.2f} to {hi:.2f}", "customers": n, "mean_predicted": g.ChurnProbability.mean(),
                     "observed_churn_rate": observed, "ci_low": ci_lo, "ci_high": ci_hi,
                     "observed_inside_band": bool(lo <= observed <= hi),
                     "predicted_inside_ci": bool(ci_lo <= g.ChurnProbability.mean() <= ci_hi)})
    return pd.DataFrame(rows)


def plot_risk_tiers(table: pd.DataFrame, calib: pd.DataFrame, path) -> None:
    _style()
    lo, hi = config.RISK_LOW_THRESHOLD, config.RISK_HIGH_THRESHOLD
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    active = table[table.CustomerStatus == "Active"]
    axes[0].axvspan(0, lo, color="#E8EDF2"); axes[0].axvspan(lo, hi, color="#F4E3E1"); axes[0].axvspan(hi, 1, color="#E9C2BF")
    axes[0].hist(active.ChurnProbability, bins=40, color=SLATE, edgecolor="white")
    for x in (lo, hi):
        axes[0].axvline(x, color="black", ls="--", lw=1)
    ymax = axes[0].get_ylim()[1]
    for x, label in [(lo / 2, "Low"), ((lo + hi) / 2, "Medium"), ((hi + 1) / 2, "High")]:
        axes[0].text(x, ymax * 0.93, label, ha="center", fontsize=10, fontweight="bold")
    axes[0].set(xlabel="Out-of-fold churn probability", ylabel="Active customers", title="Active customers by predicted risk", xlim=(0, 1))
    x = np.arange(len(calib))
    axes[1].bar(x - 0.2, calib.mean_predicted, 0.4, color=MUTED, label="Mean predicted probability")
    axes[1].bar(x + 0.2, calib.observed_churn_rate, 0.4, color=SLATE, label="Observed churn rate (all customers)")
    axes[1].errorbar(x + 0.2, calib.observed_churn_rate, yerr=[calib.observed_churn_rate - calib.ci_low, calib.ci_high - calib.observed_churn_rate],
                     fmt="none", ecolor="black", capsize=3)
    axes[1].set_xticks(x, [f"{t}\n({b})" for t, b in zip(calib.tier, calib.band)])
    axes[1].set(ylabel="Churn rate", ylim=(0, 1), title="Do the tiers mean what they say?")
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def build_report(table: pd.DataFrame, scores_summary: dict, calib: pd.DataFrame, value_cutoff: float) -> str:
    active = table[table.CustomerStatus == "Active"]
    n_all, n_act, n_churn = len(table), len(active), int(table.ChurnFlag.sum())
    by_tier = (table.groupby("RiskCategory").agg(customers=("ChurnFlag", "size"), churners=("ChurnFlag", "sum"))
               .reindex(["High", "Medium", "Low"]))
    by_tier["share_of_all_churners_pct"] = by_tier.churners / n_churn * 100
    act_tier = (active.groupby("RiskCategory").agg(active_customers=("CustomerID", "size"), monthly_charges=("MonthlyCharges", "sum"),
                                                   expected_monthly_revenue_at_risk=("ExpectedMonthlyRevenueAtRisk", "sum"),
                                                   avg_probability=("ChurnProbability", "mean")).reindex(["High", "Medium", "Low"]))
    seg = (active.groupby("RetentionSegment").agg(active_customers=("CustomerID", "size"), monthly_charges=("MonthlyCharges", "sum"),
                                                  avg_probability=("ChurnProbability", "mean")).reset_index())
    seg_order = {s: i for i, s in enumerate(["High risk + high value", "High risk + standard value", "Medium risk + high value",
                                             "Medium risk + standard value", "Low risk + high value", "Low risk + standard value"])}
    seg = seg.sort_values("RetentionSegment", key=lambda c: c.map(seg_order))
    sens_rows = []
    for cut in (0.5, 0.6, 0.7, 0.8):
        above_all, above_act = table[table.ChurnProbability > cut], active[active.ChurnProbability > cut]
        sens_rows.append({"high_risk_cutoff": cut, "active_customers_above": len(above_act),
                          "monthly_charges_of_those": above_act.MonthlyCharges.sum(),
                          "observed_churn_rate_all_customers_above": above_all.ChurnFlag.mean()})
    sens = pd.DataFrame(sens_rows)
    supported = bool(calib.observed_inside_band.all())

    lines = [
        "# Risk segmentation report",
        "",
        "Generated by `python -m src.predict`. All numbers were computed by that run.",
        "",
        "## 1. How the probabilities were produced",
        f"- {n_all:,} customers scored with 5-fold out-of-fold predictions (stratified, seed {config.RANDOM_SEED}): each customer's probability and "
        "explanation come from a model fitted on the other four folds, with the hyperparameters selected on the training split.",
        f"- Out-of-fold quality over all customers: ROC-AUC {scores_summary['auc']:.4f}, PR-AUC {scores_summary['pr_auc']:.4f}, "
        f"Brier score {scores_summary['brier']:.4f}. These are cross-validation figures, not held-out test results (those are in `output/model_report.md`).",
        f"- {n_churn:,} customers ({n_churn / n_all:.1%}) are already churned and {n_act:,} are active. Risk counts and revenue at risk below use ACTIVE customers only; "
        "churned customers cannot be retained.",
        "",
        "## 2. Are the 0.30 / 0.70 tiers justified?",
        f"The thresholds are practical project assumptions from the brief, not universal truths. **Rule, fixed before looking at results:** keep them if the "
        "observed churn rate of each tier falls inside that tier's probability band (so a 'High' customer really is more likely than not to churn, and so on).",
        "",
        md_table(calib, {"mean_predicted": "{:.3f}", "observed_churn_rate": "{:.3f}", "ci_low": "{:.3f}", "ci_high": "{:.3f}"}),
        "",
        f"**Outcome:** observed rate inside the band for every tier: **{supported}**. "
        + ("The default thresholds are kept." if supported else "The default thresholds are NOT supported; review before using them."),
        "`predicted_inside_ci` shows whether the tier's mean predicted probability lies inside the 95% Wilson interval of its observed rate (calibration within the tier).",
        "",
        "Where the customers sit (all customers, with how many of the actual churners each tier contains):",
        "",
        md_table(by_tier.reset_index(), {"share_of_all_churners_pct": "{:.1f}"}),
        "",
        "## 3. Active customers: risk and revenue exposure",
        "Revenue is monthly charges only (currency not stated, no margin, no lifetime value). Expected revenue at risk = probability x monthly charge, "
        "which assumes the probabilities are calibrated.",
        "",
        md_table(act_tier.reset_index(), {"monthly_charges": "{:,.2f}", "expected_monthly_revenue_at_risk": "{:,.2f}", "avg_probability": "{:.3f}"}),
        "",
        f"Total expected monthly revenue at risk across active customers: {active.ExpectedMonthlyRevenueAtRisk.sum():,.2f} "
        f"of {active.MonthlyCharges.sum():,.2f} total monthly charges.",
        "",
        f"## 4. Retention segments (active customers; high value = monthly charges >= {value_cutoff:.2f}, the top quartile of all customers)",
        "",
        md_table(seg, {"monthly_charges": "{:,.2f}", "avg_probability": "{:.3f}"}),
        "",
        "## 5. Sensitivity of the High-risk cut-off",
        "",
        md_table(sens, {"monthly_charges_of_those": "{:,.2f}", "observed_churn_rate_all_customers_above": "{:.3f}"}),
        "",
        "## 6. Caveats",
        "- The risk tiers describe predicted probability, not a recommended action; a retention programme would also weigh offer cost and success rate, "
        "which the dataset does not contain.",
        "- 'High value' uses monthly charges only. A customer's real value also depends on margin and expected remaining lifetime, which are unknown here.",
        "- Because the monthly bill is determined by the services held, 'high value' customers are largely those with fiber and several services.",
        "- Drivers (`TopRiskDriver1-3`) are the model's associations for that customer, not causes of churn.",
        "- The saved model (`models/churn_model.pkl`) is fitted on the 80% training split; new customers are scored with it, not with the out-of-fold models.",
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------------------------------
def score_existing_customers() -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_csv(config.CLEAN_DATA_PATH)
    X_train, X_test, _, _ = make_train_test_split(df)
    indexed = df.set_index(config.ID_COL)
    y = encode_target(indexed[config.TARGET_COL])
    X = indexed.drop(columns=[config.TARGET_COL])
    final = joblib.load(config.MODEL_PATH)

    scores = oof_scores(clone(final), X, y)
    split = pd.Series("Train", index=X.index).where(~X.index.isin(X_test.index), "Test")
    cutoff = high_value_threshold(X["MonthlyCharges"])
    table = build_predictions_table(X, y, scores, split, cutoff)

    # ---- checks
    assert len(table) == len(df) and table.CustomerID.is_unique and table.CustomerID.isin(df.customerID).all()
    assert table.ChurnProbability.between(0, 1).all() and not table.ChurnProbability.isna().any()
    assert (risk_category(table.ChurnProbability) == table.RiskCategory).all()
    assert set(table.columns) == {n for n, _ in DATA_DICTIONARY}
    assert (table.loc[table.CustomerStatus == "Churned", "ExpectedMonthlyRevenueAtRisk"] == 0).all()
    summary = {"auc": roc_auc_score(y, scores["proba"]), "pr_auc": average_precision_score(y, scores["proba"]),
               "brier": brier_score_loss(y, scores["proba"])}

    calib = tier_calibration(table)
    config.OUTPUT_DIR.mkdir(exist_ok=True); config.MODEL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.PREDICTIONS_PATH, index=False)
    (config.OUTPUT_DIR / "predictions_data_dictionary.md").write_text(
        "# customer_churn_predictions.csv: data dictionary\n\nOne row per customer (CustomerID is the primary key).\n\n"
        + md_table(pd.DataFrame(DATA_DICTIONARY, columns=["column", "description"]), {}) + "\n", encoding="utf-8")
    (config.OUTPUT_DIR / "risk_segmentation_report.md").write_text(build_report(table, summary, calib, cutoff), encoding="utf-8")
    plot_risk_tiers(table, calib, config.MODEL_FIGURES_DIR / "risk_tiers.png")
    print(f"Scored {len(table):,} customers in {time.time() - t0:.0f}s | OOF AUC {summary['auc']:.4f} | "
          f"tier rule satisfied: {bool(calib.observed_inside_band.all())}", flush=True)
    return table


def score_new_customers(raw: pd.DataFrame) -> pd.DataFrame:
    """Score customers who are not in the training data, using the saved model. `raw` has the original columns, no Churn."""
    cleaned, _ = clean_data(raw)
    X = cleaned.set_index(config.ID_COL)
    model = joblib.load(config.MODEL_PATH)
    proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    if type(model.named_steps["model"]).__name__ in TREE_MODELS:
        contrib, _ = tree_shap_contributions(model, X)
    else:
        background = make_train_test_split(pd.read_csv(config.CLEAN_DATA_PATH))[0]
        contrib, _ = feature_contributions(model, X, background)
    cutoff = high_value_threshold(pd.read_csv(config.CLEAN_DATA_PATH)["MonthlyCharges"])
    out = pd.DataFrame({"ChurnProbability": proba.round(6), "RiskCategory": risk_category(proba),
                        "ValueSegment": value_segment(X["MonthlyCharges"], cutoff), "MonthlyCharges": X["MonthlyCharges"]})
    risk = top_drivers(contrib, X, k=3, direction="risk")
    risk.columns = [f"TopRiskDriver{i}" for i in (1, 2, 3)]
    return out.join(risk).rename_axis("CustomerID").reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score customers for churn risk.")
    parser.add_argument("--input", help="CSV of NEW customers (original columns, no Churn). Omit to score the existing customers out-of-fold.")
    parser.add_argument("--output", default="scored_customers.csv", help="Where to write scores for --input.")
    args = parser.parse_args()
    if args.input:
        scored = score_new_customers(pd.read_csv(args.input))
        scored.to_csv(args.output, index=False)
        print(f"Scored {len(scored):,} new customers -> {args.output}")
    else:
        score_existing_customers()


if __name__ == "__main__":
    main()
