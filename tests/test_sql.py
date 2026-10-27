"""Every SQL query is checked against an independent pandas calculation of the same question."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.run_sql import SQL_PATH, load_database, parse_queries, run_queries

pytestmark = pytest.mark.skipif(not config.PREDICTIONS_PATH.exists(), reason="Run `python -m src.predict` first.")


@pytest.fixture(scope="module")
def t():
    return pd.read_csv(config.PREDICTIONS_PATH)


@pytest.fixture(scope="module")
def sql():
    queries = parse_queries(SQL_PATH.read_text(encoding="utf-8"))
    return queries, run_queries(load_database(), queries)


def same(result: pd.DataFrame, expected: pd.DataFrame):
    assert list(result.columns) == list(expected.columns), (list(result.columns), list(expected.columns))
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False, atol=1e-9)


def pct(num, den, d=1):
    return np.round(100.0 * num / den, d)


# ---------------------------------------------------------------- the file itself
def test_sql_file_parses_into_documented_unique_queries(sql):
    queries, _ = sql
    names = [q["name"] for q in queries]
    assert len(names) == len(set(names)) == 16
    assert all(q["question"] and q["reasoning"] and q["sql"].upper().lstrip().startswith(("SELECT", "WITH")) for q in queries)
    assert all(q["sql"].count(";") == 1 for q in queries)           # exactly one statement per block


# ---------------------------------------------------------------- headline and guards
def test_q00_sanity_checks_are_all_zero_and_correct(sql, t):
    expected = pd.DataFrame([{"duplicate_ids": len(t) - t.CustomerID.nunique(), "invalid_probabilities": int((~t.ChurnProbability.between(0, 1)).sum()),
                              "rows_with_missing_core_fields": 0, "tier_inconsistencies": 0, "churned_customers_with_exposure": 0}])
    same(sql[1]["q00_sanity_checks"], expected)
    assert (sql[1]["q00_sanity_checks"].iloc[0] == 0).all()


def test_q01_headline(sql, t):
    n, c = len(t), int(t.ChurnFlag.sum())
    same(sql[1]["q01_headline"], pd.DataFrame([{"total_customers": n, "churned_customers": c, "active_customers": n - c, "churn_rate_pct": pct(c, n, 2),
                                                "avg_monthly_charges": round(t.MonthlyCharges.mean(), 2), "avg_tenure_months": round(t.tenure.mean(), 1)}]))


# ---------------------------------------------------------------- churn drivers
def test_q02_churn_by_contract(sql, t):
    g = t.groupby("Contract").agg(customers=("ChurnFlag", "size"), churned=("ChurnFlag", "sum")).reset_index()
    overall = t.ChurnFlag.mean()
    g["churn_rate_pct"] = pct(g.churned, g.customers)
    g["share_of_customers_pct"] = pct(g.customers, len(t))
    g["share_of_churners_pct"] = pct(g.churned, t.ChurnFlag.sum())
    g["lift_vs_overall"] = np.round((g.churned / g.customers) / overall, 2)
    same(sql[1]["q02_churn_by_contract"], g.sort_values("churn_rate_pct", ascending=False))


def test_q03_churn_by_payment_method(sql, t):
    g = t.groupby("PaymentMethod").agg(customers=("ChurnFlag", "size"), rate=("ChurnFlag", "mean")).reset_index()
    g["churn_rate_pct"] = np.round(100 * g.rate, 1)
    g["points_vs_overall"] = np.round(100 * g.rate - 100 * t.ChurnFlag.mean(), 1)
    same(sql[1]["q03_churn_by_payment_method"], g.sort_values("churn_rate_pct", ascending=False)[["PaymentMethod", "customers", "churn_rate_pct", "points_vs_overall"]])


def test_q04_charges_by_churn_status(sql, t):
    g = t.groupby("CustomerStatus").agg(customers=("ChurnFlag", "size"), avg_monthly_charges=("MonthlyCharges", "mean"),
                                        total_monthly_charges=("MonthlyCharges", "sum"), avg_tenure_months=("tenure", "mean")).reset_index()
    g[["avg_monthly_charges", "total_monthly_charges"]] = g[["avg_monthly_charges", "total_monthly_charges"]].round(2)
    g["avg_tenure_months"] = g.avg_tenure_months.round(1)
    same(sql[1]["q04_charges_by_churn_status"], g.sort_values("CustomerStatus"))


def test_q05_churn_by_tenure_group(sql, t):
    g = t.groupby(["TenureGroupOrder", "TenureGroup"]).agg(customers=("ChurnFlag", "size"), churned=("ChurnFlag", "sum")).reset_index()
    g["churn_rate_pct"] = pct(g.churned, g.customers)
    same(sql[1]["q05_churn_by_tenure_group"], g.sort_values("TenureGroupOrder")[["TenureGroup", "customers", "churned", "churn_rate_pct"]])


def test_q06_contract_by_tenure_matrix(sql, t):
    p = (t.pivot_table(index="Contract", columns="TenureGroup", values="ChurnFlag", aggfunc="mean") * 100).round(1)
    p = p[["0-12", "13-24", "25-48", "49+"]]
    p.columns = ["churn_pct_0_12", "churn_pct_13_24", "churn_pct_25_48", "churn_pct_49_plus"]
    p = p.loc[["Month-to-month", "One year", "Two year"]].reset_index()
    same(sql[1]["q06_contract_by_tenure_matrix"], p)


def test_q07_fiber_vs_dsl_by_price_band(sql, t):
    n = t[t.InternetService != "No"]
    rows = []
    for (order, band), g in n.groupby(["ChargeBandOrder", "ChargeBand"]):
        dsl, fib = g[g.InternetService == "DSL"], g[g.InternetService == "Fiber optic"]
        rows.append({"ChargeBand": band, "dsl_customers": len(dsl), "dsl_churn_pct": np.round(100 * dsl.ChurnFlag.mean(), 1) if len(dsl) else np.nan,
                     "fiber_customers": len(fib), "fiber_churn_pct": np.round(100 * fib.ChurnFlag.mean(), 1) if len(fib) else np.nan})
    same(sql[1]["q07_fiber_vs_dsl_by_price_band"], pd.DataFrame(rows))


def test_q08_service_count_and_churn(sql, t):
    g = t[t.InternetService != "No"].groupby("ProtectiveAddOnCount").agg(customers=("ChurnFlag", "size"), rate=("ChurnFlag", "mean")).reset_index()
    g["churn_rate_pct"] = np.round(100 * g.rate, 1)
    same(sql[1]["q08_service_count_and_churn"], g[["ProtectiveAddOnCount", "customers", "churn_rate_pct"]])


# ---------------------------------------------------------------- model validation
def test_q09_risk_tier_validation(sql, t):
    g = t.groupby(["RiskOrder", "RiskCategory"]).agg(customers=("ChurnFlag", "size"), mean_predicted=("ChurnProbability", "mean"),
                                                     observed_churn_rate=("ChurnFlag", "mean")).reset_index()
    g[["mean_predicted", "observed_churn_rate"]] = g[["mean_predicted", "observed_churn_rate"]].round(3)
    same(sql[1]["q09_risk_tier_validation"], g.sort_values("RiskOrder", ascending=False)[["RiskCategory", "customers", "mean_predicted", "observed_churn_rate"]])


def test_q10_risk_decile_gains_matches_ntile_semantics(sql, t):
    ordered = t.sort_values(["ChurnProbability", "CustomerID"], ascending=[False, True])
    groups = np.array_split(np.arange(len(ordered)), 10)           # same sizing as NTILE: the first (n mod 10) groups get one extra row
    flags = ordered.ChurnFlag.to_numpy()
    cust = np.array([len(g) for g in groups]); ch = np.array([flags[g].sum() for g in groups])
    expected = pd.DataFrame({"risk_decile": range(1, 11), "customers": cust, "churners": ch, "churn_rate_pct": np.round(100.0 * ch / cust, 1),
                             "cumulative_pct_of_churners_captured": np.round(100.0 * np.cumsum(ch) / ch.sum(), 1),
                             "lift": np.round((ch / cust) / (ch.sum() / cust.sum()), 2)})
    same(sql[1]["q10_risk_decile_gains"], expected)
    r = sql[1]["q10_risk_decile_gains"]
    assert r.customers.sum() == len(t) and r.cumulative_pct_of_churners_captured.iloc[-1] == 100.0    # invariants


# ---------------------------------------------------------------- active customers: risk and revenue
def test_q11_active_risk_profile(sql, t):
    a = t[t.CustomerStatus == "Active"]
    g = a.groupby(["RiskOrder", "RiskCategory"]).agg(active_customers=("ChurnFlag", "size"), avg_tenure_months=("tenure", "mean"),
                                                     avg_monthly_charges=("MonthlyCharges", "mean")).reset_index()
    for col, cond in [("pct_month_to_month", a.Contract == "Month-to-month"), ("pct_fiber", a.InternetService == "Fiber optic"),
                      ("pct_electronic_check", a.PaymentMethod == "Electronic check")]:
        g[col] = (100 * cond.groupby([a.RiskOrder, a.RiskCategory]).mean()).round(1).to_numpy()
    g["avg_tenure_months"], g["avg_monthly_charges"] = g.avg_tenure_months.round(1), g.avg_monthly_charges.round(2)
    same(sql[1]["q11_active_risk_profile"], g.sort_values("RiskOrder", ascending=False).drop(columns="RiskOrder"))


def test_q12_revenue_exposure_by_risk(sql, t):
    a = t[t.CustomerStatus == "Active"]
    g = a.groupby(["RiskOrder", "RiskCategory"]).agg(active_customers=("ChurnFlag", "size"), monthly_charges=("MonthlyCharges", "sum"),
                                                     expected=("ExpectedMonthlyRevenueAtRisk", "sum")).reset_index()
    g["pct_of_active_monthly_charges"] = np.round(100 * g.monthly_charges / g.monthly_charges.sum(), 1)
    g["monthly_charges"], g["expected_monthly_revenue_at_risk"] = g.monthly_charges.round(2), g.expected.round(2)
    same(sql[1]["q12_revenue_exposure_by_risk"],
         g.sort_values("RiskOrder", ascending=False)[["RiskCategory", "active_customers", "monthly_charges", "pct_of_active_monthly_charges", "expected_monthly_revenue_at_risk"]])


def test_q13_retention_segments(sql, t):
    a = t[t.CustomerStatus == "Active"]
    g = a.groupby("RetentionSegment").agg(active_customers=("ChurnFlag", "size"), monthly_charges=("MonthlyCharges", "sum"),
                                          avg_probability=("ChurnProbability", "mean"), expected=("ExpectedMonthlyRevenueAtRisk", "sum")).reset_index()
    g["exposure_rank"] = g.expected.rank(method="min", ascending=False).astype(int)
    g["monthly_charges"], g["avg_probability"], g["expected_revenue_at_risk"] = g.monthly_charges.round(2), g.avg_probability.round(3), g.expected.round(2)
    same(sql[1]["q13_retention_segments"],
         g.sort_values("exposure_rank")[["RetentionSegment", "active_customers", "monthly_charges", "avg_probability", "expected_revenue_at_risk", "exposure_rank"]])


def test_q14_priority_retention_list(sql, t):
    a = t[t.CustomerStatus == "Active"].sort_values(["ExpectedMonthlyRevenueAtRisk", "CustomerID"], ascending=[False, True]).head(20).copy()
    a.insert(0, "priority", range(1, 21))
    a["churn_probability"], a["expected_monthly_revenue_at_risk"] = a.ChurnProbability.round(3), a.ExpectedMonthlyRevenueAtRisk.round(2)
    same(sql[1]["q14_priority_retention_list"], a[["priority", "CustomerID", "RiskCategory", "ValueSegment", "churn_probability", "MonthlyCharges",
                                                   "expected_monthly_revenue_at_risk", "TopRiskDriver1", "TopRiskDriver2"]])
    assert (sql[1]["q14_priority_retention_list"].CustomerID.isin(t.loc[t.CustomerStatus == "Active", "CustomerID"])).all()   # never lists churned customers


def test_q15_main_risk_drivers_groups_by_feature(sql, t):
    a = t[(t.CustomerStatus == "Active") & t.RiskCategory.isin(["Medium", "High"])].copy()
    a["feature"] = a.TopRiskDriver1.str.split(" = ", n=1).str[0]
    g = a.groupby("feature").agg(active_at_risk_customers=("ChurnFlag", "size"), exp=("ExpectedMonthlyRevenueAtRisk", "sum")).reset_index()
    g["share_pct"] = pct(g.active_at_risk_customers, g.active_at_risk_customers.sum())
    g["expected_monthly_revenue_at_risk"] = g.exp.round(2)
    g = g.rename(columns={"feature": "main_risk_driver"}).sort_values(["active_at_risk_customers", "main_risk_driver"], ascending=[False, True]).head(10)
    same(sql[1]["q15_main_risk_drivers"], g[["main_risk_driver", "active_at_risk_customers", "share_pct", "expected_monthly_revenue_at_risk"]])
    assert not sql[1]["q15_main_risk_drivers"].main_risk_driver.str.contains("=").any()      # grouped by feature, not feature = value
