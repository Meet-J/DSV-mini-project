"""Single source of truth for the Power BI design: star schema, DAX measures, and a pandas replica.

Power BI Desktop (a Windows application) cannot be run here, so nothing in this project pretends to execute DAX or
Power Query M. Instead the design lives in this module as plain data, and everything else is generated from it
(src/powerbi_docs.py writes the DAX file, the Power Query script, the page spec, the wireframes and the expected values).
That keeps the pieces consistent, and lets tests check what CAN be checked here:

  * the star schema is sound (primary keys unique, every foreign key has a match), using a pandas replica of it;
  * every measure and column referenced by a visual or a DAX expression actually exists;
  * every measure has a pandas evaluator, and those values agree with the independent SQL results.

What this cannot prove is that the DAX/M text parses in Power BI Desktop; powerbi/BUILD_GUIDE.md says how to confirm that.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.predict import DATA_DICTIONARY

# ----------------------------------------------------------------------------------------------
# Column types in the predictions CSV (drives the Power Query type step)
# ----------------------------------------------------------------------------------------------
INT_COLUMNS = {"ChurnFlag", "RiskOrder", "tenure", "TenureGroupOrder", "ChargeBandOrder", "NumberOfServices", "ProtectiveAddOnCount"}
NUMBER_COLUMNS = {"ChurnProbability", "MonthlyCharges", "TotalCharges", "ExpectedMonthlyRevenueAtRisk"}
CSV_COLUMNS = [name for name, _ in DATA_DICTIONARY]
ORDER_COLUMNS_MOVED_TO_DIMENSIONS = ["RiskOrder", "TenureGroupOrder", "ChargeBandOrder"]
ADDED_FACT_COLUMNS = ["TopRiskDriverFeature", "ProbabilityBinStart", "ProbabilityBin"]
FACT_COLUMNS = [c for c in CSV_COLUMNS if c not in ORDER_COLUMNS_MOVED_TO_DIMENSIONS] + ADDED_FACT_COLUMNS

# Brand colours (same restrained palette as the Python charts)
SLATE, ACCENT, MUTED, DARK = "#6B7F99", "#C0504D", "#B8C2CF", "#3F5670"


# ----------------------------------------------------------------------------------------------
# Dimensions (static lists: they guarantee order, labels and levels even if a level has no customers)
# ----------------------------------------------------------------------------------------------
@dataclass
class Dimension:
    name: str
    key: str                        # primary key column (text)
    columns: list[str]
    rows: list[tuple]
    sort_by: dict[str, str] = field(default_factory=dict)      # column -> sort-order column
    note: str = ""

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.columns)


ACTIONS = {
    "High risk + high value": "Hypothesis: personal outreach by an account manager to understand needs, then a tailored retention offer. Validate with a small pilot.",
    "High risk + standard value": "Hypothesis: low-cost automated save offer plus an early-life check-in, since these customers are typically new. Validate before scaling.",
    "Medium risk + high value": "Hypothesis: an incentive to move from month-to-month to a longer contract. Test on a sample before a wide rollout.",
    "Medium risk + standard value": "Hypothesis: contract-upgrade or autopay nudges through low-cost channels; test bundling security or support services.",
    "Low risk + high value": "Hypothesis: protect with loyalty benefits and recognition; avoid discounts that may not be needed. Monitor for risk drift.",
    "Low risk + standard value": "Hypothesis: maintain service quality; test cross-selling protective services. No retention spend indicated.",
    "Already churned": "Not a retention target. Use for win-back or root-cause analysis only.",
}
SEGMENTS = list(ACTIONS)

DIMENSIONS: dict[str, Dimension] = {d.name: d for d in [
    Dimension("DimContract", "Contract", ["Contract", "ContractOrder"],
              [("Month-to-month", 1), ("One year", 2), ("Two year", 3)], {"Contract": "ContractOrder"}),
    Dimension("DimInternetService", "InternetService", ["InternetService", "InternetServiceOrder"],
              [("DSL", 1), ("Fiber optic", 2), ("No", 3)], {"InternetService": "InternetServiceOrder"}),
    Dimension("DimPaymentMethod", "PaymentMethod", ["PaymentMethod", "PaymentMethodOrder"],
              [("Electronic check", 1), ("Mailed check", 2), ("Bank transfer (automatic)", 3), ("Credit card (automatic)", 4)],
              {"PaymentMethod": "PaymentMethodOrder"}),
    Dimension("DimTenureGroup", "TenureGroup", ["TenureGroup", "TenureGroupOrder"],
              [("0-12", 1), ("13-24", 2), ("25-48", 3), ("49+", 4)], {"TenureGroup": "TenureGroupOrder"}),
    Dimension("DimChargeBand", "ChargeBand", ["ChargeBand", "ChargeBandOrder"],
              [("<=35", 1), ("35-65", 2), ("65-95", 3), (">95", 4)], {"ChargeBand": "ChargeBandOrder"}),
    Dimension("DimRiskCategory", "RiskCategory", ["RiskCategory", "RiskOrder", "ColorHex", "Definition"],
              [("Low", 1, MUTED, "Predicted probability below 0.30"), ("Medium", 2, "#E8B4B0", "0.30 to 0.70"), ("High", 3, ACCENT, "Above 0.70")],
              {"RiskCategory": "RiskOrder"}),
    Dimension("DimCustomerStatus", "CustomerStatus", ["CustomerStatus", "StatusOrder"],
              [("Active", 1), ("Churned", 2)], {"CustomerStatus": "StatusOrder"}),
    Dimension("DimRetentionSegment", "RetentionSegment", ["RetentionSegment", "SegmentOrder", "RiskTier", "ValueTier", "SuggestedAction"],
              [(seg, i + 1, seg.split(" risk")[0] if " risk" in seg else "n/a", seg.split("+ ")[1].split(" value")[0] if "+ " in seg else "n/a", ACTIONS[seg])
               for i, seg in enumerate(SEGMENTS)], {"RetentionSegment": "SegmentOrder"}),
]}

# (fact column, dimension name): every relationship is many-to-one, dimension -> fact, single direction, active
RELATIONSHIPS = [(dim.key, name) for name, dim in DIMENSIONS.items()]

# Foreign-key columns on the fact table are hidden in Report view (standard star-schema practice), so visuals must use the
# dimension's column instead; tests/test_powerbi.py enforces that no visual references these.
HIDDEN_FACT_COLUMNS = [dim.key for dim in DIMENSIONS.values()]


# ----------------------------------------------------------------------------------------------
# Measures
# ----------------------------------------------------------------------------------------------
@dataclass
class Measure:
    name: str
    dax: str
    folder: str
    fmt: str
    description: str
    evaluator: Callable | None = None       # (filtered fact rows, all fact rows) -> number; None for formatting helpers


def _rate(c, a):
    return c.ChurnFlag.mean() if len(c) else None


def _active(c):
    return c[c.ChurnFlag == 0]


def _tier(c, tiers):
    return _active(c)[_active(c).RiskCategory.isin(tiers)]


MEASURES: list[Measure] = [
    # ---- Customers
    Measure("Total Customers", "COUNTROWS(FactCustomer)", "1 Customers", "#,0",
            "Number of customers in the current filter context. COUNTROWS is correct and faster than DISTINCTCOUNT because CustomerID is unique.",
            lambda c, a: len(c)),
    Measure("Churned Customers", "CALCULATE([Total Customers], FactCustomer[ChurnFlag] = 1)", "1 Customers", "#,0",
            "Customers who have already left (Churn = Yes).", lambda c, a: int((c.ChurnFlag == 1).sum())),
    Measure("Active Customers", "CALCULATE([Total Customers], FactCustomer[ChurnFlag] = 0)", "1 Customers", "#,0",
            "Customers still with the company. All risk and revenue-at-risk measures build on this, because churned customers cannot be retained.",
            lambda c, a: int((c.ChurnFlag == 0).sum())),
    Measure("Churn Rate", "DIVIDE([Churned Customers], [Total Customers])", "1 Customers", "0.0%",
            "Share of customers who churned. DIVIDE returns blank, not 0%, for an empty selection, so an empty slice does not show a misleading 0%.", _rate),
    Measure("Overall Churn Rate", "CALCULATE([Churn Rate], REMOVEFILTERS())", "1 Customers", "0.0%",
            "Churn rate of the whole customer base, ignoring every slicer. Used as the reference line on driver charts.", lambda c, a: a.ChurnFlag.mean()),
    Measure("Average Monthly Charges", "AVERAGE(FactCustomer[MonthlyCharges])", "1 Customers", "#,0.00",
            "Average current monthly charge (currency is not stated in the dataset).", lambda c, a: c.MonthlyCharges.mean() if len(c) else None),
    Measure("Average Tenure (Months)", "AVERAGE(FactCustomer[tenure])", "1 Customers", "0.0",
            "Average months with the company.", lambda c, a: c.tenure.mean() if len(c) else None),
    # ---- Risk (active customers only)
    Measure("High Risk Customers", 'CALCULATE([Active Customers], DimRiskCategory[RiskCategory] = "High")', "2 Risk (active customers)", "#,0",
            "Active customers whose predicted churn probability is above 0.70.", lambda c, a: len(_tier(c, ["High"]))),
    Measure("Medium Risk Customers", 'CALCULATE([Active Customers], DimRiskCategory[RiskCategory] = "Medium")', "2 Risk (active customers)", "#,0",
            "Active customers with probability from 0.30 to 0.70.", lambda c, a: len(_tier(c, ["Medium"]))),
    Measure("Low Risk Customers", 'CALCULATE([Active Customers], DimRiskCategory[RiskCategory] = "Low")', "2 Risk (active customers)", "#,0",
            "Active customers with probability below 0.30.", lambda c, a: len(_tier(c, ["Low"]))),
    Measure("At-Risk Customers", 'CALCULATE([Active Customers], DimRiskCategory[RiskCategory] IN {"Medium", "High"})', "2 Risk (active customers)", "#,0",
            "Active customers at Medium or High risk: the broad pool a retention programme would consider.", lambda c, a: len(_tier(c, ["Medium", "High"]))),
    Measure("High Risk % of Active", "DIVIDE([High Risk Customers], [Active Customers])", "2 Risk (active customers)", "0.0%",
            "High-risk customers as a share of active customers.",
            lambda c, a: len(_tier(c, ["High"])) / len(_active(c)) if len(_active(c)) else None),
    Measure("Average Churn Probability (Active)", "CALCULATE(AVERAGE(FactCustomer[ChurnProbability]), FactCustomer[ChurnFlag] = 0)", "2 Risk (active customers)", "0.0%",
            "Mean out-of-fold churn probability of active customers. A propensity (share of similar customers who left), not a time-bound forecast.",
            lambda c, a: _active(c).ChurnProbability.mean() if len(_active(c)) else None),
    Measure("Average Predicted Probability", "AVERAGE(FactCustomer[ChurnProbability])", "2 Risk (active customers)", "0.0%",
            "Mean predicted probability over ALL customers; compare with Churn Rate by risk tier to see whether the tiers are honest.",
            lambda c, a: c.ChurnProbability.mean() if len(c) else None),
    # ---- Revenue
    Measure("Total Monthly Charges", "SUM(FactCustomer[MonthlyCharges])", "3 Revenue", "#,0",
            "Sum of monthly charges. Revenue here means monthly charges only: no margin or lifetime value in the data.", lambda c, a: c.MonthlyCharges.sum()),
    Measure("Active Monthly Charges", "CALCULATE([Total Monthly Charges], FactCustomer[ChurnFlag] = 0)", "3 Revenue", "#,0",
            "Monthly charges of active customers.", lambda c, a: _active(c).MonthlyCharges.sum()),
    Measure("Churned Monthly Charges", "CALCULATE([Total Monthly Charges], FactCustomer[ChurnFlag] = 1)", "3 Revenue", "#,0",
            "Monthly charges of customers who already churned.", lambda c, a: c[c.ChurnFlag == 1].MonthlyCharges.sum()),
    Measure("Churned Revenue Share", "DIVIDE([Churned Monthly Charges], [Total Monthly Charges])", "3 Revenue", "0.0%",
            "Share of monthly charges belonging to churned customers; larger than the churn rate because churners paid more.",
            lambda c, a: c[c.ChurnFlag == 1].MonthlyCharges.sum() / c.MonthlyCharges.sum() if len(c) else None),
    Measure("Revenue at Risk (High)", 'CALCULATE([Active Monthly Charges], DimRiskCategory[RiskCategory] = "High")', "3 Revenue", "#,0",
            "Monthly charges of ACTIVE High-risk customers: the brief's 'Revenue at Risk'. Small, because the High tier is mostly customers who already left.",
            lambda c, a: _tier(c, ["High"]).MonthlyCharges.sum()),
    Measure("Revenue at Risk (Medium + High)", 'CALCULATE([Active Monthly Charges], DimRiskCategory[RiskCategory] IN {"Medium", "High"})', "3 Revenue", "#,0",
            "Monthly charges of active customers at Medium or High risk.", lambda c, a: _tier(c, ["Medium", "High"]).MonthlyCharges.sum()),
    Measure("Expected Revenue at Risk", "SUM(FactCustomer[ExpectedMonthlyRevenueAtRisk])", "3 Revenue", "#,0",
            "Probability x monthly charge, summed over active customers (the column is 0 for churned customers). Propensity-weighted exposure, not a forecast.",
            lambda c, a: c.ExpectedMonthlyRevenueAtRisk.sum()),
    Measure("Expected Revenue at Risk %", "DIVIDE([Expected Revenue at Risk], [Active Monthly Charges])", "3 Revenue", "0.0%",
            "Expected revenue at risk as a share of active monthly charges.",
            lambda c, a: c.ExpectedMonthlyRevenueAtRisk.sum() / _active(c).MonthlyCharges.sum() if len(_active(c)) else None),
    # ---- Value
    Measure("High-Value Customers", 'CALCULATE([Active Customers], FactCustomer[ValueSegment] = "High value")', "4 Value", "#,0",
            "Active customers whose monthly charge is in the top quartile of all customers.",
            lambda c, a: int(((c.ChurnFlag == 0) & (c.ValueSegment == "High value")).sum())),
    Measure("High-Value At-Risk Customers",
            'CALCULATE([Active Customers], FactCustomer[ValueSegment] = "High value", DimRiskCategory[RiskCategory] IN {"Medium", "High"})',
            "4 Value", "#,0", "High-value active customers at Medium or High risk.",
            lambda c, a: int(((c.ChurnFlag == 0) & (c.ValueSegment == "High value") & c.RiskCategory.isin(["Medium", "High"])).sum())),
    # ---- Formatting helpers (no evaluator: they return colours, not numbers)
    Measure("Churn Bar Color", 'IF([Churn Rate] > [Overall Churn Rate] + 0.02, "' + ACCENT + '", "' + MUTED + '")', "5 Formatting", "",
            "Bar colour: red when a group's churn rate is more than 2 points above the overall rate, grey-blue otherwise (the same rule as the Python charts)."),
    Measure("Risk Color", "MAX(DimRiskCategory[ColorHex])", "5 Formatting", "",
            "Hex colour of the risk tier in the current row or bar, for 'format by field value'."),
    Measure("Unmapped Customers",
            "COUNTROWS(FILTER(FactCustomer, " + " || ".join(f"ISBLANK(RELATED({n}[{d.columns[1]}]))" for n, d in DIMENSIONS.items()) + "))",
            "6 Model checks", "#,0",
            "Customers whose value in some dimension is missing from that dimension table. Must be 0; anything else means a level is missing from a dimension."),
]
MEASURE_NAMES = {m.name for m in MEASURES}

# ----------------------------------------------------------------------------------------------
# Pandas replica of the model
# ----------------------------------------------------------------------------------------------
def build_star(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """What Power Query produces: the fact table plus the eight dimension tables."""
    fact = predictions.copy()
    fact["TopRiskDriverFeature"] = fact["TopRiskDriver1"].str.split(" = ", n=1).str[0]
    start = np.minimum(np.floor(fact["ChurnProbability"] * 20), 19) / 20
    fact["ProbabilityBinStart"] = start
    lo = np.round(start * 100).astype(int)
    fact["ProbabilityBin"] = lo.astype(str) + "-" + (lo + 5).astype(str) + "%"
    fact = fact.drop(columns=ORDER_COLUMNS_MOVED_TO_DIMENSIONS)
    return {"FactCustomer": fact[FACT_COLUMNS], **{name: dim.frame() for name, dim in DIMENSIONS.items()}}


def check_integrity(star: dict[str, pd.DataFrame]) -> list[dict]:
    """Primary-key uniqueness and foreign-key coverage for every relationship."""
    fact = star["FactCustomer"]
    rows = [{"check": "FactCustomer[CustomerID] is unique", "ok": bool(fact["CustomerID"].is_unique), "detail": f"{len(fact):,} rows"}]
    for fk, dim_name in RELATIONSHIPS:
        dim = star[dim_name]
        orphans = int((~fact[fk].isin(dim[fk])).sum())
        rows.append({"check": f"{dim_name}[{fk}] is unique", "ok": bool(dim[fk].is_unique), "detail": f"{len(dim)} rows"})
        rows.append({"check": f"every FactCustomer[{fk}] exists in {dim_name}", "ok": orphans == 0, "detail": f"{orphans} unmatched customers"})
    return rows


def evaluate_measure(name: str, fact: pd.DataFrame, filters: dict | None = None):
    """Evaluate a measure with a pandas filter context ({fact column: value or list of values})."""
    measure = next(m for m in MEASURES if m.name == name)
    if measure.evaluator is None:
        raise ValueError(f"{name} has no numeric evaluator")
    ctx = fact
    for col, values in (filters or {}).items():
        ctx = ctx[ctx[col].isin(values if isinstance(values, (list, tuple, set)) else [values])]
    return measure.evaluator(ctx, fact)


# ----------------------------------------------------------------------------------------------
# Static reference checks used by the tests
# ----------------------------------------------------------------------------------------------
def model_columns() -> dict[str, set[str]]:
    cols = {"FactCustomer": set(FACT_COLUMNS)}
    cols.update({name: set(dim.columns) for name, dim in DIMENSIONS.items()})
    return cols


COLUMN_REF = re.compile(r"\b([A-Za-z_]\w*)\[([^\]]+)\]")
MEASURE_REF = re.compile(r"(?<![\w'])\[([^\]]+)\]")


def dangling_references(text: str) -> list[str]:
    """Table[Column] references that name no real column, and [Measure] references that name no real measure."""
    bad = []
    cols = model_columns()
    for table, col in COLUMN_REF.findall(text):
        if table not in cols or col not in cols[table]:
            bad.append(f"{table}[{col}]")
    stripped = COLUMN_REF.sub("", text)
    for m in MEASURE_REF.findall(stripped):
        if m not in MEASURE_NAMES:
            bad.append(f"[{m}]")
    return bad
