"""Tests for the Power BI design. Power BI Desktop cannot run here, so these check everything that CAN be checked:
the star schema (pandas replica), references between spec / DAX / model, layout geometry, agreement with the SQL
results, and that the committed artefacts are exactly what the generator produces."""
import itertools
import json
import re

import pandas as pd
import pytest

from src import config
from src.powerbi_docs import POWERBI_DIR, data_model_text, dax_text, expected_values_text, power_query_text, spec_text
from src.powerbi_model import (
    CSV_COLUMNS, DIMENSIONS, FACT_COLUMNS, HIDDEN_FACT_COLUMNS, MEASURE_NAMES, MEASURES, RELATIONSHIPS,
    build_star, check_integrity, dangling_references, evaluate_measure, model_columns,
)
from src.powerbi_pages import CANVAS_H, CANVAS_W, PAGES, all_visuals
from src.predict import retention_segment  # noqa: F401  (segment names must match what predict.py produces)
from src.run_sql import SQL_PATH, load_database, parse_queries, run_queries

pytestmark = pytest.mark.skipif(not config.PREDICTIONS_PATH.exists(), reason="Run `python -m src.predict` first.")
FORMAT_ONLY = {"Churn Bar Color", "Risk Color", "Unmapped Customers"}


@pytest.fixture(scope="module")
def predictions():
    return pd.read_csv(config.PREDICTIONS_PATH)


@pytest.fixture(scope="module")
def star(predictions):
    return build_star(predictions)


@pytest.fixture(scope="module")
def sql():
    qs = parse_queries(SQL_PATH.read_text(encoding="utf-8"))
    return run_queries(load_database(), qs)


# ---------------------------------------------------------------- the star schema
def test_star_schema_integrity(star, predictions):
    failures = [r for r in check_integrity(star) if not r["ok"]]
    assert not failures, failures
    fact = star["FactCustomer"]
    assert len(fact) == len(predictions) == 7043 and list(fact.columns) == FACT_COLUMNS
    assert len(CSV_COLUMNS) == 41 and len(RELATIONSHIPS) == len(DIMENSIONS) == 8


def test_integrity_check_actually_detects_an_orphan_and_a_duplicate_key(predictions):
    broken = build_star(predictions)
    broken["FactCustomer"].loc[0, "Contract"] = "Weekly"                                  # a level no dimension knows
    broken["DimTenureGroup"] = pd.concat([broken["DimTenureGroup"], broken["DimTenureGroup"].iloc[[0]]])   # duplicate key
    failed = {r["check"] for r in check_integrity(broken) if not r["ok"]}
    assert "every FactCustomer[Contract] exists in DimContract" in failed and "DimTenureGroup[TenureGroup] is unique" in failed


def test_retention_segments_match_what_the_scoring_code_produces(star, predictions):
    assert set(DIMENSIONS["DimRetentionSegment"].frame().RetentionSegment) == set(predictions.RetentionSegment.unique())


def test_every_sort_by_column_maps_one_to_one(star):
    for dim in DIMENSIONS.values():
        frame = dim.frame()
        for col, order in dim.sort_by.items():
            assert frame[col].is_unique and frame[order].is_unique, (dim.name, col)   # Power BI requires a 1:1 sort-by column


# ---------------------------------------------------------------- references between spec, DAX and model
def test_no_dax_expression_references_a_missing_column_or_measure():
    assert dangling_references("\n".join(m.dax for m in MEASURES)) == []
    assert len(MEASURE_NAMES) == len(MEASURES)


def test_dangling_reference_detector_catches_typos():
    assert dangling_references("SUM(FactCustomer[MonthlyCharge])") == ["FactCustomer[MonthlyCharge]"]
    assert dangling_references("[Churn Rat] + [Churn Rate]") == ["[Churn Rat]"]


def test_every_visual_field_exists_and_no_visual_uses_a_hidden_foreign_key():
    hidden = {f"FactCustomer[{c}]" for c in HIDDEN_FACT_COLUMNS}
    for page, v in all_visuals():
        refs = [f for fs in v.fields.values() for f in fs] + v.filters
        assert dangling_references(" ".join(refs)) == [], (page, v.id)
        used = {f for fs in v.fields.values() for f in fs}
        assert not (used & hidden), (page, v.id, used & hidden)


def test_every_numeric_measure_has_a_pandas_evaluator():
    assert {m.name for m in MEASURES if m.evaluator is None} == FORMAT_ONLY


# ---------------------------------------------------------------- layout
def separated(a, b) -> bool:
    """True if two visuals do not overlap (touching edges is fine)."""
    return a.x + a.w <= b.x or b.x + b.w <= a.x or a.y + a.h <= b.y or b.y + b.h <= a.y


def test_visuals_stay_on_the_canvas_and_never_overlap():
    for page in PAGES:
        ids = [v.id for v in page.visuals]
        assert len(ids) == len(set(ids)) and page.visuals[0].kind == "text"
        for v in page.visuals:
            assert v.x >= 0 and v.y >= 0 and v.x + v.w <= CANVAS_W and v.y + v.h <= CANVAS_H, (page.name, v.id)
        for a, b in itertools.combinations(page.visuals, 2):
            assert separated(a, b), f"{page.name}: {a.id} overlaps {b.id}"


def test_overlap_detector_flags_an_overlap_and_allows_touching_edges():
    first, second = PAGES[0].visuals[1], PAGES[0].visuals[2]
    assert separated(first, second)                                              # the real layout
    overlapping = type(second)(**{**second.__dict__, "x": first.x + 10})
    assert not separated(first, overlapping)                                     # shifted onto its neighbour
    touching = type(second)(**{**second.__dict__, "x": first.x + first.w, "y": first.y})
    assert separated(first, touching)


def test_page_structure_matches_the_brief():
    assert [p.name for p in PAGES[:4]] == ["Executive Overview", "Churn Drivers", "Customer Risk", "Retention Strategy"]
    assert PAGES[4].hidden and sum(not p.hidden for p in PAGES) == 4
    page1_cards = [v.title for v in PAGES[0].visuals if v.kind == "card"]
    assert page1_cards == ["Total Customers", "Churned Customers", "Churn Rate", "Average Monthly Charges", "Average Tenure (Months)",
                           "High Risk Customers", "Revenue at Risk (High)"]


# ---------------------------------------------------------------- agreement with the independent SQL results
def test_measures_agree_with_the_sql_queries(star, sql):
    fact = star["FactCustomer"]
    q01 = sql["q01_headline"].iloc[0]
    assert evaluate_measure("Total Customers", fact) == q01.total_customers and evaluate_measure("Churned Customers", fact) == q01.churned_customers
    assert round(100 * evaluate_measure("Churn Rate", fact), 2) == q01.churn_rate_pct
    assert round(evaluate_measure("Average Monthly Charges", fact), 2) == q01.avg_monthly_charges

    for _, r in sql["q02_churn_by_contract"].iterrows():
        assert round(100 * evaluate_measure("Churn Rate", fact, {"Contract": r.Contract}), 1) == r.churn_rate_pct

    for _, r in sql["q12_revenue_exposure_by_risk"].iterrows():
        f = {"RiskCategory": r.RiskCategory}
        assert evaluate_measure("Active Customers", fact, f) == r.active_customers
        assert round(evaluate_measure("Active Monthly Charges", fact, f), 2) == r.monthly_charges
        assert round(evaluate_measure("Expected Revenue at Risk", fact, f), 2) == r.expected_monthly_revenue_at_risk

    for _, r in sql["q13_retention_segments"].iterrows():
        f = {"RetentionSegment": r.RetentionSegment}
        assert evaluate_measure("Active Customers", fact, f) == r.active_customers
        assert round(evaluate_measure("Expected Revenue at Risk", fact, f), 2) == r.expected_revenue_at_risk

    for _, r in sql["q15_main_risk_drivers"].iterrows():
        assert evaluate_measure("At-Risk Customers", fact, {"TopRiskDriverFeature": r.main_risk_driver}) == r.active_at_risk_customers


def test_the_brief_headline_measures_have_the_verified_values(star):
    fact = star["FactCustomer"]
    assert evaluate_measure("High Risk Customers", fact) == 65 and evaluate_measure("Medium Risk Customers", fact) == 1168
    assert evaluate_measure("Revenue at Risk (High)", fact) == pytest.approx(5317.85)
    assert evaluate_measure("Expected Revenue at Risk", fact) == pytest.approx(69018.94, abs=0.01)
    assert evaluate_measure("Overall Churn Rate", fact, {"Contract": "Two year"}) == evaluate_measure("Churn Rate", fact)   # ignores the filter


# ---------------------------------------------------------------- the committed artefacts
def test_committed_files_are_exactly_what_the_generator_produces(predictions):
    assert (POWERBI_DIR / "dax_measures.dax").read_text(encoding="utf-8") == dax_text()
    assert (POWERBI_DIR / "power_query.m").read_text(encoding="utf-8") == power_query_text()
    assert (POWERBI_DIR / "dashboard_spec.md").read_text(encoding="utf-8") == spec_text()
    assert (POWERBI_DIR / "expected_values.md").read_text(encoding="utf-8") == expected_values_text(predictions)
    assert (POWERBI_DIR / "data_model.md").read_text(encoding="utf-8") == data_model_text(predictions)


def _strip_m(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r'"(?:[^"]|"")*"', '""', text)


def test_power_query_text_is_structurally_sound():
    m = power_query_text()
    body = _strip_m(m)
    for open_, close in ["()", "[]", "{}"]:
        assert body.count(open_) == body.count(close), open_
    assert len(re.findall(r"^let$", body, flags=re.M)) == len(re.findall(r"^in$", body, flags=re.M)) == 2 + len(DIMENSIONS)
    for col in CSV_COLUMNS:
        assert f'"{col}"' in m                                              # every CSV column is typed
    for dim in DIMENSIONS.values():
        for row in dim.rows:
            for value in row:
                assert (f'"{value}"' if isinstance(value, str) else str(value)) in m, (dim.name, value)   # every level and action is present


def test_dax_text_is_structurally_sound():
    text = dax_text()
    definitions = re.findall(r"^(?!//)(.+?) =\n((?:    .+\n?|        .+\n?)+)", text, flags=re.M)
    assert [n for n, _ in definitions] == [m.name for m in MEASURES]
    for name, expr in definitions:
        assert expr.count("(") == expr.count(")") and expr.count("{") == expr.count("}"), name
    assert "\t" not in text


def test_theme_is_valid_json_with_the_project_palette():
    theme = json.loads((POWERBI_DIR / "theme.json").read_text(encoding="utf-8"))
    assert all(re.fullmatch(r"#[0-9A-Fa-f]{6}", c) for c in theme["dataColors"])
    assert "#C0504D" in theme["dataColors"] and "#6B7F99" in theme["dataColors"]


def test_wireframes_exist_for_every_page():
    assert len(list((POWERBI_DIR / "wireframes").glob("page*.png"))) == len(PAGES)


def test_numbers_quoted_in_the_build_guide_appear_in_the_expected_values():
    """The guide is hand-written; every figure it quotes must match the generated expected values, so it cannot drift."""
    guide = (POWERBI_DIR / "BUILD_GUIDE.md").read_text(encoding="utf-8")
    expected = (POWERBI_DIR / "expected_values.md").read_text(encoding="utf-8")
    for token in ["7,043", "3,875", "1,655", "42.7%", "3,941", "1,168", "26.5%", "56.7%", "5.3%", "5,318"]:
        assert token in guide, f"guide no longer quotes {token}"
        assert token in expected, f"{token} is quoted in the guide but not in the expected values"
    assert "5,174" in guide and "65 of 5,174" in guide
    assert len(MEASURES) == 27 and "27 documented DAX measures" in guide
