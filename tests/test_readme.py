"""The README quotes many numbers and paths. Every one must trace back to a generated file, so the README cannot drift from the code."""
import json
import re

import pandas as pd
import pytest

from src import config
from src.powerbi_model import DIMENSIONS, MEASURES, RELATIONSHIPS

ROOT = config.PROJECT_ROOT
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _corpus() -> str:
    parts = [(ROOT / "powerbi" / "expected_values.md").read_text(encoding="utf-8")]
    parts += [p.read_text(encoding="utf-8") for p in (ROOT / "output").glob("*.md")]
    nb = json.loads((ROOT / "notebooks" / "churn_analysis.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        parts.append("".join(cell["source"]))
        for out in cell.get("outputs", []):
            parts.append("".join(out.get("text", "")) + "".join(out.get("data", {}).get("text/plain", "")))
    return "\n".join(parts)


CORPUS = _corpus()


# ---------------------------------------------------------------- structure and references
def test_every_section_the_brief_asks_for_is_present():
    headings = set(re.findall(r"^## (.+)$", README, flags=re.M))
    required = ["Overview", "Business Problem", "Objectives", "Dataset", "Technologies Used", "Project Architecture", "Data Cleaning",
                "Exploratory Data Analysis", "Feature Engineering", "Machine Learning", "Model Evaluation", "Churn Prediction", "Risk Segmentation",
                "Power BI Dashboard", "Key Insights", "Business Recommendations", "Project Structure", "How to Run", "Limitations and Future Improvements",
                "Disclaimer / Dataset Source"]
    assert [r for r in required if r not in headings] == []


def test_relative_links_and_images_resolve():
    targets = [t for t in re.findall(r"\]\(([^)]+)\)", README) if not t.startswith(("http", "#"))]
    assert targets and [t for t in targets if not (ROOT / t.split("#")[0]).exists()] == []


def test_paths_in_backticks_exist_except_runtime_ones():
    runtime = {"data/raw/Telco-Customer-Churn.csv", "new_customers.csv", ".venv", "data/Telco-Customer-Churn.csv",
               "IBM/telco-customer-churn-on-icp4d"}      # runtime files, and the name of an external GitHub repository
    paths = {t for t in re.findall(r"`([^`\s]+)`", README) if re.search(r"^[\w.-]+/[\w./-]+$|^[\w-]+\.(md|py|csv|txt|json|ini|sql|m|dax|ipynb|pkl)$", t)}
    missing = [p for p in paths if p not in runtime and not (ROOT / p).exists() and not (ROOT / p.rstrip("/")).exists()]
    assert missing == [], missing


def test_honest_status_statements_are_present():
    for phrase in ["`.pbix` file is not included", "placeholders", "not independently confirmed", "No licence has been chosen", "hypotheses requiring validation"]:
        assert phrase in README, phrase


# ---------------------------------------------------------------- numbers traced to generated files
def _rows_for(label: str) -> list[str]:
    return [l for l in README.splitlines() if re.match(rf"\| \**{re.escape(label)}", l)]


def test_model_table_matches_model_comparison_csv():
    """All five metrics must appear together in ONE table row (a value elsewhere in the README must not satisfy the check)."""
    comparison = pd.read_csv(ROOT / "output" / "model_comparison.csv")
    for _, r in comparison.iterrows():
        tokens = [f"{r[col]:.4f}" for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]]
        assert any(all(t in line for t in tokens) for line in _rows_for(r.model)), (r.model, tokens)
    rf = comparison.set_index("model").loc["Random Forest"]
    assert any(f"{rf.roc_auc_ci_low:.3f} to {rf.roc_auc_ci_high:.3f}" in line for line in _rows_for("Random Forest"))


def test_operating_point_matches_the_model_report():
    report = (ROOT / "output" / "model_report.md").read_text(encoding="utf-8")
    row = r"\| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|"
    for label, readme_prefix in [("0.33 (chosen)", "| **0.33** |"), ("0.50 (default)", "| 0.50 |")]:
        _, precision, recall, f1, tn, fp, fn, tp = re.search(rf"\| {re.escape(label)} {row}", report).groups()
        line = next(l for l in README.splitlines() if l.startswith(readme_prefix) and "of 374" in l)
        for token in [f"{float(precision):.3f}", f"{float(recall):.3f}", f"{tp} of {int(tp) + int(fn)}", f"| {fp} |"]:
            assert token in line, (label, token, line)
        if "chosen" in label:
            assert f"{float(f1):.3f}" in line
    for token in ["0.8504", "0.8482", "0.8451", "0.8452", "0.267", "0.265"]:
        assert token in report and token in README, token


def test_class_imbalance_table_matches_the_csv():
    for _, r in pd.read_csv(ROOT / "output" / "imbalance_check.csv", keep_default_na=False).iterrows():   # "None" is a real value here, not a missing one
        row = [f"{r.roc_auc:.4f}", f"{r.customers_flagged_pct:.1f}%", f"{r.brier:.4f}", f"{r.mean_predicted_prob:.3f}"]
        line = next(l for l in README.splitlines() if l.startswith(f"| {r.model} | {'none' if r.class_weight == 'None' else r.class_weight} |"))
        assert all(tok in line for tok in row), (r.model, r.class_weight, line)


def test_risk_tables_match_the_predictions_file():
    t = pd.read_csv(config.PREDICTIONS_PATH)
    a = t[t.CustomerStatus == "Active"]
    assert f"{len(t):,}" in README and f"{len(a):,}" in README and f"{int(t.ChurnFlag.sum()):,}" in README
    for tier in ["Low", "Medium", "High"]:
        g, ga = t[t.RiskCategory == tier], a[a.RiskCategory == tier]
        line = next(l for l in README.splitlines() if l.startswith(f"| {tier} | {len(g):,} |"))
        assert f"{g.ChurnProbability.mean():.1%}" in line and f"{g.ChurnFlag.mean():.1%}" in line, line
        line2 = next(l for l in README.splitlines() if l.startswith(f"| {tier} | {len(ga):,} |"))
        assert f"{ga.MonthlyCharges.sum():,.2f}" in line2 and f"{ga.ExpectedMonthlyRevenueAtRisk.sum():,.2f}" in line2, line2
    assert f"{a.ExpectedMonthlyRevenueAtRisk.sum():,.2f}" in README and f"{a.ExpectedMonthlyRevenueAtRisk.sum() / a.MonthlyCharges.sum():.1%}" in README
    assert f"{t.loc[t.ValueSegment == 'High value', 'MonthlyCharges'].min():.2f}" in README
    high = a[a.RiskCategory == "High"]
    assert (high.ChurnFlag == 0).all() and int(((t.RiskCategory == "High") & (t.ChurnFlag == 1)).sum()) == 297 and "297" in README
    for seg, n in a.RetentionSegment.value_counts().items():
        assert f"| {n:,} |" in README, seg


def test_segment_tenure_claim_is_true():
    a = pd.read_csv(config.PREDICTIONS_PATH).query("CustomerStatus == 'Active' and RiskCategory == 'High'")
    assert f"{a.tenure.mean():.1f}" in README and f"{a[a.ValueSegment == 'Standard value'].tenure.mean():.1f}" in README


def test_eda_and_sql_figures_appear_in_their_sources():
    for token in ["42.7%", "11.3%", "2.8%", "55.0%", "88.6%", "47.4%", "9.5%", "55.5%", "51.4%", "26.0%", "41.9%", "19.0%", "7.4%", "48.8%", "7.1%",
                  "56.7%", "5.3%", "45.3%", "53.7%", "26.9%", "26.2%", "30.5%", "0.9988", "9.51"]:
        assert token in README and token in CORPUS, token
    sql = (ROOT / "output" / "sql_results.md").read_text(encoding="utf-8")
    assert "| 1 | 705 | 530 | 75.2 | 28.4 | 2.83 |" in sql and "| 4 | 704 | 255 | 36.2 | 80.0 | 1.36 |" in sql
    seventh = re.search(r"\| 7 \| 704 \| 69 \| 9\.8 \| ([\d.]+) \|", sql).group(1)
    assert f"{100 - float(seventh):.1f}%" in README and "28.4%" in README and "2.83" in README and "80.0%" in README
    assert "13228.34" in sql and "13,228.34" in README


def test_interpretation_figures_match_their_csvs():
    perm = pd.read_csv(ROOT / "output" / "permutation_importance.csv").set_index("feature").importance_mean
    for feature in ["Contract", "tenure", "InternetService", "PaymentMethod"]:
        assert f"{perm[feature]:.3f}" in README, feature
    effects = pd.read_csv(ROOT / "output" / "level_effects.csv").set_index(["feature", "level"]).mean_effect
    for key in [("Contract", "Month-to-month"), ("Contract", "Two year"), ("InternetService", "Fiber optic"), ("InternetService", "DSL"),
                ("tenure (by band, months)", "0-12"), ("tenure (by band, months)", "49+"), ("PaymentMethod", "Electronic check")]:
        assert f"{abs(effects[key]):.1f}" in README, key
    odds = pd.read_csv(ROOT / "output" / "odds_ratios.csv").set_index(["feature", "comparison"])
    for key in [("InternetService", "Fiber optic vs DSL"), ("Contract", "Two year vs Month-to-month"), ("tenure", "+12 months"),
                ("OnlineSecurity", "Yes vs No"), ("TechSupport", "Yes vs No")]:
        r = odds.loc[key]
        assert f"{r.odds_ratio:.2f} ({r.or_low:.2f} to {r.or_high:.2f})" in README or f"{r.odds_ratio:.2f}" in README, key
    interp = (ROOT / "output" / "model_interpretation.md").read_text(encoding="utf-8")
    assert "18 of 19" in interp and "18 of 19" in README


def test_scoring_quality_matches_the_risk_report():
    risk = (ROOT / "output" / "risk_segmentation_report.md").read_text(encoding="utf-8")
    for token in ["0.8455", "0.6525", "0.1356"]:
        assert token in risk and token in README, token


def test_power_bi_counts_match_the_code():
    assert f"{len(MEASURES)} DAX measures" in README or f"{len(MEASURES)} documented DAX measures" in README
    assert f"{len(DIMENSIONS)} dimensions" in README and f"{len(RELATIONSHIPS)} many-to-one relationships" in README
    assert f"all {len(DIMENSIONS) + 2} queries" in README
    assert "16 documented queries" in README and len(re.findall(r"^-- name:", (ROOT / "sql" / "churn_analysis.sql").read_text(encoding="utf-8"), flags=re.M)) == 16
