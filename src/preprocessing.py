"""Data cleaning for the Telco Customer Churn dataset.

Design rule: every change to the data is explicit, justified, and written to a log.
Nothing is modified silently, and if the data violates an assumption the code stops
with an error instead of "fixing" it quietly.

Run from the project root:
    python -m src.preprocessing
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

# Columns whose levels we expect to be a small fixed set; used to catch inconsistent categories.
EXPECTED_LEVELS = {
    "gender": {"Male", "Female"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}
INTERNET_ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


@dataclass
class LogEntry:
    """One documented decision: what changed, why, and how it affects later work."""
    name: str
    what: str
    why: str
    rows_affected: int
    effect: str


def load_raw_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as shipped (no type coercion) and check the schema."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run `python -m src.download_data` first."
        )
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw file is missing expected columns: {missing}")
    return df


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[LogEntry]]:
    """Apply the cleaning transformations. Returns (clean_df, log_entries)."""
    df = raw.copy()
    log: list[LogEntry] = []

    # --- 1. Strip stray whitespace in text columns -------------------------------------
    # Why first: blank cells like " " are invisible to isna() and would hide in later steps.
    # We only report changes; for this file the only affected cells are the blank TotalCharges.
    text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    stripped = df[text_cols].apply(lambda col: col.str.strip())
    cells_changed = int((stripped != df[text_cols]).sum().sum())
    df[text_cols] = stripped
    log.append(LogEntry(
        name="Trim whitespace in text columns",
        what=f"Removed leading/trailing spaces from {len(text_cols)} text columns.",
        why="Whitespace-only cells look like values but are really missing data; "
            "trailing spaces also create fake duplicate categories.",
        rows_affected=cells_changed,
        effect="No category labels changed; blank TotalCharges cells became empty strings "
               "so they can be detected exactly in the next step.",
    ))

    # --- 2. TotalCharges: text -> number, with blanks handled explicitly ----------------
    blank_mask = df["TotalCharges"].eq("")
    numeric_total = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Guard 1: every unparseable value must be a blank. Anything else is a surprise -> stop.
    unexpected = numeric_total.isna() & ~blank_mask
    if unexpected.any():
        raise ValueError(
            f"{int(unexpected.sum())} non-blank TotalCharges values could not be parsed: "
            f"{df.loc[unexpected, 'TotalCharges'].unique()[:5]}"
        )
    # Guard 2: the 'brand-new customer, not yet billed' interpretation only holds if
    # every blank row has tenure == 0. If not, filling with 0 would be unjustified.
    not_new = blank_mask & (df["tenure"] != 0)
    if not_new.any():
        raise ValueError(
            f"{int(not_new.sum())} blank TotalCharges rows have tenure > 0; "
            "cannot treat them as not-yet-billed customers."
        )

    n_blank = int(blank_mask.sum())
    df["TotalCharges"] = numeric_total.fillna(0.0).astype(float)
    log.append(LogEntry(
        name="TotalCharges: convert to numeric, set blanks to 0",
        what=f"Converted TotalCharges from text to float. {n_blank} blank cells were set to 0.0.",
        why="All blank rows have tenure = 0 (checked in code), i.e. customers who have not "
            "completed a billing cycle, so cumulative charges of 0 is the consistent value. "
            "Mean/median imputation would invent revenue that was never billed; dropping "
            "the rows would discard 0.16% of customers for no analytical gain. "
            "This is an interpretation of the pattern, not something the dataset states.",
        rows_affected=n_blank,
        effect="Column becomes numeric and usable in models and Power BI. All affected "
               "customers have Churn = No, so the target is unaffected; they are brand-new "
               "customers, so tenure-based charts will show a tenure-0 group of 11.",
    ))

    # --- 3. Deliberate non-changes (documented so they are decisions, not oversights) ---
    log.append(LogEntry(
        name="Kept 'No phone service' / 'No internet service' levels",
        what="No change in the cleaned file.",
        why="These levels are valid and informative for EDA/Power BI. They duplicate "
            "PhoneService / InternetService, which adds collinearity for linear models.",
        rows_affected=0,
        effect="Collapsed to 'No' later, inside feature engineering, where the reason "
               "can be documented next to the modelling code.",
    ))
    log.append(LogEntry(
        name="Kept SeniorCitizen as 0/1; Churn as Yes/No",
        what="No change.",
        why="Already clean and unambiguous. Human-readable labels for Power BI and "
            "numeric encoding for models are produced downstream.",
        rows_affected=0,
        effect="None.",
    ))
    return df, log


def validate_clean_data(df: pd.DataFrame) -> list[str]:
    """Check the cleaned data against explicit expectations. Raises on failure.

    Returns a list of human-readable checks that passed, for the cleaning log.
    """
    passed: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"Validation failed: {message}")
        passed.append(message)

    check(df.shape[1] == len(EXPECTED_COLUMNS), f"{df.shape[1]} columns as expected")
    check(not df.isna().any().any(), "no missing values in any column")
    check(not df.duplicated().any(), "no fully duplicated rows")
    check(df[config.ID_COL].is_unique, "customerID is unique")
    check(df[config.ID_COL].str.fullmatch(r"\d{4}-[A-Z]{5}").all(), "customerID format is ####-AAAAA for every row")
    check(pd.api.types.is_float_dtype(df["TotalCharges"]), "TotalCharges is numeric")
    check((df["tenure"] >= 0).all() and (df["tenure"] <= 72).all(), "tenure within 0-72 months")
    check((df["MonthlyCharges"] > 0).all(), "MonthlyCharges strictly positive")
    check((df["TotalCharges"] >= 0).all(), "TotalCharges non-negative")
    check(df["SeniorCitizen"].isin([0, 1]).all(), "SeniorCitizen only contains 0/1")
    for col, allowed in EXPECTED_LEVELS.items():
        extra = set(df[col].unique()) - allowed
        check(not extra, f"{col} contains only expected categories")
    check(
        ((df["PhoneService"] == "No") == (df["MultipleLines"] == "No phone service")).all(),
        "MultipleLines is 'No phone service' exactly when PhoneService = No",
    )
    no_internet = df["InternetService"] == "No"
    check(
        all(((df[c] == "No internet service") == no_internet).all() for c in INTERNET_ADDONS),
        "internet add-ons are 'No internet service' exactly when InternetService = No",
    )
    return passed


def write_cleaning_log(
    path: Path, raw: pd.DataFrame, clean: pd.DataFrame,
    log: list[LogEntry], checks: list[str],
) -> None:
    """Write the cleaning decisions to Markdown so they can be reviewed (and linked from the README)."""
    lines = [
        "# Data Cleaning Log",
        "",
        f"Raw: {raw.shape[0]:,} rows x {raw.shape[1]} columns. "
        f"Clean: {clean.shape[0]:,} rows x {clean.shape[1]} columns. No rows were dropped.",
        "",
        "## Transformations and decisions",
        "",
    ]
    for i, e in enumerate(log, 1):
        lines += [
            f"### {i}. {e.name}",
            f"- **What:** {e.what}",
            f"- **Why:** {e.why}",
            f"- **Rows/cells affected:** {e.rows_affected}",
            f"- **Effect on analysis/modelling:** {e.effect}",
            "",
        ]
    lines += ["## Validation checks passed on the cleaned data", ""]
    lines += [f"- {c}" for c in checks]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raw = load_raw_data()
    clean, log = clean_data(raw)
    checks = validate_clean_data(clean)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(config.CLEAN_DATA_PATH, index=False)
    write_cleaning_log(config.CLEANING_LOG_PATH, raw, clean, log, checks)

    logger.info("Loaded raw data:   %s", raw.shape)
    logger.info("Cleaned data:      %s -> %s", clean.shape, config.CLEAN_DATA_PATH.name)
    logger.info("Validation checks: %d passed", len(checks))
    logger.info("Cleaning log:      %s", config.CLEANING_LOG_PATH.name)


if __name__ == "__main__":
    main()
