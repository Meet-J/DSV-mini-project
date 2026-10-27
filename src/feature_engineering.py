"""Feature engineering and preprocessing pipeline.

Two kinds of transformation live here, kept deliberately separate:

1. STATELESS features (`add_engineered_features`): computed from a single row only, so they
   cannot leak information between rows. Safe to apply before or after the train/test split.
2. FITTED preprocessing (`build_preprocessor`): scaling and one-hot encoding learn from data
   (means, category lists). These live inside a scikit-learn Pipeline so they are fitted on
   training data only; the test set is only ever *transformed*.

The full pipeline accepts the raw cleaned DataFrame and returns predictions, so the same object
can be used for training, cross-validation and scoring new customers.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config

# ----------------------------------------------------------------------------------------------
# Column groups (after cleaning)
# ----------------------------------------------------------------------------------------------
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_FEATURES = ["SeniorCitizen"]  # already 0/1
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
INTERNET_ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
PROTECTIVE_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]

# Engineered features. Each began as an EDA hypothesis (notebooks/churn_analysis.ipynb, section 10).
# The ablation (section 11, src/feature_ablation.py) found that none improves cross-validated accuracy
# beyond noise, so they are NOT model inputs by default (DEFAULT_ENGINEERED below). They remain
# available for descriptive use and for re-testing.
TENURE_BINS = [-1, 12, 24, 48, np.inf]          # open-ended top bin so tenure > 72 never becomes NaN
TENURE_LABELS = ["0-12", "13-24", "25-48", "49+"]
ENGINEERED_FEATURES = ("TenureGroup", "ProtectiveAddOnCount", "AutoPay")

# Variables the EDA found nearly unrelated to churn. The ablation (src/feature_ablation.py) showed that
# dropping all five together costs ~0.002 ROC-AUC, so they are kept, except `gender` (see below).
LOW_SIGNAL_FEATURES = ["gender", "PhoneService", "MultipleLines", "StreamingTV", "StreamingMovies"]

# Final model feature set (evidence: output/feature_ablation.md, notebook section 11):
#  - Engineered features are excluded: no measurable accuracy gain, so no added complexity.
DEFAULT_ENGINEERED: tuple[str, ...] = ()
#  - TotalCharges is dropped: it is roughly tenure x MonthlyCharges (VIF 9.5), which makes linear-model
#    coefficients unstable. The accuracy cost is within noise. It stays in the data for reporting.
#  - gender is dropped: no association with churn, no accuracy cost to remove, and one fewer protected
#    attribute in a retention model. The other four low-signal columns are KEPT: removing them as a
#    group lowered ROC-AUC.
#  - MonthlyCharges is dropped: the monthly bill is almost entirely determined by the services a customer
#    has (R^2 = 0.9988 against the other model inputs, VIF ~ 860), so it duplicates the service columns and
#    makes their coefficients unstable. Removing it changes cross-validated AUC by no more than noise in
#    logistic regression, random forest and gradient boosting. It stays in the data for revenue analysis.
DEFAULT_DROP = ("TotalCharges", "gender", "MonthlyCharges")

_ALL_KNOWN = set(NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES + list(ENGINEERED_FEATURES))


# ----------------------------------------------------------------------------------------------
# 1. Stateless feature engineering
# ----------------------------------------------------------------------------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with redundant levels collapsed and engineered columns added.

    Row-wise only: nothing here is learned from the data, so it cannot cause leakage.
    """
    required = {"tenure", "PaymentMethod", "MultipleLines", *INTERNET_ADDONS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for feature engineering: {sorted(missing)}")

    out = df.copy()

    # Collapse 'No phone service' / 'No internet service' into 'No'. Those levels repeat what
    # PhoneService / InternetService already say, which would make one-hot columns linearly
    # dependent for linear models. The information is NOT lost: it stays in PhoneService and
    # InternetService.
    out["MultipleLines"] = out["MultipleLines"].replace({"No phone service": "No"})
    for col in INTERNET_ADDONS:
        out[col] = out[col].replace({"No internet service": "No"})

    # TenureGroup: churn falls steeply and non-linearly with tenure (EDA section 3). Bands let
    # a linear model capture that shape; tree models can find the cut-points on their own.
    out["TenureGroup"] = pd.cut(out["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)

    # ProtectiveAddOnCount (0-4): churn falls steadily as protective services accumulate
    # (EDA section 10). Streaming services are excluded because they showed almost no signal.
    # Customers without internet get 0; the InternetService column separates them from
    # internet customers who simply have no add-ons.
    out["ProtectiveAddOnCount"] = (out[PROTECTIVE_ADDONS] == "Yes").sum(axis=1).astype(int)

    # AutoPay: automatic card/bank payment churns far less than manual methods (EDA section 7);
    # the two automatic levels behave alike, so one flag summarises them.
    out["AutoPay"] = out["PaymentMethod"].str.contains("automatic", regex=False).astype(int)
    return out


# ----------------------------------------------------------------------------------------------
# 2. Fitted preprocessing (leakage-safe because it lives inside the Pipeline)
# ----------------------------------------------------------------------------------------------
def build_preprocessor(
    scale_numeric: bool = True,
    engineered: Sequence[str] = DEFAULT_ENGINEERED,
    drop: Sequence[str] = DEFAULT_DROP,
) -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categoricals.

    Args:
        scale_numeric: Standardise numeric columns. Needed by logistic regression (regularised,
            scale-sensitive); irrelevant for tree models, which are invariant to scaling.
        engineered: Which engineered features to include. Defaults to none (see DEFAULT_ENGINEERED).
        drop: Base or engineered feature names to exclude. Defaults to DEFAULT_DROP.
    """
    unknown = (set(engineered) | set(drop)) - _ALL_KNOWN
    if unknown:
        raise ValueError(f"Unknown feature names: {sorted(unknown)}")
    keep = lambda names: [c for c in names if c not in drop]  # noqa: E731

    numeric = keep(NUMERIC_FEATURES + [f for f in ["ProtectiveAddOnCount"] if f in engineered])
    binary = keep(BINARY_FEATURES + [f for f in ["AutoPay"] if f in engineered])
    categorical = keep(CATEGORICAL_FEATURES + [f for f in ["TenureGroup"] if f in engineered])

    transformers = []
    if numeric:
        transformers.append(("numeric", StandardScaler() if scale_numeric else "passthrough", numeric))
    if binary:
        transformers.append(("binary", "passthrough", binary))
    if categorical:
        # drop="if_binary": Yes/No columns become one 0/1 column. handle_unknown="ignore": a category
        # never seen in training encodes as all zeros instead of crashing at prediction time.
        transformers.append((
            "categorical",
            OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
            categorical,
        ))

    # remainder="drop" is intentional: customerID and anything not listed above never reaches the model.
    preprocessor = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    preprocessor.set_output(transform="pandas")  # keep readable column names for interpretability
    return preprocessor


def build_pipeline(model: BaseEstimator, scale_numeric: bool = True, **preprocessor_kwargs) -> Pipeline:
    """engineer features -> preprocess -> model, as one object that takes the cleaned DataFrame."""
    return Pipeline([
        ("engineer", FunctionTransformer(add_engineered_features)),
        ("preprocess", build_preprocessor(scale_numeric=scale_numeric, **preprocessor_kwargs)),
        ("model", model),
    ])


# ----------------------------------------------------------------------------------------------
# 3. Target and split
# ----------------------------------------------------------------------------------------------
def encode_target(churn: pd.Series) -> pd.Series:
    """Churn Yes/No -> 1/0. Unexpected labels raise instead of silently becoming 0."""
    unexpected = set(churn.unique()) - {"Yes", "No"}
    if unexpected:
        raise ValueError(f"Unexpected Churn labels: {unexpected}")
    return (churn == "Yes").astype(int)


def make_train_test_split(
    df: pd.DataFrame, test_size: float = config.TEST_SIZE, seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split. X is indexed by customerID and contains no target and no ID column.

    Stratifying keeps the churn rate (~26.5%) the same in both parts, which matters for an
    imbalanced target: an unlucky random split could otherwise distort evaluation.
    """
    indexed = df.set_index(config.ID_COL)
    y = encode_target(indexed[config.TARGET_COL])
    X = indexed.drop(columns=[config.TARGET_COL])
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
