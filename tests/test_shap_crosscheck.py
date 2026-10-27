"""Optional cross-check: our closed-form per-customer contributions must equal SHAP's LinearExplainer.

Skipped automatically when the `shap` package is not installed (it is not a project requirement).
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

shap = pytest.importorskip("shap")

from src import config
from src.feature_engineering import build_pipeline, make_train_test_split


def test_closed_form_contributions_equal_shap_linear_explainer():
    if not config.CLEAN_DATA_PATH.exists():
        pytest.skip("Run `python -m src.preprocessing` first.")
    X_tr, X_te, y_tr, _ = make_train_test_split(pd.read_csv(config.CLEAN_DATA_PATH))
    pipe = build_pipeline(LogisticRegression(max_iter=2000, random_state=config.RANDOM_SEED, C=1)).fit(X_tr, y_tr)
    engineer, pre, model = pipe.named_steps["engineer"], pipe.named_steps["preprocess"], pipe.named_steps["model"]
    Z_bg, Z_te = pre.transform(engineer.transform(X_tr)), pre.transform(engineer.transform(X_te))

    ours = (Z_te - Z_bg.mean()) * pd.Series(model.coef_[0], index=Z_te.columns)       # per transformed column
    # SHAP subsamples the background to 100 rows by default; use all rows so both methods share the same 'average customer'.
    masker = shap.maskers.Independent(Z_bg.to_numpy(), max_samples=len(Z_bg))
    theirs = shap.LinearExplainer(model, masker).shap_values(Z_te.to_numpy())
    np.testing.assert_allclose(ours.to_numpy(), theirs, atol=1e-6)
