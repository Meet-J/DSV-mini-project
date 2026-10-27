"""Interpretation of the final churn model, with a logistic-regression companion.

Views, each answering a different question (all describe the MODEL's associations, not causes of churn):

  1. Permutation importance on held-out data: how much does ranking quality (ROC-AUC) depend on each feature?
  2. Per-customer contributions: WHY did this customer get this probability?
       - tree ensembles: exact tree SHAP values, in percentage points of churn probability
       - logistic regression: coefficient x (value - average value), in log-odds; this equals SHAP exactly
  3. Level effects: the average contribution of each category (and tenure band) across customers, which gives
     the direction and size of each pattern in the model.
  4. Companion logistic regression: odds ratios with bootstrap intervals, as an independent, transparent
     second opinion. Direction agreement between the final model and the companion is reported.

Run from the project root (after `python -m src.train_model`):
    python -m src.interpretation
"""
from __future__ import annotations

import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from src import config
from src.evaluation import ACCENT, MUTED, SLATE, _style
from src.feature_engineering import add_engineered_features, build_pipeline, make_train_test_split
from src.reporting import md_table as _md

# Reference levels for multi-level features (companion odds ratios are reported against these).
REFERENCE_LEVELS = {"Contract": "Month-to-month", "InternetService": "DSL", "PaymentMethod": "Electronic check"}
YES_VS_NO_FEATURES = [
    "Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
]
# Natural units for numeric features (the logistic model itself works in standard deviations).
UNIT_CHANGES = {"tenure": (12, "+12 months"), "MonthlyCharges": (10, "+10 in monthly charges")}
TREE_MODELS = {"RandomForestClassifier", "GradientBoostingClassifier", "DecisionTreeClassifier"}


# ----------------------------------------------------------------------------------------------
# Logistic regression: odds ratios and exact closed-form contributions
# ----------------------------------------------------------------------------------------------
def _fitted_parts(pipeline):
    pre, model = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
    names = list(pre.get_feature_names_out())
    return pre, model, names, pd.Series(model.coef_[0], index=names)


def odds_ratio_table(pipeline) -> pd.DataFrame:
    """One row per interpretable comparison, as an odds ratio (>1 = higher predicted churn odds).

    Multi-level features are compared against a reference level using coefficient DIFFERENCES: one-hot columns
    of a multi-level feature are collinear with the intercept, so a single dummy's coefficient has no meaning
    on its own, but differences between levels do.
    """
    pre, _, names, coef = _fitted_parts(pipeline)
    scaler = pre.named_transformers_["numeric"]
    sd = dict(zip(scaler.feature_names_in_, scaler.scale_))

    rows = []
    for feat, (delta, label) in UNIT_CHANGES.items():
        if feat in sd:                                                    # only numeric features the model actually uses
            rows.append((feat, label, coef[feat] / sd[feat] * delta))     # per natural unit, not per standard deviation
    rows.append(("SeniorCitizen", "Senior citizen vs not", coef["SeniorCitizen"]))
    for feat in YES_VS_NO_FEATURES:
        rows.append((feat, "Yes vs No", coef[f"{feat}_Yes"]))
    for feat, ref in REFERENCE_LEVELS.items():
        levels = [c[len(feat) + 1:] for c in names if c.startswith(feat + "_")]
        for level in levels:
            if level != ref:
                rows.append((feat, f"{level} vs {ref}", coef[f"{feat}_{level}"] - coef[f"{feat}_{ref}"]))
    out = pd.DataFrame(rows, columns=["feature", "comparison", "log_odds"])
    out["odds_ratio"] = np.exp(out["log_odds"])
    return out


def bootstrap_odds_ratios(pipeline, X, y, n_boot: int = 300, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Refit on resampled TRAINING rows to get 95% intervals, so small coefficients are not over-read."""
    rng = np.random.default_rng(seed)
    base = odds_ratio_table(pipeline)
    draws, skipped = [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        try:
            refit = clone(pipeline).fit(X.iloc[idx], y.iloc[idx])
            draws.append(odds_ratio_table(refit)["odds_ratio"].to_numpy())
        except (KeyError, ValueError):                       # a resample that lost a category level
            skipped += 1
    draws = np.vstack(draws)
    base["or_low"], base["or_high"] = np.percentile(draws, [2.5, 97.5], axis=0)
    base.attrs["n_boot"], base.attrs["skipped"] = len(draws), skipped
    return base


def _raw_feature_of(column: str, categorical_raw: list[str]) -> str:
    """Map a transformed column name (e.g. 'Contract_One year') back to its raw feature ('Contract')."""
    matches = [r for r in categorical_raw if column.startswith(r + "_")]
    return max(matches, key=len) if matches else column


def _group_to_raw(per_column: pd.DataFrame, pre) -> pd.DataFrame:
    categorical_raw = list(pre.named_transformers_["categorical"].feature_names_in_)
    return per_column.T.groupby(lambda c: _raw_feature_of(c, categorical_raw)).sum().T


def feature_contributions(pipeline, X, background) -> tuple[pd.DataFrame, float]:
    """Linear model: contribution of each raw feature to each customer's log-odds, relative to the average customer.

    contribution = coefficient x (customer's value - average value in `background`), summed over a feature's
    one-hot columns. Returns (contributions, baseline) with
        baseline + contributions.sum(axis=1) == the model's log-odds (exactly).
    """
    engineer, pre, model = pipeline.named_steps["engineer"], pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
    names = list(pre.get_feature_names_out())
    coef = pd.Series(model.coef_[0], index=names)
    Z = pre.transform(engineer.transform(X))
    mean = pre.transform(engineer.transform(background)).mean()
    baseline = float(model.intercept_[0] + (coef * mean).sum())
    return _group_to_raw((Z - mean) * coef, pre), baseline


# ----------------------------------------------------------------------------------------------
# Tree ensembles: exact tree SHAP
# ----------------------------------------------------------------------------------------------
def tree_shap_contributions(pipeline, X) -> tuple[pd.DataFrame, float]:
    """Exact tree SHAP contributions per raw feature, in units of churn PROBABILITY (0.05 = 5 points).

    Returns (contributions, baseline) with baseline + contributions.sum(axis=1) == predict_proba(X)[:, 1].
    One-hot columns are summed back to their raw feature; sums of SHAP values remain valid attributions.
    """
    try:
        import shap
    except ImportError as exc:                                # shap is a dependency of this module only
        raise ImportError("Tree explanations need the `shap` package: pip install shap") from exc
    engineer, pre, model = pipeline.named_steps["engineer"], pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
    Z = pre.transform(engineer.transform(X))
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(Z.to_numpy())
    if isinstance(values, list):                              # older shap: one array per class
        values, expected = values[1], np.atleast_1d(explainer.expected_value)[1]
    else:
        expected = np.atleast_1d(explainer.expected_value)
        if values.ndim == 3:                                  # newer shap: (rows, features, classes)
            values, expected = values[:, :, 1], expected[1]
        else:
            expected = expected[0]
    return _group_to_raw(pd.DataFrame(values, index=Z.index, columns=Z.columns), pre), float(expected)


# ----------------------------------------------------------------------------------------------
# Shared: importance, drivers, level effects
# ----------------------------------------------------------------------------------------------
def permutation_importance_table(pipeline, X, y, n_repeats: int = 30, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Drop in ROC-AUC when one raw column is shuffled (breaking its link to churn). Run on held-out data.

    Shuffling is done on the RAW columns fed to the pipeline, so a multi-level feature such as Contract is
    shuffled as a whole, not one dummy column at a time.
    """
    res = permutation_importance(pipeline, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=1)
    out = pd.DataFrame({"feature": X.columns, "importance_mean": res.importances_mean, "importance_std": res.importances_std})
    return out.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def _display_value(feature: str, value) -> str:
    """Readable value for labels: SeniorCitizen is stored as 0/1 but shown as No/Yes."""
    return {0: "No", 1: "Yes"}[value] if feature == "SeniorCitizen" else str(value)


def top_drivers(contributions: pd.DataFrame, X: pd.DataFrame, k: int = 3, direction: str = "risk") -> pd.DataFrame:
    """For each customer, the k features pushing hardest toward ('risk') or away from ('protective') churn.

    Only features that actually push in that direction are listed; unused slots are empty strings.
    """
    sign = 1 if direction == "risk" else -1
    values = sign * contributions.to_numpy()
    order = np.argsort(-values, axis=1)[:, :k]
    cols = contributions.columns
    out = {}
    for rank in range(k):
        feat_idx = order[:, rank]
        feats = cols[feat_idx]
        strength = values[np.arange(len(values)), feat_idx]
        out[f"top_{direction}_driver_{rank + 1}"] = [
            f"{f} = {_display_value(f, X.iloc[i][f])}" if s > 0 else "" for i, (f, s) in enumerate(zip(feats, strength))
        ]
    return pd.DataFrame(out, index=contributions.index)


def level_effects(contributions: pd.DataFrame, X: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """Average contribution of each category level (and tenure band) across customers, times `scale`.

    For a tree model with scale=100 the unit is percentage points of churn probability: how much that level
    pushes the average customer's predicted churn probability up or down.
    """
    engineered = add_engineered_features(X)
    rows = []
    for feat in contributions.columns:
        if feat == "tenure":
            groups, name = engineered["TenureGroup"], "tenure (by band, months)"
        elif feat == "SeniorCitizen":
            groups, name = X[feat].map({0: "No", 1: "Yes"}), feat
        elif feat in X.columns and X[feat].nunique() <= 6:
            groups, name = X[feat], feat
        else:
            continue
        for level, idx in groups.groupby(groups).groups.items():
            rows.append({"feature": name, "level": str(level), "customers": len(idx),
                         "mean_effect": float(contributions.loc[idx, feat].mean() * scale)})
    return pd.DataFrame(rows)


def direction_agreement(odds: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    """For each odds-ratio comparison 'A vs B', does the final model's level effect point the same way?

    Agreement means sign(log odds ratio) == sign(effect of A - effect of B). Tenure rows are skipped
    (the two methods express it in different units).
    """
    lookup = effects.set_index(["feature", "level"])["mean_effect"]
    rows = []
    for r in odds.itertuples():
        if r.feature in ("tenure", "MonthlyCharges"):
            continue
        a, b = ("Yes", "No") if r.comparison in ("Yes vs No", "Senior citizen vs not") else r.comparison.split(" vs ")
        key_a, key_b = (r.feature, a), (r.feature, b)
        if key_a in lookup.index and key_b in lookup.index:
            diff = lookup[key_a] - lookup[key_b]
            rows.append({"feature": r.feature, "comparison": r.comparison, "companion_log_odds": r.log_odds,
                         "final_model_effect_diff": diff, "agree": bool(np.sign(r.log_odds) == np.sign(diff))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------------
def plot_odds_ratios(table: pd.DataFrame, path) -> None:
    _style()
    t = table.sort_values("odds_ratio").reset_index(drop=True)
    labels = [f"{r.feature}: {r.comparison}" for r in t.itertuples()]
    colors = [ACCENT if lo > 1 else SLATE if hi < 1 else MUTED for lo, hi in zip(t.or_low, t.or_high)]
    fig, ax = plt.subplots(figsize=(9, 0.34 * len(t) + 1.6))
    y = np.arange(len(t))
    ax.hlines(y, t.or_low, t.or_high, color=colors, lw=2)
    ax.scatter(t.odds_ratio, y, color=colors, s=42, zorder=3)
    ax.axvline(1, color="black", lw=1)
    ax.set_xscale("log"); ax.set_yticks(y, labels)
    ax.set_xticks([0.2, 0.35, 0.5, 1, 2, 3]); ax.set_xticklabels(["0.2", "0.35", "0.5", "1", "2", "3"])
    ax.set_xlabel("Odds ratio for churn (log scale; 95% bootstrap interval)")
    ax.set_title("Companion logistic regression: odds ratios, other features held fixed")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], color=ACCENT, marker="o", label="Higher odds (interval above 1)"),
                       Line2D([], [], color=SLATE, marker="o", label="Lower odds (interval below 1)"),
                       Line2D([], [], color=MUTED, marker="o", label="Interval includes 1")],
              frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_importance(perm: pd.DataFrame, mean_abs: pd.Series, path, unit_label: str, top_n: int = 12) -> None:
    _style()
    top = perm.head(top_n).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].barh(top.feature, top.importance_mean, xerr=top.importance_std, color=SLATE, ecolor="black", capsize=2)
    axes[0].set(xlabel="Drop in test ROC-AUC when the feature is shuffled", title="Permutation importance (held-out data)")
    axes[1].barh(top.feature, mean_abs.reindex(top.feature).values, color="#3F5670")
    axes[1].set(xlabel=f"Mean |contribution| ({unit_label})", title="Average per-customer contribution (SHAP)")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_level_effects(effects: pd.DataFrame, path, unit_label: str,
                       features=("Contract", "tenure (by band, months)", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport")) -> None:
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    lim = effects.loc[effects.feature.isin(features), "mean_effect"].abs().max() * 1.1
    for ax, feat in zip(axes.ravel(), features):
        t = effects[effects.feature == feat].sort_values("mean_effect")
        ax.barh(t.level, t.mean_effect, color=[ACCENT if v > 0 else SLATE for v in t.mean_effect])
        ax.axvline(0, color="black", lw=1)
        ax.set_title(feat, fontsize=11); ax.grid(axis="y", visible=False); ax.set_xlim(-lim, lim)
    fig.supxlabel(f"Average contribution to predicted churn ({unit_label}); right = pushes toward churn", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_local_explanation(contrib_row: pd.Series, x_row: pd.Series, probability: float, baseline: float, path,
                           unit_label: str, scale: float, top_n: int = 8) -> None:
    _style()
    top = contrib_row.reindex(contrib_row.abs().sort_values(ascending=False).index).head(top_n).iloc[::-1] * scale
    labels = [f"{f} = {_display_value(f, x_row[f])}" for f in top.index]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.barh(labels, top.values, color=[ACCENT if v > 0 else SLATE for v in top.values])
    ax.axvline(0, color="black", lw=1)
    ax.set(xlabel="Contribution to churn probability (points); right = toward churn" if scale == 100 else "Contribution to log-odds; right = toward churn",
           title=f"Highest-risk test customer: scores {probability:.0%} (average {baseline:.0%})")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError("Run `python -m src.train_model` first.")
    final = joblib.load(config.MODEL_PATH)
    final_name = type(final.named_steps["model"]).__name__
    is_tree = final_name in TREE_MODELS
    if not is_tree and final_name != "LogisticRegression":
        raise ValueError(f"No explanation method for {final_name}.")

    df = pd.read_csv(config.CLEAN_DATA_PATH)
    X_train, X_test, y_train, y_test = make_train_test_split(df)
    config.MODEL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- final model: per-customer contributions (test customers) and additivity check
    if is_tree:
        contrib, baseline = tree_shap_contributions(final, X_test)
        model_output = final.predict_proba(X_test)[:, 1]
        scale, unit_label, method = 100.0, "percentage points of churn probability", "exact tree SHAP"
    else:
        contrib, baseline = feature_contributions(final, X_test, background=X_train)
        model_output = final.decision_function(X_test)
        scale, unit_label, method = 1.0, "log-odds", "closed-form linear contributions (equal to SHAP)"
    max_gap = float(np.abs(baseline + contrib.sum(axis=1).to_numpy() - model_output).max())
    assert max_gap < 1e-6, f"contributions do not add up to the model output (gap {max_gap})"
    proba = final.predict_proba(X_test)[:, 1]
    baseline_p = baseline if is_tree else float(1 / (1 + np.exp(-baseline)))

    # --- permutation importance (explanation only; nothing in the model is changed because of it)
    perm = permutation_importance_table(final, X_test, y_test)
    perm.to_csv(config.OUTPUT_DIR / "permutation_importance.csv", index=False)
    dropped = perm[perm.feature.isin(["TotalCharges", "gender", "MonthlyCharges"])]
    assert (dropped.importance_mean == 0).all(), "columns the pipeline drops must have exactly zero importance"

    mean_abs = (contrib.abs().mean() * scale).sort_values(ascending=False)
    mean_abs.rename("mean_abs_contribution").to_csv(config.OUTPUT_DIR / "mean_abs_contribution.csv")
    used = perm[perm.feature.isin(mean_abs.index)]
    rho = spearmanr(used.set_index("feature").importance_mean, mean_abs.reindex(used.feature).values)[0]
    effects = level_effects(contrib, X_test, scale)
    effects.to_csv(config.OUTPUT_DIR / "level_effects.csv", index=False)

    top_idx = int(np.argmax(proba))
    plot_importance(used, mean_abs, config.MODEL_FIGURES_DIR / "importance.png", unit_label)
    plot_level_effects(effects, config.MODEL_FIGURES_DIR / "level_effects.png", unit_label)
    plot_local_explanation(contrib.iloc[top_idx], X_test.iloc[top_idx], proba[top_idx], baseline_p,
                           config.MODEL_FIGURES_DIR / "local_explanation_example.png", unit_label, scale)

    # --- companion logistic regression (transparent second opinion), fitted on training data
    print("Bootstrapping companion logistic-regression odds ratios (training data)...", flush=True)
    if final_name == "LogisticRegression":
        companion = final
    else:
        best = pd.read_csv(config.OUTPUT_DIR / "model_comparison.csv").set_index("model").loc["Logistic Regression", "best_params"]
        companion = build_pipeline(LogisticRegression(max_iter=5000, random_state=config.RANDOM_SEED, **json.loads(best))).fit(X_train, y_train)
    orr = bootstrap_odds_ratios(companion, X_train, y_train)
    orr.to_csv(config.OUTPUT_DIR / "odds_ratios.csv", index=False)
    plot_odds_ratios(orr, config.MODEL_FIGURES_DIR / "odds_ratios.png")

    agreement = direction_agreement(orr, effects) if is_tree else pd.DataFrame()
    n_agree = int(agreement.agree.sum()) if len(agreement) else 0

    report = [
        "# Model interpretation report",
        "",
        "Generated by `python -m src.interpretation`. All statements describe the model's *associations* with churn, not causes.",
        "",
        f"Final model: **{final_name}**. Per-customer explanations use {method}; unit: {unit_label}.",
        "",
        "## 1. Permutation importance (held-out test data, 30 shuffles per feature)",
        "",
        _md(perm, {"importance_mean": "{:.4f}", "importance_std": "{:.4f}"}),
        "",
        "`gender`, `TotalCharges` and `MonthlyCharges` are dropped by the pipeline, so their importance is exactly 0 (checked in code). "
        "Permutation importance underestimates features that are strongly correlated with others in the model.",
        "",
        "## 2. Per-customer contributions (test customers)",
        "",
        f"`baseline + sum of contributions` equals the model's output for every test customer (largest gap {max_gap:.1e}, checked in code). "
        f"Baseline (the average customer's predicted {'probability' if is_tree else 'log-odds'}): {baseline_p:.1%}.",
        f"Rank agreement between permutation importance and mean absolute contribution (Spearman, features in the model): {rho:.2f}.",
        "",
        _md(mean_abs.rename("mean_abs_contribution").reset_index().rename(columns={"index": "feature"}), {"mean_abs_contribution": "{:.2f}"}),
        "",
        f"## 3. Level effects ({unit_label}; positive = pushes toward churn)",
        "",
        "Average contribution of each category (or tenure band) across the test customers.",
        "",
        _md(effects.sort_values(["feature", "mean_effect"], ascending=[True, False]), {"mean_effect": "{:+.2f}"}),
        "",
        "## 4. Companion logistic regression (odds ratios, training data, 95% bootstrap intervals)",
        "",
        f"Odds ratio above 1 = higher predicted odds of churn, other features held fixed. {orr.attrs['n_boot']} refits on resampled "
        f"training rows ({orr.attrs['skipped']} skipped). Multi-level features are compared with a reference level using coefficient differences.",
        "",
        _md(orr.sort_values("odds_ratio", ascending=False)[["feature", "comparison", "odds_ratio", "or_low", "or_high"]],
            {"odds_ratio": "{:.2f}", "or_low": "{:.2f}", "or_high": "{:.2f}"}),
        "",
    ]
    if is_tree:
        report += [
            "### Direction agreement between the final model and the companion",
            "",
            f"{n_agree} of {len(agreement)} comparisons point the same way (sign of the companion's log odds ratio vs sign of the final model's "
            "level-effect difference).",
            "",
            _md(agreement[~agreement.agree][["feature", "comparison", "companion_log_odds", "final_model_effect_diff"]],
                {"companion_log_odds": "{:+.2f}", "final_model_effect_diff": "{:+.2f}"}) if (~agreement.agree).any() else "All comparisons agree.",
            "",
        ]
    report += [
        "## 5. Caveats",
        "- Effects are conditional on the other features. A feature can look protective or harmful here while showing the opposite in a "
        "one-variable EDA comparison, because features overlap.",
        "- Since the monthly bill is determined by the services (see the feature-selection section of the notebook), each service column "
        "now carries both 'has the service' and 'pays for it'; this dataset cannot separate the two.",
        "- The model treats 'No internet service' and 'No phone service' as 'No' for each add-on, so those levels share one input value; "
        "their different average effects in the level table reflect interactions with other features (mainly InternetService).",
        "- Importance and contributions use the held-out test set for explanation only; nothing was changed because of them.",
        "- A statistical association in one snapshot is not evidence that changing a feature would change churn.",
        "",
    ]
    (config.OUTPUT_DIR / "model_interpretation.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Done in {time.time() - t0:.0f}s. {final_name}; additivity gap {max_gap:.1e}; Spearman {rho:.2f}; "
          f"direction agreement {n_agree}/{len(agreement)}", flush=True)


if __name__ == "__main__":
    main()
