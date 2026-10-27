"""Does each engineered feature (or each proposed removal) actually change model performance?

Method: repeated stratified cross-validation on the TRAINING split only (the hold-out test set is
never touched). Every configuration is scored on the same folds, so differences are paired:
for each fold we subtract the reference configuration's ROC-AUC from this one's.

Caveat on the uncertainty: folds from repeated CV overlap in their training data, so the standard
error below is optimistic. Treat differences smaller than about 0.002 AUC as noise.

Run from the project root:
    python -m src.feature_ablation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from src import config
from src.feature_engineering import (
    DEFAULT_DROP, DEFAULT_ENGINEERED, ENGINEERED_FEATURES, LOW_SIGNAL_FEATURES, build_pipeline, make_train_test_split,
)

N_REPEATS = 3

A, E = "A base features only", "E A + all three engineered"
I = "I A - TotalCharges - gender"
PRIOR_DROP = ("TotalCharges", "gender")      # the feature set adopted before MonthlyCharges was found redundant (pinned: never use DEFAULT_DROP here for old configs)


def cfg(engineered=ENGINEERED_FEATURES, drop=()):
    """Always pass `engineered` and `drop` explicitly so changing the pipeline defaults can never alter an experiment."""
    return dict(engineered=tuple(engineered), drop=tuple(drop))


# name -> (preprocessor kwargs, reference config the paired difference is measured against)
# "base" = the cleaned columns with redundant 'No phone/internet service' levels collapsed; that
# collapse is applied in every configuration and is not itself ablated.
CONFIGS = {
    A: (cfg(engineered=()), None),
    "B A + TenureGroup": (cfg(engineered=("TenureGroup",)), A),
    "C A + ProtectiveAddOnCount": (cfg(engineered=("ProtectiveAddOnCount",)), A),
    "D A + AutoPay": (cfg(engineered=("AutoPay",)), A),
    E: (cfg(), A),
    "F A - five low-signal columns": (cfg(engineered=(), drop=LOW_SIGNAL_FEATURES), A),
    "G A - gender": (cfg(engineered=(), drop=("gender",)), A),
    "H A - TotalCharges": (cfg(engineered=(), drop=("TotalCharges",)), A),
    I: (cfg(engineered=(), drop=PRIOR_DROP), A),
    "J E - TotalCharges": (cfg(drop=("TotalCharges",)), E),
    "K E - TotalCharges - gender": (cfg(drop=PRIOR_DROP), E),
    # Adopted set: I minus MonthlyCharges. Paired against I, so the delta isolates the effect of removing MonthlyCharges.
    "L A - TotalCharges - gender - MonthlyCharges (adopted)": (cfg(engineered=DEFAULT_ENGINEERED, drop=DEFAULT_DROP), I),
}

# (model factory, scale numeric?)  Scaling matters for logistic regression only.
MODELS = {
    "Logistic Regression": (lambda: LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED), True),
    "Gradient Boosting": (lambda: GradientBoostingClassifier(random_state=config.RANDOM_SEED), False),
}


def run_ablation() -> pd.DataFrame:
    df = pd.read_csv(config.CLEAN_DATA_PATH)
    X_train, _, y_train, _ = make_train_test_split(df)      # test split deliberately discarded
    cv = RepeatedStratifiedKFold(n_splits=config.CV_FOLDS, n_repeats=N_REPEATS, random_state=config.RANDOM_SEED)

    rows = []
    for model_name, (make_model, scale) in MODELS.items():
        fold_scores = {}
        for cfg_name, (kwargs, _) in CONFIGS.items():
            pipe = build_pipeline(make_model(), scale_numeric=scale, **kwargs)
            fold_scores[cfg_name] = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        for cfg_name, (_, ref) in CONFIGS.items():
            scores = fold_scores[cfg_name]
            row = {"model": model_name, "configuration": cfg_name,
                   "mean_auc": scores.mean(), "std_auc": scores.std(ddof=1)}
            if ref is not None:
                diff = scores - fold_scores[ref]
                row.update(vs=ref.split(" ")[0], delta_auc=diff.mean(), delta_se=diff.std(ddof=1) / np.sqrt(len(diff)))
            rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(results: pd.DataFrame) -> str:
    lines = [
        "# Feature ablation (cross-validated ROC-AUC, training split only)",
        "",
        f"{config.CV_FOLDS}-fold stratified CV repeated {N_REPEATS} times, seed {config.RANDOM_SEED}. "
        "`delta` is the paired mean AUC difference against the reference configuration (`vs`); "
        "`se` is its standard error (optimistic, see module docstring). Differences under about 0.002 are noise.",
        "",
    ]
    for model_name, grp in results.groupby("model", sort=False):
        lines += [f"## {model_name}", "", "| Configuration | Mean AUC | Std | vs | Delta AUC | SE |", "|---|---|---|---|---|---|"]
        for _, r in grp.iterrows():
            delta = "" if pd.isna(r.get("delta_auc")) else f"{r['delta_auc']:+.4f}"
            se = "" if pd.isna(r.get("delta_se")) else f"{r['delta_se']:.4f}"
            vs = "" if pd.isna(r.get("vs")) else r["vs"]
            lines.append(f"| {r['configuration']} | {r['mean_auc']:.4f} | {r['std_auc']:.4f} | {vs} | {delta} | {se} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    results = run_ablation()
    out_path = config.PROJECT_ROOT / "output" / "feature_ablation.md"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(to_markdown(results) + "\n", encoding="utf-8")
    results.to_csv(out_path.with_suffix(".csv"), index=False)      # machine-readable copy; the notebook reads this
    print(to_markdown(results))
    print(f"Saved to {out_path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
