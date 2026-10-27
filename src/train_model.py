"""Train, tune, compare and select the churn model.

The order of operations is the point of this script. Nothing is decided by looking at the test set:

  1. Tune every model with cross-validation on the TRAINING split only.
  2. Pick the final model with a rule declared in advance (one-standard-error rule, see evaluation.py).
  3. Pick the decision threshold from out-of-fold training predictions.
  4. Check the effect of class weighting, again on training data only.
  ---- the test set is first touched below this line, once, for every model ----
  5. Report test metrics for all four models, with bootstrap intervals.
  6. Save the model, metadata, tables, figures and a generated report.

Run from the project root:
    python -m src.train_model
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.evaluation import (
    best_f1_threshold, bootstrap_ci, calibration_table, classification_metrics, paired_bootstrap_diff,
    plot_calibration, plot_confusion_grid, plot_final_confusions, plot_roc_pr, plot_threshold_tradeoff,
    select_by_one_standard_error, threshold_table,
)
from src.feature_engineering import build_pipeline, make_train_test_split
from src.reporting import md_table

SEED = config.RANDOM_SEED
N_JOBS = 1                 # the reference machine has one CPU; results do not depend on this setting
SIMPLICITY_ORDER = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]


@dataclass
class ModelSpec:
    estimator: BaseEstimator
    scale_numeric: bool                     # scaling matters for logistic regression only
    param_space: dict = field(default_factory=dict)
    n_iter: int | None = None               # None = exhaustive grid search, int = randomised search


MODEL_SPECS: dict[str, ModelSpec] = {
    "Logistic Regression": ModelSpec(
        LogisticRegression(max_iter=2000, random_state=SEED), True,
        {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]},     # extended after the first run put the optimum on the old edge (C=10)
    ),
    "Decision Tree": ModelSpec(
        DecisionTreeClassifier(random_state=SEED), False,
        {"model__max_depth": [3, 4, 5, 6, 8, 10], "model__min_samples_leaf": [10, 25, 50, 100]},
    ),
    "Random Forest": ModelSpec(
        RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1), False,
        {"model__max_depth": [4, 6, 8, 12, None], "model__min_samples_leaf": [1, 3, 5, 10, 20],
         "model__max_features": ["sqrt", 0.3, 0.5]},
        n_iter=15,
    ),
    "Gradient Boosting": ModelSpec(
        GradientBoostingClassifier(random_state=SEED), False,
        {"model__n_estimators": [100, 200, 300], "model__learning_rate": [0.03, 0.05, 0.1],
         "model__max_depth": [2, 3, 4], "model__subsample": [0.7, 0.85, 1.0], "model__min_samples_leaf": [1, 10, 30]},
        n_iter=15,
    ),
}


# ----------------------------------------------------------------------------------------------
# Steps 1-2: tuning and selection (training data only)
# ----------------------------------------------------------------------------------------------
def tune_models(X_train, y_train, cv) -> dict:
    """Cross-validated hyperparameter search for every model, scored by ROC-AUC."""
    searches = {}
    for name, spec in MODEL_SPECS.items():
        start = time.time()
        pipeline = build_pipeline(spec.estimator, scale_numeric=spec.scale_numeric)
        if spec.n_iter is None:
            search = GridSearchCV(pipeline, spec.param_space, scoring="roc_auc", cv=cv, n_jobs=N_JOBS, refit=True)
        else:
            search = RandomizedSearchCV(pipeline, spec.param_space, n_iter=spec.n_iter, scoring="roc_auc",
                                        cv=cv, n_jobs=N_JOBS, refit=True, random_state=SEED)
        search.fit(X_train, y_train)
        searches[name] = search
        print(f"  tuned {name:20s} CV AUC {search.best_score_:.4f}  ({time.time() - start:.0f}s)", flush=True)
    return searches


def summarize_cv(searches: dict) -> pd.DataFrame:
    rows = []
    for name, s in searches.items():
        i = s.best_index_
        std = s.cv_results_["std_test_score"][i]
        rows.append({"model": name, "search": "grid" if MODEL_SPECS[name].n_iter is None else f"random ({MODEL_SPECS[name].n_iter} candidates)",
                     "candidates_tried": len(s.cv_results_["params"]), "cv_mean": s.best_score_, "cv_std": std,
                     "cv_se": std / np.sqrt(config.CV_FOLDS), "best_params": {k.replace("model__", ""): v for k, v in s.best_params_.items()}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Step 4: does class weighting help? (training data only)
# ----------------------------------------------------------------------------------------------
def imbalance_check(searches: dict, X_train, y_train, cv) -> pd.DataFrame:
    """Compare no weighting vs class_weight='balanced' for the two models that support it."""
    rows = []
    for name in ["Logistic Regression", "Random Forest"]:
        for weight in [None, "balanced"]:
            est = clone(searches[name].best_estimator_).set_params(model__class_weight=weight)
            oof = cross_val_predict(est, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
            m = classification_metrics(y_train, oof, 0.5)
            rows.append({"model": name, "class_weight": str(weight), "roc_auc": m["roc_auc"], "brier": m["brier"],
                         "precision_at_0.5": m["precision"], "recall_at_0.5": m["recall"],
                         "customers_flagged_pct": m["flagged_share"] * 100,
                         "mean_predicted_prob": float(oof.mean())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------------------------------
def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def build_report(ctx: dict) -> str:
    cvs, cmp_, imb = ctx["cv_summary"], ctx["comparison"], ctx["imbalance"]
    final, thr = ctx["selected"], ctx["threshold"]
    fm_default, fm_thr = ctx["final_at_default"], ctx["final_at_threshold"]
    cal = ctx["calibration_test"]
    lines = [
        "# Model training and selection report",
        "",
        "Generated by `python -m src.train_model`. Every number below was computed by that run.",
        "",
        "## 1. Setup",
        f"- Stratified split (seed {SEED}): {ctx['n_train']:,} training rows, {ctx['n_test']:,} test rows; churn rate "
        f"{ctx['train_rate']:.2%} in training, {ctx['test_rate']:.2%} in test.",
        f"- Tuning and selection: {config.CV_FOLDS}-fold stratified CV on the **training split only**, scored by ROC-AUC.",
        "- All models use the feature set from `src/feature_engineering.py` (defaults). No class weights are used in the compared models (see section 4).",
        "- The test split was first used after the model, hyperparameters and threshold were fixed.",
        "",
        "## 2. Tuning and selection (training data only)",
        "",
        md_table(cvs.assign(best_params=cvs["best_params"].map(lambda d: ", ".join(f"{k}={v}" for k, v in d.items())))
                 [["model", "search", "candidates_tried", "cv_mean", "cv_se", "best_params"]],
                 {"cv_mean": "{:.4f}", "cv_se": "{:.4f}"}),
        "",
        "**Selection rule (declared before running):** among models whose mean CV ROC-AUC is within one standard error "
        "(fold std / sqrt(folds)) of the best, choose the simplest, in the order Logistic Regression, Decision Tree, "
        "Random Forest, Gradient Boosting. Selecting the best CV mean after tuning is slightly optimistic for every model alike; "
        "the test set below provides the unbiased estimate.",
        "",
        f"**Outcome:** {ctx['rationale']}",
        "",
        "## 3. Test-set comparison (threshold 0.5 for every model)",
        "",
        md_table(cmp_[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "roc_auc_ci", "pr_auc", "brier", "cv_roc_auc"]],
                 {"accuracy": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}", "roc_auc": "{:.4f}",
                  "pr_auc": "{:.4f}", "brier": "{:.4f}", "cv_roc_auc": "{:.4f}"}),
        "",
        "`roc_auc_ci` is a 95% percentile bootstrap interval (1000 resamples of the test customers). Brier score: lower is better; "
        f"predicting the base rate for everyone scores {ctx['brier_baseline']:.4f}. Confusion-matrix counts are in `output/model_comparison.csv`.",
        "",
        md_table(cmp_[["model", "tn", "fp", "fn", "tp"]]),
        "",
        f"**Paired comparison against the selected model ({final}).** Post-selection diagnostic only; it was not used to choose the model. "
        "Each cell is `other model minus selected model` with a 95% paired-bootstrap interval (same resampled customers for both). "
        "For ROC-AUC and PR-AUC positive means the other model is better; for Brier score negative means the other model is better.",
        "",
        md_table(ctx["paired"]),
        "",
        "## 4. Class imbalance check (training data, out-of-fold)",
        "",
        md_table(imb, {"roc_auc": "{:.4f}", "brier": "{:.4f}", "precision_at_0.5": "{:.3f}", "recall_at_0.5": "{:.3f}",
                       "customers_flagged_pct": "{:.1f}", "mean_predicted_prob": "{:.3f}"}),
        "",
        f"The observed churn rate in training is {ctx['train_rate']:.3f}. `mean_predicted_prob` shows whether predicted probabilities stay "
        "anchored to that rate. The compared models use no class weights.",
        "",
        f"## 5. Final model: {final}",
        "",
        f"- Selected hyperparameters: {', '.join(f'{k}={v}' for k, v in ctx['best_params'].items())}",
        f"- **Operating threshold: {thr:.2f}**, the threshold that maximises F1 on out-of-fold *training* predictions "
        "(equal weight on precision and recall; it needs no cost assumption). It is a convention, not a business-optimal value; see section 7.",
        "",
        md_table(pd.DataFrame([{**{"threshold": "0.50 (default)"}, **{k: fm_default[k] for k in ["accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]}},
                               {**{"threshold": f"{thr:.2f} (chosen)"}, **{k: fm_thr[k] for k in ["accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]}}]),
                 {"accuracy": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}"}),
        "",
        f"Test-set 95% bootstrap intervals at the chosen threshold: precision {ctx['ci_precision'][0]:.3f} to {ctx['ci_precision'][1]:.3f}, "
        f"recall {ctx['ci_recall'][0]:.3f} to {ctx['ci_recall'][1]:.3f}.",
        "",
        "## 6. Calibration of the final model",
        "",
        f"Brier score on the test set: {ctx['final_brier']:.4f}. Mean predicted probability {ctx['mean_pred_test']:.3f} vs observed churn rate {ctx['test_rate']:.3f}.",
        "",
        md_table(cal, {"customers": "{:.0f}", "mean_predicted": "{:.3f}", "observed_rate": "{:.3f}"}),
        "",
        "## 7. Threshold sensitivity (out-of-fold training predictions)",
        "",
        md_table(ctx["threshold_sensitivity"], {"threshold": "{:.2f}", "customers_flagged_pct": "{:.1f}", "precision": "{:.3f}",
                                                "recall": "{:.3f}", "f1": "{:.3f}", "f2": "{:.3f}",
                                                "churners_caught": "{:.0f}", "false_alarms": "{:.0f}"}),
        "",
        "**How to read this against costs.** Contacting a customer with churn probability `p` is worth it when `p x s x V > c`, where `V` is the value "
        "of a retained customer, `s` the chance an intervention succeeds, and `c` the cost of the intervention. So the break-even threshold is `c / (s x V)`, "
        "which is valid only if probabilities are well calibrated (section 6). **The dataset contains no values for `c`, `s` or `V`**, so no business-optimal "
        "threshold is claimed here: a company supplies the ratio, and the table shows what each threshold implies for workload, precision and recall.",
        "",
        "## 8. Caveats",
        "- Single stratified split: the test set has about "
        f"{int(round(ctx['test_rate'] * ctx['n_test']))} churners, so metric differences between models of a few thousandths are not distinguishable (see the bootstrap interval).",
        "- Hyperparameter search was small (grids or 15 random candidates) to keep the run reproducible on a single CPU.",
        "- Figures are in `images/model/`; artefacts in `models/`.",
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    df = pd.read_csv(config.CLEAN_DATA_PATH)
    X_train, X_test, y_train, y_test = make_train_test_split(df)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=SEED)
    print(f"train {X_train.shape}, test {X_test.shape}; churn rate {y_train.mean():.4f} / {y_test.mean():.4f}")

    # --- 1-2. tune and select, training data only
    print("Tuning (training split, cross-validated):", flush=True)
    searches = tune_models(X_train, y_train, cv)
    cv_summary = summarize_cv(searches)
    selected, rationale = select_by_one_standard_error(cv_summary, SIMPLICITY_ORDER)
    final = searches[selected].best_estimator_
    print(f"Selected: {selected}\n  {rationale}", flush=True)

    # --- 3. threshold from out-of-fold training predictions
    oof = cross_val_predict(clone(final), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = best_f1_threshold(y_train, oof)
    print(f"Operating threshold (max F1 on OOF training predictions): {threshold:.2f}", flush=True)

    # --- 4. class-imbalance check, training data only
    imbalance = imbalance_check(searches, X_train, y_train, cv)

    # ================= the test set is touched from here on =================
    probas = {name: s.best_estimator_.predict_proba(X_test)[:, 1] for name, s in searches.items()}
    rows = []
    for name, p in probas.items():
        m = classification_metrics(y_test, p, 0.5)
        lo, hi = bootstrap_ci(y_test, p, roc_auc_score, seed=SEED)
        cvrow = cv_summary.loc[cv_summary.model == name].iloc[0]
        rows.append({"model": name, **{k: m[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "tn", "fp", "fn", "tp"]},
                     "roc_auc_ci": f"{lo:.3f} to {hi:.3f}", "roc_auc_ci_low": lo, "roc_auc_ci_high": hi,
                     "cv_roc_auc": cvrow.cv_mean, "cv_se": cvrow.cv_se,
                     "best_params": json.dumps(cvrow.best_params, default=_jsonable)})
    comparison = pd.DataFrame(rows)

    p_final = probas[selected]
    final_default = classification_metrics(y_test, p_final, 0.5)
    final_thr = classification_metrics(y_test, p_final, threshold)
    ci_prec = bootstrap_ci(y_test, p_final, lambda y, p: precision_score(y, (p >= threshold).astype(int), zero_division=0), seed=SEED)
    ci_rec = bootstrap_ci(y_test, p_final, lambda y, p: recall_score(y, (p >= threshold).astype(int), zero_division=0), seed=SEED)

    # Post-selection diagnostic (NOT used to choose the model): is each other model's test-set difference from the
    # selected model distinguishable from sampling noise? Same resampled customers for both models.
    paired_rows = []
    for name, p in probas.items():
        if name == selected:
            continue
        d_auc = paired_bootstrap_diff(y_test, p, p_final, roc_auc_score, seed=SEED)
        d_ap = paired_bootstrap_diff(y_test, p, p_final, average_precision_score, seed=SEED)
        d_brier = paired_bootstrap_diff(y_test, p, p_final, brier_score_loss, seed=SEED)
        fmt = lambda t: f"{t[0]:+.4f} ({t[1]:+.4f} to {t[2]:+.4f})"
        paired_rows.append({"model": name, "delta_roc_auc": fmt(d_auc), "delta_pr_auc": fmt(d_ap), "delta_brier": fmt(d_brier)})
    paired = pd.DataFrame(paired_rows)

    # --- artefacts
    config.MODELS_DIR.mkdir(exist_ok=True); config.OUTPUT_DIR.mkdir(exist_ok=True); config.MODEL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, config.MODEL_PATH)
    feature_names = list(final.named_steps["preprocess"].get_feature_names_out())
    metadata = {
        "selected_model": selected, "best_params": cv_summary.loc[cv_summary.model == selected].iloc[0].best_params,
        "operating_threshold": threshold,
        "threshold_rule": "maximises F1 on out-of-fold training predictions; a convention, not a business-optimal value",
        "selection_rule": "one-standard-error rule on CV ROC-AUC, simplest model first",
        "random_seed": SEED, "cv_folds": config.CV_FOLDS, "test_size": config.TEST_SIZE,
        "n_train": len(X_train), "n_test": len(X_test),
        "scikit_learn_version": sklearn.__version__, "model_input_columns": feature_names,
        "cv_summary": cv_summary.drop(columns=["best_params"]).to_dict(orient="records"),
        "note": "Pipeline takes the cleaned DataFrame (customerID as index, no Churn column). Load with joblib from the project root so the `src` package is importable.",
    }
    config.MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=_jsonable), encoding="utf-8")
    comparison.drop(columns=["roc_auc_ci"]).to_csv(config.OUTPUT_DIR / "model_comparison.csv", index=False)
    imbalance.to_csv(config.OUTPUT_DIR / "imbalance_check.csv", index=False)

    sens = threshold_table(y_train, oof)
    plot_roc_pr(y_test, probas, selected, config.MODEL_FIGURES_DIR / "roc_pr_curves.png")
    plot_confusion_grid(y_test, probas, config.MODEL_FIGURES_DIR / "confusion_matrices_all_models.png")
    plot_final_confusions(y_test, p_final, [0.5, threshold], selected, config.MODEL_FIGURES_DIR / "final_model_confusion.png")
    plot_threshold_tradeoff(y_train, oof, threshold, config.MODEL_FIGURES_DIR / "threshold_tradeoff.png",
                            f"{selected}: precision/recall trade-off (out-of-fold, training data)")
    plot_calibration({"Test set": (y_test.values, p_final), "Out-of-fold (training)": (y_train.values, oof)},
                     config.MODEL_FIGURES_DIR / "calibration.png", f"{selected}: calibration")

    ctx = {
        "cv_summary": cv_summary, "comparison": comparison, "imbalance": imbalance, "selected": selected, "rationale": rationale,
        "threshold": threshold, "final_at_default": final_default, "final_at_threshold": final_thr,
        "ci_precision": ci_prec, "ci_recall": ci_rec, "calibration_test": calibration_table(y_test, p_final),
        "final_brier": final_default["brier"], "mean_pred_test": float(p_final.mean()),
        "threshold_sensitivity": sens, "best_params": metadata["best_params"], "paired": paired, "selected_name": selected,
        "n_train": len(X_train), "n_test": len(X_test), "train_rate": float(y_train.mean()), "test_rate": float(y_test.mean()),
        "brier_baseline": float(y_test.mean() * (1 - y_test.mean())),
    }
    (config.OUTPUT_DIR / "model_report.md").write_text(build_report(ctx), encoding="utf-8")
    print(f"Done in {time.time() - t0:.0f}s. Report: output/model_report.md", flush=True)


if __name__ == "__main__":
    main()
