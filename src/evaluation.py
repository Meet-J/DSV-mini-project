"""Metrics, threshold analysis, calibration, bootstrap intervals and plots for the churn models.

Everything here takes plain arrays (true labels, predicted churn probabilities), so it works for
any model and can be unit-tested on tiny examples.
"""
from __future__ import annotations

from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    fbeta_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
    f1_score,
)

# Same restrained palette as the EDA notebook.
SLATE, ACCENT, MUTED = "#6B7F99", "#C0504D", "#B8C2CF"
MODEL_COLORS = {"Logistic Regression": "#C0504D", "Decision Tree": "#B8C2CF",
                "Random Forest": "#8FA3B8", "Gradient Boosting": "#3F5670"}


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """All headline metrics at one decision threshold. ROC-AUC, PR-AUC and Brier use the raw probabilities."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "flagged_share": float(pred.mean()),
    }


def threshold_table(y_true, proba, thresholds: Sequence[float] | None = None) -> pd.DataFrame:
    """Precision/recall/F1/F2 and workload at each threshold: the retention trade-off in one table."""
    thresholds = np.round(np.arange(0.05, 0.801, 0.05), 2) if thresholds is None else thresholds   # above ~0.8 almost no customers are flagged
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "customers_flagged_pct": pred.mean() * 100,
            "precision": precision_score(y_true, pred, zero_division=0) if pred.sum() else np.nan,   # undefined if nobody is flagged
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
            "churners_caught": int(((pred == 1) & (y_true == 1)).sum()),
            "false_alarms": int(((pred == 1) & (y_true == 0)).sum()),
        })
    return pd.DataFrame(rows)


def best_f1_threshold(y_true, proba, grid: Sequence[float] | None = None) -> float:
    """Threshold that maximises F1 (equal weight on precision and recall) over a fine grid."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else grid
    scores = [f1_score(y_true, (np.asarray(proba) >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def bootstrap_ci(y_true, proba, metric: Callable, n_boot: int = 1000, seed: int = 42, level: float = 0.95):
    """Percentile bootstrap interval: resample customers with replacement and recompute the metric."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    n, stats = len(y_true), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():      # both classes needed
            continue
        stats.append(metric(y_true[idx], proba[idx]))
    lo, hi = np.percentile(stats, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; behaves sensibly for small n and rates near 0 or 1."""
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))      # clip: rounding can push a bound a hair outside [0, 1]


def paired_bootstrap_diff(y_true, proba_a, proba_b, metric: Callable, n_boot: int = 1000, seed: int = 42, level: float = 0.95):
    """Bootstrap the difference metric(a) - metric(b) using the SAME resampled customers for both models.

    Pairing removes the shared sampling noise, so it is far more sensitive than comparing two separate
    intervals. Returns (mean difference, low, high).
    """
    rng = np.random.default_rng(seed)
    y_true, a, b = np.asarray(y_true), np.asarray(proba_a), np.asarray(proba_b)
    n, diffs = len(y_true), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        diffs.append(metric(y_true[idx], a[idx]) - metric(y_true[idx], b[idx]))
    lo, hi = np.percentile(diffs, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(np.mean(diffs)), float(lo), float(hi)


def calibration_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability vs observed churn rate in equal-sized probability bins."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    out = df.groupby("bin", observed=True).agg(customers=("y", "size"), mean_predicted=("p", "mean"),
                                               observed_rate=("y", "mean")).reset_index(drop=True)
    return out


def select_by_one_standard_error(results: pd.DataFrame, simplicity_order: Sequence[str]) -> tuple[str, str]:
    """One-standard-error rule: among models whose CV score is within one SE of the best, pick the simplest.

    `results` needs columns: model, cv_mean, cv_se. `simplicity_order` lists models simplest first.
    Returns (chosen model, plain-language rationale).
    """
    best = results.loc[results["cv_mean"].idxmax()]
    floor = best["cv_mean"] - best["cv_se"]
    eligible = results[results["cv_mean"] >= floor]["model"].tolist()
    chosen = next(m for m in simplicity_order if m in eligible)
    rationale = (f"Best CV ROC-AUC: {best['model']} ({best['cv_mean']:.4f}, SE {best['cv_se']:.4f}). Models within one SE "
                 f"(>= {floor:.4f}): {', '.join(eligible)}. Simplest of those: {chosen}.")
    return chosen, rationale


# ----------------------------------------------------------------------------------------------
# Plots (each saves to disk and returns the figure)
# ----------------------------------------------------------------------------------------------
def _style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False})


def plot_roc_pr(y_true, probas: dict[str, np.ndarray], selected: str, path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    prevalence = float(np.mean(y_true))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        prec, rec, _ = precision_recall_curve(y_true, p)
        lw = 2.8 if name == selected else 1.6
        axes[0].plot(fpr, tpr, color=MODEL_COLORS[name], lw=lw, label=f"{name} (AUC {roc_auc_score(y_true, p):.3f})")
        axes[1].plot(rec, prec, color=MODEL_COLORS[name], lw=lw, label=f"{name} (AP {average_precision_score(y_true, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    axes[1].axhline(prevalence, color="black", ls="--", lw=1, label=f"Chance (churn rate {prevalence:.1%})")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curves (test set)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test set)", ylim=(0, 1.02))
    for ax in axes:
        ax.legend(frameon=False, loc="lower right" if ax is axes[0] else "upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _cm_heatmap(ax, y_true, proba, threshold, title):
    pred = (np.asarray(proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    labels = np.array([[f"{cm[i, j]:,}\n({pct[i, j]:.0f}% of actual)" for j in range(2)] for i in range(2)])
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, linewidths=1,
                xticklabels=["Predicted stay", "Predicted churn"], yticklabels=["Actually stayed", "Actually churned"], ax=ax)
    ax.set_title(title)


def plot_confusion_grid(y_true, probas: dict[str, np.ndarray], path, threshold: float = 0.5) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, p) in zip(axes.ravel(), probas.items()):
        _cm_heatmap(ax, y_true, p, threshold, f"{name} (threshold {threshold:.2f})")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_final_confusions(y_true, proba, thresholds: Sequence[float], name: str, path) -> None:
    _style()
    fig, axes = plt.subplots(1, len(thresholds), figsize=(5.2 * len(thresholds), 4.4))
    for ax, t in zip(np.atleast_1d(axes), thresholds):
        _cm_heatmap(ax, y_true, proba, t, f"{name}, threshold {t:.2f}")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_threshold_tradeoff(y_true, proba, chosen: float, path, title: str) -> None:
    _style()
    grid = np.round(np.arange(0.05, 0.801, 0.01), 2)
    tbl = threshold_table(y_true, proba, grid)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(tbl.threshold, tbl.precision, color=SLATE, lw=2, label="Precision")
    ax.plot(tbl.threshold, tbl.recall, color=ACCENT, lw=2, label="Recall")
    ax.plot(tbl.threshold, tbl.f1, color="black", lw=1.6, label="F1")
    ax.plot(tbl.threshold, tbl.customers_flagged_pct / 100, color=MUTED, lw=2, ls="--", label="Share of customers flagged")
    ax.axvline(chosen, color="black", ls=":", lw=1)
    ax.text(chosen + 0.01, 0.04, f"chosen {chosen:.2f}", fontsize=9)
    ax.set(xlabel="Decision threshold on churn probability", ylabel="Value", ylim=(0, 1.02), title=title)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_calibration(series: dict[str, tuple[np.ndarray, np.ndarray]], path, title: str) -> None:
    """series: label -> (y_true, proba). Reliability curve: points near the diagonal = trustworthy probabilities."""
    _style()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for (label, (y, p)), color in zip(series.items(), [ACCENT, SLATE]):
        t = calibration_table(y, p)
        ax.plot(t.mean_predicted, t.observed_rate, "o-", color=color, label=label)
    ax.set(xlabel="Mean predicted churn probability", ylabel="Observed churn rate", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
