"""Subject-level aggregation and predictive metrics (Sections 2.7 and 2.15)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, roc_auc_score, roc_curve)


def aggregate(pred: pd.DataFrame, task_balanced: bool = False, prob_col: str = "prob") -> pd.DataFrame:
    """Segments -> recordings -> subjects (Section 2.7).

    ``pred`` needs subject_id, recording_id, label, task_family and ``prob_col``.
    With ``task_balanced`` the recordings are first averaged within each task
    family and the family means are averaged with equal weight.
    """
    keys = [k for k in ("repeat", "fold") if k in pred.columns]
    rec = (pred.groupby(keys + ["subject_id", "recording_id", "task_family", "label"], as_index=False)
           [prob_col].mean())
    if task_balanced:
        fam = rec.groupby(keys + ["subject_id", "task_family", "label"], as_index=False)[prob_col].mean()
        return fam.groupby(keys + ["subject_id", "label"], as_index=False)[prob_col].mean()
    return rec.groupby(keys + ["subject_id", "label"], as_index=False)[prob_col].mean()


def calibration(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    """Calibration slope (logistic regression of y on logit p) and intercept (logit p as offset)."""
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    slope = _logistic_fit(np.column_stack([np.ones_like(lp), lp]), y)[1]
    intercept = _logistic_fit(np.ones((len(y), 1)), y, offset=lp)[0]
    return float(intercept), float(slope)


def _logistic_fit(X, y, offset=None, n_iter: int = 50) -> np.ndarray:
    """Unpenalized logistic regression by Newton-Raphson (tiny problems only)."""
    beta = np.zeros(X.shape[1])
    off = np.zeros(len(y)) if offset is None else offset
    for _ in range(n_iter):
        mu = 1 / (1 + np.exp(-np.clip(X @ beta + off, -30, 30)))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        step = np.linalg.solve(X.T @ (X * w[:, None]) + 1e-9 * np.eye(X.shape[1]), X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= threshold).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    both = len(np.unique(y)) == 2
    out = {
        "auroc": roc_auc_score(y, p) if both else np.nan,
        "pr_auc": average_precision_score(y, p) if both else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y, yhat) if both else np.nan,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y, yhat, zero_division=0),
        "brier": brier_score_loss(y, p),
        "n_subjects": len(y),
    }
    if both:
        out["cal_intercept"], out["cal_slope"] = calibration(y, p)
    return out


def repeated_cv_summary(subj: pd.DataFrame, threshold: float = 0.5, prob_col: str = "prob") -> pd.DataFrame:
    """Metrics per repetition (pooled out-of-fold subjects) and their mean and SD across repetitions."""
    per_rep = pd.DataFrame([
        {"repeat": r, **binary_metrics(g["label"].to_numpy(), g[prob_col].to_numpy(), threshold)}
        for r, g in subj.groupby("repeat")
    ])
    stats = per_rep.drop(columns="repeat").agg(["mean", "std"]).T
    return per_rep, stats


def bootstrap_ci(subj: pd.DataFrame, metric: str, threshold: float = 0.5, n_boot: int = 2000,
                 level: float = 0.95, seed: int = 0, prob_col: str = "prob") -> tuple[float, float, float]:
    """Subject bootstrap CI of a metric averaged across repetitions.

    Subjects (not recordings) are resampled, keeping all of a subject's
    repetitions together; resampling is stratified by diagnosis.
    """
    rng = np.random.default_rng(seed)
    wide = subj.pivot_table(index=["subject_id", "label"], columns="repeat", values=prob_col).reset_index()
    y = wide["label"].to_numpy()
    P = wide.drop(columns=["subject_id", "label"]).to_numpy()

    def stat(idx):
        return np.nanmean([binary_metrics(y[idx], P[idx, r], threshold)[metric] for r in range(P.shape[1])])

    point = stat(np.arange(len(y)))
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boots = [stat(np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]))
             for _ in range(n_boot)]
    a = (1 - level) / 2
    lo, hi = np.nanquantile(boots, [a, 1 - a])
    return float(point), float(lo), float(hi)
