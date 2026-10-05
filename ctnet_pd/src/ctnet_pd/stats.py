"""Paired comparisons and multiplicity control (Section 2.16)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict:
    """Paired DeLong test for two correlated AUROCs on the same subjects (Sun & Xu, 2014)."""
    y = np.asarray(y).astype(int)
    preds = np.vstack([p1, p2]).astype(float)
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    aucs, v01, v10 = [], [], []
    for k in range(2):
        tx, ty = _midrank(pos[k]), _midrank(neg[k])
        tz = _midrank(np.concatenate([pos[k], neg[k]]))
        aucs.append(tz[:m].sum() / (m * n) - (m + 1) / (2 * n))
        v01.append((tz[:m] - tx) / n)
        v10.append(1 - (tz[m:] - ty) / m)
    aucs = np.array(aucs)
    s01, s10 = np.cov(np.array(v01)), np.cov(np.array(v10))
    cov = s01 / m + s10 / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": float(aucs[0] - aucs[1]),
            "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


def paired_bootstrap(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, metric_fn, n_boot: int = 2000,
                     level: float = 0.95, seed: int = 0) -> dict:
    """CI and two-sided bootstrap p for metric(p1) - metric(p2) on the same subjects."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    point = metric_fn(y, p1) - metric_fn(y, p2)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs.append(metric_fn(y[idx], p1[idx]) - metric_fn(y[idx], p2[idx]))
    diffs = np.asarray(diffs)
    a = (1 - level) / 2
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {"diff": float(point), "ci": np.quantile(diffs, [a, 1 - a]).tolist(), "p": float(min(p, 1.0))}


def wilcoxon_greater(a: np.ndarray, b: np.ndarray) -> dict:
    """One-sided Wilcoxon signed-rank test of a > b on subject-level paired values (H1)."""
    res = stats.wilcoxon(a, b, alternative="greater")
    return {"statistic": float(res.statistic), "p": float(res.pvalue),
            "median_diff": float(np.median(np.asarray(a) - np.asarray(b)))}


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj
