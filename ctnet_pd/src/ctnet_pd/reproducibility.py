"""Cross-cohort reproducibility of explanations against null references (Section 2.12)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def subject_mean_maps(maps: np.ndarray, subject_ids) -> tuple[np.ndarray, np.ndarray]:
    """Average segment maps per subject. Returns (subjects, maps)."""
    s = pd.Series(np.asarray(subject_ids))
    subjects = s.unique()
    idx = {k: v for k, v in s.groupby(s).groups.items()}
    return subjects, np.stack([maps[np.asarray(idx[k])].mean(axis=0) for k in subjects])


def frequency_profiles(maps: np.ndarray) -> np.ndarray:
    """Eq. (14): marginalize over time (columns) -> one value per frequency row."""
    return maps.mean(axis=-1)


def contrast_profile(profiles: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    return profiles[labels == 1].mean(axis=0) - profiles[labels == 0].mean(axis=0)


def spearman(a, b) -> float:
    r = stats.spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else 0.0


def cross_cohort_rho(prof_a, lab_a, prof_b, lab_b) -> float:
    """Eq. (15) on the class-contrast profiles."""
    return spearman(contrast_profile(prof_a, lab_a), contrast_profile(prof_b, lab_b))


def permutation_null(prof_a, lab_a, prof_b, lab_b, n_perm: int, seed: int = 0) -> dict:
    """Label-permutation null: permute diagnoses within each cohort independently."""
    rng = np.random.default_rng(seed)
    observed = cross_cohort_rho(prof_a, lab_a, prof_b, lab_b)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = cross_cohort_rho(prof_a, rng.permutation(lab_a), prof_b, rng.permutation(lab_b))
    p = (1 + np.sum(null >= observed)) / (1 + n_perm)
    return {"rho": observed, "p_perm": float(p), "null_q95": float(np.quantile(null, 0.95)), "null": null}


def split_half_ceiling(profiles, labels, n_splits: int, seed: int = 0) -> dict:
    """Noise ceiling: stratified split-half rho of the contrast profile, Spearman-Brown corrected."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    vals = []
    for _ in range(n_splits):
        half = np.zeros(len(labels), dtype=bool)
        for c in (0, 1):
            idx = np.flatnonzero(labels == c)
            half[rng.choice(idx, len(idx) // 2, replace=False)] = True
        r = spearman(contrast_profile(profiles[half], labels[half]),
                     contrast_profile(profiles[~half], labels[~half]))
        vals.append(2 * r / (1 + r) if r > -1 else -1.0)
    vals = np.asarray(vals)
    return {"rho_sb_median": float(np.median(vals)), "rho_sb": vals}


def normalized_rho(rho_cross: float, ceiling_a: float, ceiling_b: float) -> float:
    denom = np.sqrt(max(ceiling_a, 1e-12) * max(ceiling_b, 1e-12))
    return float(rho_cross / denom)


def partial_spearman(x, y, control) -> float:
    """Spearman correlation of x and y after removing the rank-linear effect of ``control``."""
    rx, ry, rc = (stats.rankdata(v) for v in (x, y, control))
    res = []
    for r in (rx, ry):
        beta = np.polyfit(rc, r, 1)
        res.append(r - np.polyval(beta, rc))
    return float(np.corrcoef(res[0], res[1])[0, 1])


def bootstrap_rho(prof_a, lab_a, prof_b, lab_b, n_boot: int, level: float = 0.95, seed: int = 0) -> list:
    """Subject bootstrap within each cohort (stratified by diagnosis)."""
    rng = np.random.default_rng(seed)
    lab_a, lab_b = np.asarray(lab_a), np.asarray(lab_b)
    vals = []
    for _ in range(n_boot):
        ia = _strat_resample(lab_a, rng)
        ib = _strat_resample(lab_b, rng)
        vals.append(cross_cohort_rho(prof_a[ia], lab_a[ia], prof_b[ib], lab_b[ib]))
    a = (1 - level) / 2
    return np.quantile(vals, [a, 1 - a]).tolist()


def _strat_resample(labels, rng):
    return np.concatenate([rng.choice(np.flatnonzero(labels == c), (labels == c).sum()) for c in (0, 1)])
