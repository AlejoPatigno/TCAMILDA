"""Speaker-disjoint partitioning (Section 2.4).

Folds are built on the subject table (one row per subject, stratified by
diagnosis) and then mapped back to recordings and segments, so no subject can
appear in two partitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def subject_table(df: pd.DataFrame) -> pd.DataFrame:
    subjects = df.groupby("subject_id")["label"].agg(["first", "nunique"])
    if (subjects["nunique"] > 1).any():
        bad = subjects.index[subjects["nunique"] > 1].tolist()
        raise ValueError(f"Subjects with inconsistent labels: {bad[:5]}")
    return subjects["first"].rename("label").reset_index()


def make_outer_folds(df: pd.DataFrame, n_folds: int, n_repeats: int, seed: int) -> pd.DataFrame:
    """Assignment table with columns subject_id, label, repeat, fold (test fold of the subject)."""
    subjects = subject_table(df)
    out = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + rep)
        fold = np.empty(len(subjects), dtype=int)
        for k, (_, test_idx) in enumerate(skf.split(subjects["subject_id"], subjects["label"])):
            fold[test_idx] = k
        out.append(subjects.assign(repeat=rep, fold=fold))
    return pd.concat(out, ignore_index=True)


def outer_split(assign: pd.DataFrame, repeat: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Train and test subject IDs of one outer fold."""
    a = assign[assign["repeat"] == repeat]
    return a.loc[a["fold"] != fold, "subject_id"].to_numpy(), a.loc[a["fold"] == fold, "subject_id"].to_numpy()


def inner_splits(train_subjects: np.ndarray, labels: pd.Series, strategy: str, val_fraction: float,
                 n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Subject-disjoint inner (train, validation) splits of an outer training set.

    ``labels`` maps subject_id -> label. ``holdout`` returns a single stratified
    split; ``kfold`` returns ``n_folds`` splits.
    """
    y = labels.loc[train_subjects].to_numpy()
    if strategy == "holdout":
        tr, va = train_test_split(train_subjects, test_size=val_fraction, stratify=y, random_state=seed)
        return [(np.asarray(tr), np.asarray(va))]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(train_subjects[a], train_subjects[b]) for a, b in skf.split(train_subjects, y)]


def assert_disjoint(*subject_sets) -> None:
    sets = [set(s) for s in subject_sets]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise AssertionError(f"Partitions {i} and {j} share subjects: {sorted(overlap)[:5]}")


def rows_of(df: pd.DataFrame, subjects) -> np.ndarray:
    """Positional indices of the rows of ``df`` belonging to ``subjects``."""
    return np.flatnonzero(df["subject_id"].isin(set(subjects)).to_numpy())
