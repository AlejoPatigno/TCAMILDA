"""Corpus indexing: one row per recording with subject, label, task and task family.

The subject (not the recording) is the statistical unit (Section 2.2), so every
row carries a cohort-unique ``subject_id`` used later for grouping.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

LABELS = {"HC": 0, "PD": 1}

# Column-name candidates for the demographic variables in corpus metadata.
_AGE_COLUMNS = ("age", "edad", "Age", "AGE", "Edad")
_SEX_COLUMNS = ("sex", "sexo", "gender", "Sex", "SEX", "Gender", "Sexo")
_UPDRS_COLUMNS = ("UPDRS", "UPDRS-III", "UPDRS_III", "MDS-UPDRS-III", "updrs", "UPDRS III")
_HY_COLUMNS = ("H-Y", "HY", "Hoehn & Yahr", "Hoehn-Yahr", "H&Y", "hoehn_yahr")
_DURATION_COLUMNS = ("Duration", "Disease duration", "duration", "Years since diagnosis")


def assign_task_family(task: str, families: dict) -> str:
    """Map a task name to its family using the ordered regex table in the config."""
    name = task.upper()
    for family, pattern in families.items():
        if family == "default":
            continue
        if re.match(pattern, name):
            return family
    return families.get("default", "other")


def index_cohort(cfg: Config, cohort: str) -> pd.DataFrame:
    """List the audio files of a cohort and parse group, task and subject from the names."""
    ccfg = cfg.cohorts[cohort]
    if not ccfg.root or not ccfg.filename_regex:
        raise ValueError(f"Cohort {cohort!r} is not configured (root / filename_regex)")
    audio_dir = Path(ccfg.root) / (ccfg.audio_dir or "")
    pattern = re.compile(ccfg.filename_regex, re.IGNORECASE)

    rows, skipped = [], []
    for name in sorted(os.listdir(audio_dir)):
        if not name.lower().endswith(".wav"):
            continue
        match = pattern.match(name)
        if match is None:
            skipped.append(name)
            continue
        group = match["group"].upper()
        task = match["task"].upper()
        subject_num = int(match["subject"])
        rows.append(
            {
                "path": str(audio_dir / name),
                "filename": name,
                "cohort": cohort,
                "group": group,
                "label": LABELS[group],
                "subject_num": subject_num,
                "subject_id": f"{cohort}-{group}-{subject_num:04d}",
                "task": task,
                "task_family": assign_task_family(task, cfg.task_families),
            }
        )
    if skipped:
        print(f"[index_cohort] {len(skipped)} files did not match filename_regex, e.g. {skipped[:3]}")
    index = pd.DataFrame(rows)
    index["recording_id"] = np.arange(len(index))
    return index


def _first_column(df: pd.DataFrame, candidates) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    lower = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in lower:
            return lower[col.lower()]
    return None


def load_metadata(cfg: Config, cohort: str) -> pd.DataFrame:
    """Subject-level metadata with standardized age, sex and clinical columns.

    The raw columns are kept with a ``meta_`` prefix so nothing is lost.
    """
    ccfg = cfg.cohorts[cohort]
    frames = []
    for group, key in (("HC", "metadata_hc"), ("PD", "metadata_pd")):
        if not ccfg.get(key):
            continue
        df = pd.read_csv(Path(ccfg.root) / ccfg[key])
        id_col = _first_column(df, ("ID", "id", "Id", "subject", "Subject"))
        if id_col is None:
            raise KeyError(f"No ID column in {ccfg[key]}")
        out = pd.DataFrame(
            {
                "subject_id": [f"{cohort}-{group}-{int(v):04d}" for v in df[id_col]],
                "group": group,
                "label": LABELS[group],
            }
        )
        for name, cands in (
            ("age", _AGE_COLUMNS),
            ("sex", _SEX_COLUMNS),
            ("updrs3", _UPDRS_COLUMNS),
            ("hoehn_yahr", _HY_COLUMNS),
            ("disease_duration", _DURATION_COLUMNS),
        ):
            col = _first_column(df, cands)
            out[name] = df[col].values if col else np.nan
        for col in df.columns:
            out[f"meta_{col}"] = df[col].values
        frames.append(out)
    meta = pd.concat(frames, ignore_index=True)
    meta["sex"] = meta["sex"].map(_normalize_sex)
    return meta


def _normalize_sex(value):
    if pd.isna(value):
        return np.nan
    v = str(value).strip().upper()
    if v in ("M", "MALE", "H", "HOMBRE", "MASCULINO"):
        return "M"
    if v in ("F", "FEMALE", "MUJER", "FEMENINO", "W"):
        return "F"
    return v  # numeric codes are left as-is: check the corpus documentation


def check_index_against_metadata(index: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Report subjects present in audio but not in metadata (and vice versa)."""
    audio_ids = set(index["subject_id"])
    meta_ids = set(meta["subject_id"])
    report = pd.DataFrame(
        {
            "audio_only": [sorted(audio_ids - meta_ids)],
            "metadata_only": [sorted(meta_ids - audio_ids)],
            "n_audio_subjects": [len(audio_ids)],
            "n_metadata_subjects": [len(meta_ids)],
        }
    )
    return report


def task_summary(index: pd.DataFrame) -> pd.DataFrame:
    """Recordings and subjects per task family and group; use it to verify the task regexes."""
    return (
        index.groupby(["task_family", "group"])
        .agg(recordings=("recording_id", "size"), subjects=("subject_id", "nunique"),
             tasks=("task", lambda s: ", ".join(sorted(set(s))[:12])))
        .reset_index()
    )


def demographics_table(meta: pd.DataFrame) -> pd.DataFrame:
    """Table 1 of the manuscript, computed from metadata (never copied by hand)."""
    rows = {}
    for group in ("PD", "HC"):
        g = meta[meta["group"] == group]
        rows[group] = {
            "Subjects (M/F)": f"{(g['sex'] == 'M').sum()}/{(g['sex'] == 'F').sum()}",
            "Age, M, mean (SD)": _mean_sd(g.loc[g["sex"] == "M", "age"]),
            "Age, F, mean (SD)": _mean_sd(g.loc[g["sex"] == "F", "age"]),
            "Age range": _range(g["age"]),
            "Disease duration, mean (SD)": _mean_sd(g["disease_duration"]),
            "MDS-UPDRS-III, mean (SD)": _mean_sd(g["updrs3"]),
            "Hoehn & Yahr, median (range)": _median_range(g["hoehn_yahr"]),
        }
    return pd.DataFrame(rows)


def demographic_tests(meta: pd.DataFrame) -> dict:
    """PD vs HC age (Welch t) and sex (chi-square) differences."""
    from scipy import stats

    pd_age = pd.to_numeric(meta.loc[meta["group"] == "PD", "age"], errors="coerce").dropna()
    hc_age = pd.to_numeric(meta.loc[meta["group"] == "HC", "age"], errors="coerce").dropna()
    out = {}
    if len(pd_age) > 1 and len(hc_age) > 1:
        t = stats.ttest_ind(pd_age, hc_age, equal_var=False)
        out["age_welch_t"], out["age_p"] = float(t.statistic), float(t.pvalue)
    table = pd.crosstab(meta["group"], meta["sex"])
    if table.shape == (2, 2):
        chi2, p, _, _ = stats.chi2_contingency(table)
        out["sex_chi2"], out["sex_p"] = float(chi2), float(p)
    return out


def _num(s):
    return pd.to_numeric(s, errors="coerce").dropna()


def _mean_sd(s) -> str:
    s = _num(s)
    return f"{s.mean():.1f} ({s.std():.1f})" if len(s) else "–"


def _range(s) -> str:
    s = _num(s)
    return f"{s.min():.0f}–{s.max():.0f}" if len(s) else "–"


def _median_range(s) -> str:
    s = _num(s)
    return f"{s.median():.1f} ({s.min():.0f}–{s.max():.0f})" if len(s) else "–"
