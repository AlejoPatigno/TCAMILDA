"""Post-training analyses on saved fold models: explanations, faithfulness (H1) and sanity checks."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import pandas as pd

from . import model as _model  # noqa: F401  (registers PositionalEmbedding for loading)
from . import splits as S
from .config import Config
from .faithfulness import faithfulness, make_baseline, model_prob_fn, q_grid
from .stats import wilcoxon_greater
from .xai import explain


def load_fold(out_dir: str | Path, rep: int, fold: int):
    """Model, normalization statistics and test subjects of one saved outer fold."""
    out = Path(out_dir)
    model = keras.models.load_model(out / f"model_r{rep}_f{fold}.keras")
    stats = np.load(out / f"norm_r{rep}_f{fold}.npz")
    assign = pd.read_csv(out / "folds.csv")
    _, test_subjects = S.outer_split(assign, rep, fold)
    return model, (stats["mean"], stats["std"]), test_subjects


def normalize(X, stats, eps: float = 1e-6) -> np.ndarray:
    mean, std = stats
    return (np.asarray(X, dtype=np.float32) - mean[None, :, None]) / (std[None, :, None] + eps)


def fold_explanations(model, stats, X, seg: pd.DataFrame, test_subjects, target: str = "pred") -> dict:
    """q_pre, q_post and q_J for every test segment of a fold.

    ``target`` = "pred" explains the predicted class (Section 2.8, test time);
    "true" explains the true class (used for the class prototypes of 2.12).
    """
    rows = S.rows_of(seg, test_subjects)
    Xn = normalize(X[rows], stats)
    seg_te = seg.iloc[rows].reset_index(drop=True)
    tgt = None if target == "pred" else seg_te["label"].to_numpy()
    maps = explain(model, Xn, tgt)
    maps.update({"X": Xn, "seg": seg_te})
    return maps


def fold_faithfulness(model, maps: dict, cfg: Config, map_key: str = "q_joint",
                      max_segments_per_subject: int | None = 3, seed: int = 0) -> pd.DataFrame:
    """Per-segment deletion/insertion results, aggregated to one row per subject."""
    rng = np.random.default_rng(seed)
    seg, X = maps["seg"], maps["X"]
    qs = q_grid(cfg.xai.q_step)
    chosen = (seg.groupby("subject_id").head(max_segments_per_subject).index
              if max_segments_per_subject else seg.index)
    rows = []
    for i in chosen:
        c = int(maps["target"][i])
        B = make_baseline(X[i].shape, cfg.xai.baseline, rng)
        res = faithfulness(model_prob_fn(model, c), X[i], maps[map_key][i], B, qs,
                           cfg.xai.n_random_masks, cfg.xai.min_shift_tokens, rng, cfg.xai.q_points)
        res = {k: v for k, v in res.items() if not k.startswith("curve")}
        rows.append({"subject_id": seg.loc[i, "subject_id"], "label": seg.loc[i, "label"],
                     "task_family": seg.loc[i, "task_family"], "map": map_key, **res})
    per_seg = pd.DataFrame(rows)
    num = per_seg.select_dtypes("number").columns.drop("label")
    return per_seg.groupby(["subject_id", "label", "map"], as_index=False)[list(num)].mean()


def h1_test(per_subject: pd.DataFrame) -> dict:
    """H1: G(q_J) > G_random on test subjects (one-sided Wilcoxon signed-rank)."""
    return wilcoxon_greater(per_subject["gap"].to_numpy(), per_subject["gap_random"].to_numpy())
