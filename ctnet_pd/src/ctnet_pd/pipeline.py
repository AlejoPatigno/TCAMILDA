"""Experiment drivers: within-cohort repeated nested CV (Exp. I/II/IV) and external validation (Exp. III)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import splits as S
from .config import Config, apply_overrides, token_grid, validate_config
from .features import BandNormalizer
from .metrics import aggregate, binary_metrics, youden_threshold
from .training import (build_and_compile, clear_session, fit, predict_proba, sample_weights, tune)
from . import vision


# Spectrogram-based models run through the same nested-CV pipeline (Experiments I-II).
VARIANTS = {
    "ctnet": [],
    "cnn_only": ["model.n_transformer_layers=0"],
    "ctnet_flatten": ["model.head=flatten"],
    "resnet50": ["model.architecture=resnet50"],
    "efficientnetb0": ["model.architecture=efficientnetb0"],
    "vit": ["model.architecture=vit"],      # PyTorch (vision.py), preliminary-study baseline
    "swin": ["model.architecture=swin"],    # PyTorch (vision.py), preliminary-study baseline
}


def variant_config(cfg: Config, variant: str) -> Config:
    """Config for a named model variant (see VARIANTS)."""
    if variant not in VARIANTS:
        raise KeyError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    out = apply_overrides(cfg, VARIANTS[variant])
    validate_config(out)
    return out


def _subset(X, seg, rows):
    return np.asarray(X[rows], dtype=np.float32), seg.iloc[rows].reset_index(drop=True)


def _prior_subset(priors, rows):
    return None if priors is None else np.asarray(priors[rows], dtype=np.float32)


def train_one(cfg: Config, X, seg, train_subj, val_subj, priors=None, seed=0, tune_dir=None,
              tune_project="tune", verbose=0):
    """Fit normalizer, (optionally) tune, and train one model on an inner split.

    Returns (model, normalizer, hp, best_epoch).
    """
    S.assert_disjoint(train_subj, val_subj)
    r_tr, r_val = S.rows_of(seg, train_subj), S.rows_of(seg, val_subj)
    X_tr, seg_tr = _subset(X, seg, r_tr)
    X_val, seg_val = _subset(X, seg, r_val)
    norm = BandNormalizer().fit(X_tr) if cfg.normalization.per_band else None
    if norm is not None:
        X_tr, X_val = norm.transform(X_tr), norm.transform(X_val)
    t = cfg.training
    w_tr = sample_weights(seg_tr, t.subject_balanced, t.class_balanced)
    w_val = sample_weights(seg_val, t.subject_balanced, t.class_balanced)
    y_tr, y_val = seg_tr["label"].to_numpy(), seg_val["label"].to_numpy()
    p_tr, p_val = _prior_subset(priors, r_tr), _prior_subset(priors, r_val)

    if cfg.model.architecture in vision.TORCH_ARCHITECTURES:
        model, hp, best_epoch = vision.train(cfg, X_tr, y_tr, w_tr, X_val, y_val, w_val, seed=seed,
                                             work_dir=tune_dir)
        return model, norm, hp, best_epoch

    hp = {}
    if cfg.tuning.enabled:
        hp = tune(cfg, X_tr, y_tr, w_tr, X_val, y_val, w_val, directory=tune_dir or "tuning",
                  project=tune_project, seed=seed, prior_tr=p_tr, prior_val=p_val)
        clear_session()
    model = build_and_compile(cfg, hp, seed=seed)
    _, best_epoch = fit(model, X_tr, y_tr, w_tr, X_val, y_val, w_val, cfg, prior_tr=p_tr,
                        prior_val=p_val, seed=seed, verbose=verbose)
    return model, norm, hp, best_epoch


def run_within_cohort(cfg: Config, X, seg: pd.DataFrame, out_dir: str | Path, variant: str = "ctnet",
                      priors=None, repeats=None, folds=None, save_models: bool = False,
                      verbose: int = 0) -> pd.DataFrame:
    """Experiment I: repeated, speaker-disjoint, nested CV within one cohort.

    Writes ``folds.csv``, ``predictions.csv`` (segment level) and ``runs.jsonl``
    (hyperparameters and best epoch per fold) to ``out_dir``. Existing outer folds
    present in ``runs.jsonl`` are skipped, so an interrupted run can be resumed.
    """
    out = Path(out_dir) / variant
    out.mkdir(parents=True, exist_ok=True)
    folds_path = out / "folds.csv"
    if folds_path.exists():
        assign = pd.read_csv(folds_path)
    else:
        assign = S.make_outer_folds(seg, cfg.cv.n_outer_folds, cfg.cv.n_repeats, cfg.project.seed)
        assign.to_csv(folds_path, index=False)
    labels = S.subject_table(seg).set_index("subject_id")["label"]

    runs_path, pred_path = out / "runs.jsonl", out / "predictions.csv"
    done = set()
    if runs_path.exists():
        done = {(r["repeat"], r["fold"]) for r in map(json.loads, runs_path.read_text().splitlines())}

    reps = range(cfg.cv.n_repeats) if repeats is None else repeats
    fold_ids = range(cfg.cv.n_outer_folds) if folds is None else folds
    for rep in reps:
        for k in fold_ids:
            if (rep, k) in done:
                continue
            seed = cfg.project.seed + 1000 * rep + k
            train_subj, test_subj = S.outer_split(assign, rep, k)
            inner_tr, inner_val = S.inner_splits(train_subj, labels, cfg.cv.inner, cfg.cv.inner_val_fraction,
                                                 cfg.cv.n_inner_folds, seed)[0]
            S.assert_disjoint(inner_tr, inner_val, test_subj)
            (out / "tuning").mkdir(exist_ok=True)
            model, norm, hp, best_epoch = train_one(
                cfg, X, seg, inner_tr, inner_val, priors, seed, tune_dir=str(out / "tuning"),
                tune_project=f"r{rep}_f{k}", verbose=verbose)
            r_te = S.rows_of(seg, test_subj)
            X_te, seg_te = _subset(X, seg, r_te)
            if norm is not None:
                X_te = norm.transform(X_te)
            pred = seg_te.assign(repeat=rep, fold=k, prob=predict_proba(model, X_te), variant=variant)
            pred.to_csv(pred_path, mode="a", header=not pred_path.exists(), index=False)
            if save_models:
                if isinstance(model, vision.TorchModelHandle):
                    model.save(out / f"model_r{rep}_f{k}.pt")
                else:
                    getattr(model, "base", model).save(out / f"model_r{rep}_f{k}.keras")
                if norm is not None:
                    np.savez(out / f"norm_r{rep}_f{k}.npz", mean=norm.mean_, std=norm.std_)
            if isinstance(model, vision.TorchModelHandle):
                model.cleanup()
            with open(runs_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"repeat": rep, "fold": k, "best_epoch": best_epoch, "hp": hp,
                                     "n_train_subjects": len(inner_tr), "n_val_subjects": len(inner_val),
                                     "n_test_subjects": len(test_subj)}, default=float) + "\n")
            print(f"[{variant}] repeat {rep} fold {k}: best epoch {best_epoch}, hp {hp}")
            clear_session()
    return pd.read_csv(pred_path)


def run_external(cfg: Config, X_src, seg_src, X_tgt, seg_tgt, out_dir: str | Path, variant: str = "ctnet",
                 priors_src=None, source_oof: pd.DataFrame | None = None, verbose: int = 0) -> dict:
    """Experiment III: train on the source cohort, evaluate once on the target cohort.

    Step 1 selects hyperparameters and the number of epochs on a source inner
    split; step 2 retrains on all source subjects for that number of epochs.
    No target subject is used before the final prediction.
    """
    out = Path(out_dir) / variant
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg.project.seed
    labels = S.subject_table(seg_src).set_index("subject_id")["label"]
    all_src = labels.index.to_numpy()
    tr, va = S.inner_splits(all_src, labels, "holdout", cfg.cv.inner_val_fraction, cfg.cv.n_inner_folds, seed)[0]
    (out / "tuning").mkdir(exist_ok=True)
    selected, _, hp, best_epoch = train_one(cfg, X_src, seg_src, tr, va, priors_src, seed,
                                            tune_dir=str(out / "tuning"), tune_project="external", verbose=verbose)
    if isinstance(selected, vision.TorchModelHandle):
        selected.cleanup()
    clear_session()

    norm = BandNormalizer().fit(X_src) if cfg.normalization.per_band else None
    Xs = norm.transform(X_src) if norm else np.asarray(X_src, dtype=np.float32)
    w = sample_weights(seg_src, cfg.training.subject_balanced, cfg.training.class_balanced)
    if cfg.model.architecture in vision.TORCH_ARCHITECTURES:
        model, _, _ = vision.train(cfg, Xs, seg_src["label"].to_numpy(), w, seed=seed, work_dir=out / "tuning",
                                   learning_rates=[hp["learning_rate"]], epochs=best_epoch)
    else:
        model = build_and_compile(cfg, hp, seed=seed)
        fit(model, Xs, seg_src["label"].to_numpy(), w, None, None, None, cfg, epochs=best_epoch,
            prior_tr=None if priors_src is None else np.asarray(priors_src), early_stopping=False, seed=seed,
            verbose=verbose)

    Xt = norm.transform(X_tgt) if norm else np.asarray(X_tgt, dtype=np.float32)
    pred = seg_tgt.assign(prob=predict_proba(model, Xt), variant=variant)
    pred.to_csv(out / "predictions_external.csv", index=False)
    if isinstance(model, vision.TorchModelHandle):
        model.save(out / "model_external.pt")
        model.cleanup()
    else:
        getattr(model, "base", model).save(out / "model_external.keras")

    subj = aggregate(pred)
    thresholds = {"0.5": 0.5}
    if source_oof is not None:
        src_subj = aggregate(source_oof).groupby(["subject_id", "label"], as_index=False)["prob"].mean()
        thresholds["youden_source"] = youden_threshold(src_subj["label"], src_subj["prob"])
    result = {name: binary_metrics(subj["label"], subj["prob"], thr) for name, thr in thresholds.items()}
    result["hp"], result["epochs"] = hp, best_epoch
    (out / "external_metrics.json").write_text(json.dumps(result, indent=2, default=float))
    return result


def grid_shape(cfg: Config) -> tuple[int, int]:
    return token_grid(cfg)
