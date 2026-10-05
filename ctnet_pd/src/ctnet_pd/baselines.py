"""Same-protocol baselines (Experiment II, Table 6) that are not trained on log-Mel segments.

- Handcrafted features per recording: eGeMAPSv02 functionals (openSMILE) and
  classical Praat measures (Parselmouth), classified with an RBF SVM.
- Frozen pretrained encoders (WavLM, XLS-R, AST): mean-pooled hidden states of
  every layer, a logistic-regression probe, and the layer chosen in inner CV.

All of them use the outer folds written by ``pipeline.run_within_cohort``
(``folds.csv``), an inner CV grouped by subject, class-balanced and
subject-balanced sample weights, and return recording-level predictions with
the same columns as CTNet, so ``metrics.aggregate`` applies unchanged.

The log-Mel CNN baselines (CNN-only, CTNet-Flatten, ResNet-50, EfficientNet-B0)
run through ``pipeline.run_within_cohort`` with ``pipeline.VARIANTS``.
"""

from __future__ import annotations

import copy
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import splits as S
from .config import Config
from .features import load_audio

REC_COLUMNS = ["recording_id", "subject_id", "label", "task_family"]


# ------------------------------------------------------------------ feature extraction
#
# openSMILE and PyTorch crash (heap corruption / segfault) when they run in a
# process where TensorFlow is already loaded, which is the case in the
# experiment notebook. Their extraction therefore runs in a fresh Python
# interpreter that imports only this module, never TensorFlow. A plain
# subprocess is used instead of multiprocessing "spawn", which would re-import
# the caller's main script (and TensorFlow with it) in the child.

_CHILD = ("import pickle, sys; fn, args = pickle.load(open(sys.argv[1], 'rb')); "
          "pickle.dump(fn(*args), open(sys.argv[2], 'wb'))")


def _isolated(fn, *args):
    """Run ``fn(*args)`` (a module-level function) in a separate interpreter and return its result."""
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "in.pkl", Path(tmp) / "out.pkl"
        with open(inp, "wb") as fh:
            pickle.dump((fn, args), fh)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
        proc = subprocess.run([sys.executable, "-c", _CHILD, str(inp), str(out)], env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Isolated extraction failed (exit {proc.returncode}):\n{proc.stderr[-3000:]}")
        with open(out, "rb") as fh:
            return pickle.load(fh)


def _audio_cfg(cfg: Config, sample_rate: int) -> Config:
    """Same preprocessing as the spectrogram models, at another sampling rate."""
    out = copy.deepcopy(cfg)
    out.audio.sample_rate = sample_rate
    return out


def _cached(cache_path, index: pd.DataFrame):
    if cache_path is None:
        return None
    path = Path(cache_path)
    if path.with_suffix(".csv").exists():
        table = pd.read_csv(path.with_suffix(".csv"))
        if set(index["recording_id"]) <= set(table["recording_id"]):
            return table.set_index("recording_id").loc[index["recording_id"]].reset_index()
    return None


def egemaps_features(index: pd.DataFrame, cfg: Config, cache_path=None) -> pd.DataFrame:
    """eGeMAPSv02 functionals (88 features) per recording; requires ``opensmile``."""
    cached = _cached(cache_path, index)
    if cached is not None:
        return cached
    rows = _isolated(_egemaps_rows, index[["path", "recording_id"]].to_dict("records"), cfg)
    table = pd.DataFrame(rows)
    _save_table(table, cache_path)
    return table


def _egemaps_rows(records: list[dict], cfg: Config) -> list[dict]:
    import opensmile

    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)
    sr = cfg.audio.sample_rate
    return [{"recording_id": r["recording_id"],
             **smile.process_signal(load_audio(r["path"], cfg), sr).iloc[0].to_dict()} for r in records]


def praat_features(index: pd.DataFrame, cfg: Config, cache_path=None) -> pd.DataFrame:
    """Classical phonation and articulation measures per recording (Parselmouth).

    F0 mean and SD (Hz and semitones), jitter (local, RAP, PPQ5), shimmer
    (local, APQ3, APQ5), mean HNR, F1/F2 mean and SD, intensity mean and SD,
    voiced fraction and duration. Measures that Praat cannot compute for a
    recording (e.g. too few voiced periods) are NaN and imputed inside the
    classifier pipeline with training-fold medians.
    """
    cached = _cached(cache_path, index)
    if cached is not None:
        return cached
    rows = [{"recording_id": rec.recording_id, **_praat_measures(load_audio(rec.path, cfg),
                                                                  cfg.audio.sample_rate, cfg)}
            for rec in index.itertuples(index=False)]
    table = pd.DataFrame(rows)
    _save_table(table, cache_path)
    return table


def _praat_measures(y: np.ndarray, sr: int, cfg: Config) -> dict:
    import parselmouth
    from parselmouth.praat import call

    f0min, f0max = cfg.baselines.praat_f0_min, cfg.baselines.praat_f0_max
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    out = {"duration": snd.get_total_duration()}

    def safe(name, fn):
        try:
            value = float(fn())
            out[name] = value if np.isfinite(value) else np.nan
        except Exception:  # Praat raises on unvoiced or very short signals
            out[name] = np.nan

    pitch = snd.to_pitch(time_step=0.01, pitch_floor=f0min, pitch_ceiling=f0max)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    out["voiced_fraction"] = float(len(voiced) / max(len(f0), 1))
    out["f0_mean"] = float(voiced.mean()) if len(voiced) else np.nan
    out["f0_sd"] = float(voiced.std()) if len(voiced) > 1 else np.nan
    out["f0_sd_semitones"] = (float(np.std(12 * np.log2(voiced / voiced.mean())))
                              if len(voiced) > 1 else np.nan)

    pp = call(snd, "To PointProcess (periodic, cc)", f0min, f0max)
    safe("jitter_local", lambda: call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
    safe("jitter_rap", lambda: call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3))
    safe("jitter_ppq5", lambda: call(pp, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3))
    safe("shimmer_local", lambda: call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    safe("shimmer_apq3", lambda: call([snd, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    safe("shimmer_apq5", lambda: call([snd, pp], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    safe("hnr_mean", lambda: call(snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=f0min), "Get mean", 0, 0))

    intensity = snd.to_intensity(minimum_pitch=f0min).values.ravel()
    out["intensity_mean"] = float(intensity.mean()) if intensity.size else np.nan
    out["intensity_sd"] = float(intensity.std()) if intensity.size else np.nan

    formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500)
    times = pitch.xs()[f0 > 0]
    for k in (1, 2):
        vals = np.array([formant.get_value_at_time(k, t) for t in times], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[f"f{k}_mean"] = float(vals.mean()) if len(vals) else np.nan
        out[f"f{k}_sd"] = float(vals.std()) if len(vals) > 1 else np.nan
    return out


def pretrained_embeddings(index: pd.DataFrame, cfg: Config, model_name: str | None = None,
                          cache_path=None, device: str | None = None) -> np.ndarray:
    """Mean-pooled hidden states of every layer: array (n_recordings, n_layers, dim) aligned with ``index``.

    ``model_name`` is a key of ``baselines.embedding_models`` (wavlm, xlsr, ast),
    a Hugging Face id or a local directory saved with ``save_pretrained``.
    Recordings longer than ``baselines.chunk_seconds`` are embedded chunk by
    chunk and averaged with weights proportional to chunk length, so long
    monologues are not truncated. Extraction runs in a separate interpreter
    (see ``_isolated``). Requires ``transformers`` and ``torch``.
    """
    if cache_path is not None and Path(cache_path).with_suffix(".npy").exists():
        ids = pd.read_csv(Path(cache_path).with_suffix(".ids.csv"))["recording_id"].to_numpy()
        E = np.load(Path(cache_path).with_suffix(".npy"))
        pos = {r: i for i, r in enumerate(ids)}
        if all(r in pos for r in index["recording_id"]):
            return E[[pos[r] for r in index["recording_id"]]]

    name = cfg.baselines.embedding_models.get(model_name, model_name)
    E = _isolated(_embed_records, index["path"].tolist(), cfg, name, device)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(Path(cache_path).with_suffix(".npy"), E)
        index[["recording_id"]].to_csv(Path(cache_path).with_suffix(".ids.csv"), index=False)
    return E


def _embed_records(paths: list[str], cfg: Config, name: str, device) -> np.ndarray:
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    extractor = AutoFeatureExtractor.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    sr = int(getattr(extractor, "sampling_rate", 16000))
    acfg = _audio_cfg(cfg, sr)
    chunk = int(cfg.baselines.chunk_seconds * sr)
    out = []
    for path in paths:
        y = load_audio(path, acfg)
        starts = list(range(0, max(len(y) - sr, 1), chunk))  # a final chunk shorter than 1 s is dropped
        pooled, weights = [], []
        for start in starts:
            piece = y[start:start + chunk]
            inputs = extractor(piece, sampling_rate=sr, return_tensors="pt").to(device)
            with torch.no_grad():
                hidden = model(**inputs, output_hidden_states=True).hidden_states
            pooled.append(torch.stack([h.mean(dim=1)[0] for h in hidden]).cpu().numpy())
            weights.append(len(piece))
        out.append(np.average(np.stack(pooled), axis=0, weights=np.asarray(weights, dtype=float)))
    return np.stack(out).astype(np.float32)


def _save_table(table: pd.DataFrame, cache_path) -> None:
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(Path(cache_path).with_suffix(".csv"), index=False)


def feature_matrix(table: pd.DataFrame, index: pd.DataFrame) -> np.ndarray:
    """Feature table rows aligned with ``index`` (by recording_id), as a float matrix."""
    t = table.set_index("recording_id").loc[index["recording_id"]]
    return t.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)


# ------------------------------------------------------------------ classifiers

def _estimator(kind: str, cfg: Config, seed: int):
    b = cfg.baselines
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if kind == "svm":
        steps.append(("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)))
        grid = {"clf__C": list(b.svm_C), "clf__gamma": list(b.svm_gamma)}
    elif kind == "logreg":
        steps.append(("clf", LogisticRegression(max_iter=5000, class_weight="balanced")))
        grid = {"clf__C": list(b.logreg_C)}
    else:
        raise ValueError(kind)
    return Pipeline(steps), grid


def _subject_weights(rec: pd.DataFrame) -> np.ndarray:
    """Each subject's recordings share a total weight of 1 (as in the CTNet loss)."""
    counts = rec.groupby("subject_id")["subject_id"].transform("size").to_numpy()
    w = 1.0 / counts
    return w / w.mean()


def fit_classifier(X: np.ndarray, rec: pd.DataFrame, cfg: Config, kind: str, seed: int) -> GridSearchCV:
    """Grid search with a subject-grouped, stratified inner CV, scored by AUROC."""
    est, grid = _estimator(kind, cfg, seed)
    cv = StratifiedGroupKFold(n_splits=cfg.baselines.n_inner_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1, error_score="raise")
    search.fit(X, rec["label"].to_numpy(), groups=rec["subject_id"].to_numpy(),
               clf__sample_weight=_subject_weights(rec))
    return search


def _fit_best_layer(E: np.ndarray, rec: pd.DataFrame, cfg: Config, seed: int):
    """Probe every layer with logistic regression; keep the best inner-CV AUROC."""
    best = None
    for layer in range(E.shape[1]):
        search = fit_classifier(E[:, layer], rec, cfg, "logreg", seed)
        if best is None or search.best_score_ > best[0]:
            best = (search.best_score_, layer, search)
    return best[1], best[2]


# ------------------------------------------------------------------ experiments

def _recordings(index: pd.DataFrame) -> pd.DataFrame:
    return index[REC_COLUMNS].reset_index(drop=True)


def run_feature_baseline(X: np.ndarray, index: pd.DataFrame, assign: pd.DataFrame, cfg: Config,
                         kind: str = "svm", variant: str = "features_svm") -> pd.DataFrame:
    """Experiment I/II: out-of-fold recording predictions on the CTNet outer folds.

    ``X`` is (n_recordings, n_features) aligned with ``index`` rows.
    """
    rec = _recordings(index)
    preds = []
    for (rep, k), _ in assign.groupby(["repeat", "fold"]):
        seed = cfg.project.seed + 1000 * rep + k
        train_subj, test_subj = S.outer_split(assign, rep, k)
        tr, te = S.rows_of(rec, train_subj), S.rows_of(rec, test_subj)
        search = fit_classifier(X[tr], rec.iloc[tr], cfg, kind, seed)
        preds.append(rec.iloc[te].assign(repeat=rep, fold=k, prob=search.predict_proba(X[te])[:, 1],
                                         variant=variant, params=str(search.best_params_)))
    return pd.concat(preds, ignore_index=True)


def run_embedding_probe(E: np.ndarray, index: pd.DataFrame, assign: pd.DataFrame, cfg: Config,
                        variant: str = "embedding_probe") -> pd.DataFrame:
    """Experiment II: frozen-encoder probe with the layer selected in the inner CV of each fold."""
    rec = _recordings(index)
    preds = []
    for (rep, k), _ in assign.groupby(["repeat", "fold"]):
        seed = cfg.project.seed + 1000 * rep + k
        train_subj, test_subj = S.outer_split(assign, rep, k)
        tr, te = S.rows_of(rec, train_subj), S.rows_of(rec, test_subj)
        layer, search = _fit_best_layer(E[tr], rec.iloc[tr], cfg, seed)
        preds.append(rec.iloc[te].assign(repeat=rep, fold=k, prob=search.predict_proba(E[te, layer])[:, 1],
                                         variant=variant, layer=layer, params=str(search.best_params_)))
    return pd.concat(preds, ignore_index=True)


def run_feature_external(X_src, index_src, X_tgt, index_tgt, cfg: Config, kind: str = "svm",
                         variant: str = "features_svm") -> pd.DataFrame:
    """Experiment III for feature baselines: fit on all source recordings, predict the target once."""
    rec_src, rec_tgt = _recordings(index_src), _recordings(index_tgt)
    search = fit_classifier(X_src, rec_src, cfg, kind, cfg.project.seed)
    return rec_tgt.assign(prob=search.predict_proba(X_tgt)[:, 1], variant=variant,
                          params=str(search.best_params_))


def run_embedding_external(E_src, index_src, E_tgt, index_tgt, cfg: Config,
                           variant: str = "embedding_probe") -> pd.DataFrame:
    """Experiment III for frozen encoders: layer and C chosen on the source cohort only."""
    rec_src, rec_tgt = _recordings(index_src), _recordings(index_tgt)
    layer, search = _fit_best_layer(E_src, rec_src, cfg, cfg.project.seed)
    return rec_tgt.assign(prob=search.predict_proba(E_tgt[:, layer])[:, 1], variant=variant, layer=layer,
                          params=str(search.best_params_))
