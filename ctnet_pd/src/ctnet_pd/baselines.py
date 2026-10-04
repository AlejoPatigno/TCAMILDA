"""Same-protocol baselines (Experiment II, Table 6).

All baselines use the fold assignment written by ``pipeline.run_within_cohort``
(``folds.csv``), so they are evaluated on exactly the same subjects.

- SVM on per-recording acoustic features (eGeMAPS via openSMILE, or any table);
- frozen pretrained embeddings (WavLM / XLS-R / AST) + logistic regression;
- CNN-only and CTNet-Flatten are CTNet variants: run them with
  ``model.n_transformer_layers=0`` or ``model.head=flatten``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import splits as S


def egemaps_features(index: pd.DataFrame, sample_rate: int = 16000) -> pd.DataFrame:
    """eGeMAPSv02 functionals per recording (requires the ``opensmile`` package)."""
    import opensmile

    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)
    rows = [smile.process_file(p).iloc[0] for p in index["path"]]
    return pd.DataFrame(rows).reset_index(drop=True).assign(recording_id=index["recording_id"].values)


def pretrained_embeddings(index: pd.DataFrame, model_name: str = "microsoft/wavlm-base-plus",
                          max_seconds: float = 20.0, device: str | None = None) -> dict[int, np.ndarray]:
    """Mean-pooled hidden states of every layer per recording: {recording_id: (n_layers, dim)}.

    Works for speech SSL models (WavLM, XLS-R) and AST through the HF auto classes.
    Requires ``transformers`` and ``torch``.
    """
    import librosa
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device).eval()
    sr = getattr(extractor, "sampling_rate", 16000)
    out = {}
    for rec in index.itertuples(index=False):
        y, _ = librosa.load(rec.path, sr=sr, mono=True)
        y = y[: int(max_seconds * sr)]
        inputs = extractor(y, sampling_rate=sr, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**inputs).hidden_states
        out[rec.recording_id] = torch.stack([h.mean(dim=1)[0] for h in hidden]).cpu().numpy()
    return out


def _recording_table(index: pd.DataFrame) -> pd.DataFrame:
    return index[["recording_id", "subject_id", "label", "task_family"]].reset_index(drop=True)


def run_feature_baseline(features: np.ndarray, index: pd.DataFrame, assign: pd.DataFrame,
                         kind: str = "svm", seed: int = 0, n_inner: int = 4) -> pd.DataFrame:
    """Recording-level classifier with subject-grouped inner CV; returns out-of-fold predictions.

    ``features`` is (n_recordings, n_features) aligned with ``index`` rows.
    The output has the same columns as CTNet predictions (prob per recording),
    so ``metrics.aggregate`` applies unchanged.
    """
    rec = _recording_table(index)
    if kind == "svm":
        est = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=seed))
        grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 1e-3, 1e-2, 1e-1]}
    elif kind == "logreg":
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        grid = {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1, 10]}
    else:
        raise ValueError(kind)

    preds = []
    for (rep, k), _ in assign.groupby(["repeat", "fold"]):
        train_subj, test_subj = S.outer_split(assign, rep, k)
        tr, te = S.rows_of(rec, train_subj), S.rows_of(rec, test_subj)
        cv = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed + rep)
        search = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(features[tr], rec["label"].iloc[tr], groups=rec["subject_id"].iloc[tr])
        prob = search.predict_proba(features[te])[:, 1]
        preds.append(rec.iloc[te].assign(repeat=rep, fold=k, prob=prob, variant=kind))
    return pd.concat(preds, ignore_index=True)


def run_embedding_probe(embeddings: dict[int, np.ndarray], index: pd.DataFrame, assign: pd.DataFrame,
                        seed: int = 0, n_inner: int = 4) -> pd.DataFrame:
    """Logistic-regression probe on frozen embeddings; the layer is chosen by inner CV."""
    rec = _recording_table(index)
    E = np.stack([embeddings[r] for r in rec["recording_id"]])  # (n, layers, dim)
    preds = []
    for (rep, k), _ in assign.groupby(["repeat", "fold"]):
        train_subj, test_subj = S.outer_split(assign, rep, k)
        tr, te = S.rows_of(rec, train_subj), S.rows_of(rec, test_subj)
        cv = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed + rep)
        best = (-np.inf, None, None)
        for layer in range(E.shape[1]):
            search = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                                  {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1]}, scoring="roc_auc", cv=cv)
            search.fit(E[tr, layer], rec["label"].iloc[tr], groups=rec["subject_id"].iloc[tr])
            if search.best_score_ > best[0]:
                best = (search.best_score_, layer, search.best_estimator_)
        prob = best[2].predict_proba(E[te, best[1]])[:, 1]
        preds.append(rec.iloc[te].assign(repeat=rep, fold=k, prob=prob, variant="ssl_probe", layer=best[1]))
    return pd.concat(preds, ignore_index=True)
