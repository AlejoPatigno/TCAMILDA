import importlib.util

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from ctnet_pd import baselines as B
from ctnet_pd import splits as S
from ctnet_pd.config import apply_overrides
from ctnet_pd.metrics import aggregate, binary_metrics
from tiny_models import save_tiny_wavlm


def _vowel(path, f0, sr=22050, dur=1.5, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    inst = f0 * (1 + jitter * rng.standard_normal(t.size).cumsum() / np.sqrt(t.size))
    phase = 2 * np.pi * np.cumsum(inst) / sr
    y = 0.3 * (np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase))
    sf.write(path, (y + 0.003 * rng.standard_normal(t.size)).astype(np.float32), sr)


def _index(tmp_path, n_subjects=20, recs=2):
    rows, rid = [], 0
    for s in range(n_subjects):
        label = int(s < n_subjects // 2)
        for r in range(recs):
            p = tmp_path / f"s{s}_{r}.wav"
            _vowel(p, 120 + 60 * label, seed=rid)
            rows.append({"path": str(p), "recording_id": rid, "subject_id": f"S{s:02d}", "label": label,
                         "task_family": "vowels"})
            rid += 1
    return pd.DataFrame(rows)


def _assign(index, cfg):
    return S.make_outer_folds(index, n_folds=5, n_repeats=1, seed=cfg.project.seed)


def test_praat_and_egemaps_features(cfg, tmp_path):
    index = _index(tmp_path, n_subjects=2, recs=1)
    praat = B.praat_features(index, cfg, cache_path=tmp_path / "cache" / "praat")
    assert {"jitter_local", "shimmer_local", "hnr_mean", "f1_mean", "f0_sd_semitones"} <= set(praat.columns)
    expected = 120 + 60 * index["label"].to_numpy()
    assert np.all(np.abs(praat["f0_mean"].to_numpy() - expected) < 5)
    assert (tmp_path / "cache" / "praat.csv").exists()
    again = B.praat_features(index, cfg, cache_path=tmp_path / "cache" / "praat")  # from cache
    pd.testing.assert_frame_equal(praat.reset_index(drop=True), again.reset_index(drop=True), check_dtype=False)
    ege = B.egemaps_features(index, cfg)
    assert ege.shape == (2, 89)  # 88 functionals + recording_id
    X = B.feature_matrix(ege, index.iloc[::-1])
    assert X.shape == (2, 88) and np.allclose(X[0], B.feature_matrix(ege, index)[1], equal_nan=True)


def test_feature_baseline_within_and_external(cfg, tmp_path):
    rng = np.random.default_rng(0)
    index = pd.DataFrame({"recording_id": range(80), "subject_id": [f"S{i // 2:02d}" for i in range(80)],
                          "label": [int(i // 2 < 20) for i in range(80)], "task_family": "ddk", "path": ""})
    X = rng.standard_normal((80, 5)) + 1.5 * index["label"].to_numpy()[:, None]
    X[3, 2] = np.nan  # imputed inside the pipeline
    c = apply_overrides(cfg, ["baselines.svm_C=[1.0]", "baselines.svm_gamma=[scale]"])
    assign = _assign(index, c)
    pred = B.run_feature_baseline(X, index, assign, c, kind="svm")
    assert len(pred) == 80 and pred["recording_id"].is_unique
    subj = aggregate(pred)
    assert binary_metrics(subj["label"], subj["prob"])["auroc"] > 0.9
    ext = B.run_feature_external(X, index, X[:10], index.iloc[:10], c, kind="logreg")
    assert len(ext) == 10 and ext["prob"].between(0, 1).all()


def test_embedding_probe_selects_informative_layer(cfg):
    rng = np.random.default_rng(1)
    index = pd.DataFrame({"recording_id": range(60), "subject_id": [f"S{i // 2:02d}" for i in range(60)],
                          "label": [int(i // 2 < 15) for i in range(60)], "task_family": "ddk", "path": ""})
    E = rng.standard_normal((60, 3, 8))
    E[:, 1, :] += 2.0 * index["label"].to_numpy()[:, None]
    pred = B.run_embedding_probe(E, index, _assign(index, cfg), cfg)
    assert (pred["layer"] == 1).all()
    ext = B.run_embedding_external(E, index, E, index, cfg)
    assert (ext["layer"] == 1).all()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None,
                    reason="torch and transformers are optional")
def test_pretrained_embeddings_chunked(cfg, tmp_path):
    index = _index(tmp_path, n_subjects=2, recs=1)
    long_path = tmp_path / "long.wav"
    _vowel(long_path, 150, dur=12.5)  # 2 chunks of 10 s
    index = pd.concat([index, pd.DataFrame([{"path": str(long_path), "recording_id": 99, "subject_id": "S99",
                                              "label": 0, "task_family": "monologue"}])], ignore_index=True)
    model_dir = tmp_path / "tiny_wavlm"
    B._isolated(save_tiny_wavlm, str(model_dir))
    E = B.pretrained_embeddings(index, cfg, model_name=str(model_dir), cache_path=tmp_path / "emb" / "tiny")
    assert E.shape == (3, 3, 16) and np.isfinite(E).all()  # embeddings output + 2 layers
    assert not np.allclose(E[0], E[1])
    cached = B.pretrained_embeddings(index.iloc[::-1], cfg, cache_path=tmp_path / "emb" / "tiny")
    np.testing.assert_allclose(cached, E[::-1])
