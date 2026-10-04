import numpy as np
import pandas as pd

from ctnet_pd import splits as S
from ctnet_pd.features import BandNormalizer, log_mel, segment


def test_segment_long_and_short():
    spec = np.random.randn(128, 500).astype(np.float32)
    segs, fill = segment(spec, 229, 115, 0.5)
    assert segs.shape[1:] == (128, 229)
    np.testing.assert_allclose(segs[0], spec[:, :229])
    assert fill[:-1].min() == 1.0
    short = np.random.randn(128, 40).astype(np.float32)
    segs, fill = segment(short, 229, 115, 0.5)
    assert segs.shape == (1, 128, 229) and np.isclose(fill[0], 40 / 229)
    assert np.all(segs[0, :, 40:] == short.min())


def test_log_mel_shape(cfg):
    sr = cfg.audio.sample_rate
    t = np.arange(int(2.0 * sr)) / sr
    y = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    S_db = log_mel(y, cfg)
    assert S_db.shape == (128, 1 + len(y) // cfg.spectrogram.hop_length)
    assert S_db.max() == 0.0 and S_db.min() >= -80.0


def test_band_normalizer():
    X = np.random.randn(50, 128, 229).astype(np.float32) * 5 + 3
    Z = BandNormalizer().fit(X).transform(X)
    np.testing.assert_allclose(Z.mean(axis=(0, 2)), 0, atol=1e-3)
    np.testing.assert_allclose(Z.std(axis=(0, 2)), 1, atol=1e-3)


def test_outer_folds_are_subject_disjoint(toy_segments):
    assign = S.make_outer_folds(toy_segments, n_folds=5, n_repeats=3, seed=0)
    labels = S.subject_table(toy_segments).set_index("subject_id")["label"]
    for (rep, k), _ in assign.groupby(["repeat", "fold"]):
        tr, te = S.outer_split(assign, rep, k)
        S.assert_disjoint(tr, te)
        assert labels.loc[te].sum() == 3  # stratified: 15 PD over 5 folds
        inner_tr, inner_val = S.inner_splits(tr, labels, "holdout", 0.2, 4, 0)[0]
        S.assert_disjoint(inner_tr, inner_val, te)
        assert set(inner_tr) | set(inner_val) == set(tr)
    # each subject is tested exactly once per repeat
    assert (assign.groupby(["repeat", "subject_id"]).size() == 1).all()


def test_inconsistent_labels_rejected():
    df = pd.DataFrame({"subject_id": ["a", "a"], "label": [0, 1]})
    try:
        S.subject_table(df)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
