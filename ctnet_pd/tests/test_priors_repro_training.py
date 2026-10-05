import numpy as np
import pandas as pd

from ctnet_pd.config import apply_overrides
from ctnet_pd.priors import (aperiodic_band, phonatory_band, prior_distribution, recording_prior,
                             shuffled_priors, voicing_transitions)
from ctnet_pd.reproducibility import (contrast_profile, cross_cohort_rho, frequency_profiles,
                                      permutation_null, split_half_ceiling, subject_mean_maps)
from ctnet_pd.training import build_and_compile, fit, js_divergence, predict_proba


def test_concepts_on_synthetic_vowel(cfg):
    sr = cfg.audio.sample_rate
    t = np.arange(int(1.5 * sr)) / sr
    y = 0.5 * np.sin(2 * np.pi * 150 * t) * (t > 0.3) * (t < 1.2)
    n_frames = 1 + len(y) // cfg.spectrogram.hop_length
    f0 = np.full(n_frames, 150.0)
    voiced = np.zeros(n_frames, bool); voiced[20:50] = True
    band = phonatory_band(f0, voiced, cfg)
    assert band.shape == (128, n_frames) and band[:, voiced].max() == 1.0 and band[:, ~voiced].max() == 0
    trans = voicing_transitions(voiced, 128, 3)
    assert trans[0, 20] == 1.0 and trans[0, 50] == 1.0 and trans[0, 35] == 0
    assert aperiodic_band(voiced, cfg)[:, ~voiced].max() == 0
    prior = recording_prior(y.astype(np.float32), n_frames, cfg, (14, 25))
    assert prior.shape[1:] == (14, 25) and 0 <= prior.min() and prior.max() <= 1
    p = prior_distribution(prior)
    np.testing.assert_allclose(p.sum(axis=(1, 2)), 1, rtol=1e-5)


def test_shuffled_prior_is_a_derangement():
    rng = np.random.default_rng(0)
    priors = np.arange(10)[:, None, None] * np.ones((10, 2, 2))
    out = shuffled_priors(priors, np.array([0] * 5 + [1] * 5), rng)
    assert not np.any(out[:, 0, 0] == priors[:, 0, 0])
    assert set(out[:5, 0, 0]) == set(range(5))


def test_reproducibility_statistics():
    rng = np.random.default_rng(0)
    pattern = np.linspace(-1, 1, 14)

    def cohort(n):
        labels = np.repeat([0, 1], n // 2)
        prof = rng.normal(0, 0.3, (n, 14)) + np.outer(labels, pattern)
        return prof, labels

    pa, la = cohort(60)
    pb, lb = cohort(60)
    res = permutation_null(pa, la, pb, lb, n_perm=500)
    assert res["rho"] > 0.8 and res["p_perm"] < 0.01
    ceiling = split_half_ceiling(pa, la, 50)
    assert ceiling["rho_sb_median"] > 0.8
    maps = rng.random((6, 14, 25))
    subjects, means = subject_mean_maps(maps, ["a", "a", "b", "b", "c", "c"])
    assert list(subjects) == ["a", "b", "c"] and frequency_profiles(means).shape == (3, 14)


def test_prior_trainer_runs(cfg):
    import tensorflow as tf

    c = apply_overrides(cfg, ["prior.enabled=true", "prior.lambda_prior=1.0", "tuning.enabled=false",
                              "training.batch_size=4"])
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 128, 229)).astype(np.float32)
    y = np.array([0, 1] * 4)
    w = np.ones(8, np.float32)
    prior = rng.random((8, 14, 25)).astype(np.float32)
    prior /= prior.sum(axis=(1, 2), keepdims=True)
    trainer = build_and_compile(c, seed=0)
    hist, best = fit(trainer, X, y, w, X, y, w, c, epochs=2, prior_tr=prior, prior_val=prior)
    assert np.isfinite(hist.history["prior_loss"]).all() and hist.history["prior_loss"][0] > 0
    assert predict_proba(trainer, X).shape == (8,)
    p = tf.constant(prior[:2]); assert float(tf.reduce_max(tf.abs(js_divergence(p, p)))) < 1e-6
