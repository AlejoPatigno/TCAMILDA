"""Acoustic concept maps and the biomarker-guided prior (Section 2.10).

Concept maps are computed from the waveform only (never from saliency maps),
on the same frame grid as the log-Mel spectrogram (same sample rate, hop and
centring), then segmented with the same operator R and downsampled to the
token grid with D_RF (grid.downsample).

The concept list is configurable and must be frozen before any
prior-regularized model is trained.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .features import segment
from .grid import downsample


def _mel_centres(cfg: Config) -> np.ndarray:
    import librosa

    s = cfg.spectrogram
    sr = cfg.audio.sample_rate
    return librosa.mel_frequencies(n_mels=s.n_mels, fmin=s.fmin, fmax=s.fmax or sr / 2)


def f0_track(y: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """pYIN F0 (Hz, NaN when unvoiced) and voicing flags on the spectrogram frame grid."""
    import librosa

    p, s = cfg.prior, cfg.spectrogram
    f0, voiced, _ = librosa.pyin(y, fmin=p.f0_min, fmax=p.f0_max, sr=cfg.audio.sample_rate,
                                 frame_length=s.n_fft, hop_length=s.hop_length, center=s.center)
    return f0, voiced.astype(bool)


def phonatory_band(f0: np.ndarray, voiced: np.ndarray, cfg: Config) -> np.ndarray:
    """Concept 1: Mel bins between F0 and ``harmonics`` x F0 in voiced frames (soft edges)."""
    centres = np.maximum(_mel_centres(cfg), 1.0)  # avoid log(0) for the 0 Hz bin
    tol = cfg.prior.band_tolerance_semitones
    out = np.zeros((len(centres), len(f0)), dtype=np.float32)
    for t in np.flatnonzero(voiced & np.isfinite(f0)):
        lo, hi = f0[t], cfg.prior.harmonics * f0[t]
        semis = np.where(centres < lo, 12 * np.log2(lo / centres),
                         np.where(centres > hi, 12 * np.log2(centres / hi), 0.0))
        out[:, t] = np.exp(-0.5 * (semis / tol) ** 2)
    return out


def voicing_transitions(voiced: np.ndarray, n_mels: int, halfwidth: int) -> np.ndarray:
    """Concept 3: all bins in a window of +-halfwidth frames around voicing onsets/offsets."""
    changes = np.flatnonzero(np.diff(voiced.astype(int)) != 0) + 1
    weight = np.zeros(len(voiced), dtype=np.float32)
    for c in changes:
        lo, hi = max(0, c - halfwidth), min(len(voiced), c + halfwidth + 1)
        dist = np.abs(np.arange(lo, hi) - c)
        weight[lo:hi] = np.maximum(weight[lo:hi], 1 - dist / (halfwidth + 1))
    return np.repeat(weight[None, :], n_mels, axis=0)


def aperiodic_band(voiced: np.ndarray, cfg: Config) -> np.ndarray:
    """Concept 4: bins above the cutoff frequency in voiced frames (HNR / breathiness)."""
    centres = _mel_centres(cfg)
    band = (centres >= cfg.prior.aperiodic_cutoff_hz).astype(np.float32)
    return band[:, None] * voiced[None, :].astype(np.float32)


def formant_bands(y: np.ndarray, voiced: np.ndarray, cfg: Config) -> np.ndarray:
    """Concept 2: bands around F1 and F2 (requires praat-parselmouth)."""
    import parselmouth  # optional dependency

    s = cfg.spectrogram
    sr = cfg.audio.sample_rate
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    formant = snd.to_formant_burg(time_step=s.hop_length / sr)
    centres = _mel_centres(cfg)
    bw = cfg.prior.formant_bandwidth_hz
    times = np.arange(len(voiced)) * s.hop_length / sr
    out = np.zeros((len(centres), len(voiced)), dtype=np.float32)
    for t in np.flatnonzero(voiced):
        for k in (1, 2):
            f = formant.get_value_at_time(k, times[t])
            if np.isfinite(f):
                out[:, t] = np.maximum(out[:, t], np.exp(-0.5 * ((centres - f) / bw) ** 2))
    return out


def concept_maps(y: np.ndarray, n_frames: int, cfg: Config) -> dict[str, np.ndarray]:
    """All configured concept maps for one (preprocessed) waveform, shape (n_mels, n_frames)."""
    f0, voiced = f0_track(y, cfg)
    f0, voiced = f0[:n_frames], voiced[:n_frames]
    n_mels = cfg.spectrogram.n_mels
    builders = {
        "phonatory_band": lambda: phonatory_band(f0, voiced, cfg),
        "voicing_transitions": lambda: voicing_transitions(voiced, n_mels, cfg.prior.transition_halfwidth_frames),
        "aperiodic_band": lambda: aperiodic_band(voiced, cfg),
        "formant_bands": lambda: formant_bands(y, voiced, cfg),
    }
    unknown = set(cfg.prior.concepts) - set(builders)
    if unknown:
        raise ValueError(f"Unknown concepts: {unknown}")
    return {name: builders[name]() for name in cfg.prior.concepts}


def recording_prior(y: np.ndarray, n_frames: int, cfg: Config, grid_shape) -> np.ndarray:
    """Prior Pi for every segment of a recording: (n_segments, rows, cols), values in [0, 1].

    Uniform, pre-specified concept weights (omega_k = 1/K); padded frames carry no prior.
    """
    maps = concept_maps(y, n_frames, cfg)
    weights = {k: 1.0 / len(maps) for k in maps}
    seg = cfg.segmentation
    combined = None
    for name, m in maps.items():
        segs, _ = segment(m, seg.frames, seg.hop_frames, seg.min_fill, pad_value=0.0)
        low = downsample(segs, grid_shape)
        combined = weights[name] * low if combined is None else combined + weights[name] * low
    return combined.astype(np.float32)


def prior_distribution(prior: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """p = N_1(Pi)."""
    p = np.asarray(prior, dtype=np.float64) + eps
    return (p / p.sum(axis=(-2, -1), keepdims=True)).astype(np.float32)


def shuffled_priors(priors: np.ndarray, groups, rng: np.random.Generator) -> np.ndarray:
    """Random-Prior control: give each segment the prior of a different segment of the same
    group (e.g. task family), breaking the recording-specific acoustic correspondence."""
    groups = np.asarray(groups)
    out = np.empty_like(priors)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        perm = rng.permutation(idx)
        if len(idx) > 1:
            fixed = perm == idx
            while fixed.any():
                perm = rng.permutation(idx)
                fixed = perm == idx
        out[idx] = priors[perm]
    return out
