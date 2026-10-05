"""Audio preprocessing, log-Mel spectrograms and fixed-length segmentation (Section 2.5).

Replaces the preliminary-study pipeline, which (i) truncated every recording to
mean + 1.5 SD samples and (ii) "padded" short recordings with
``np.fft.ifft(np.fft.fft(x), n)``. That operation interpolates the spectrum and
time-stretches the signal (changing F0 and speech rate) and discards the
imaginary part. Here recordings are never stretched: they are cut into
overlapping windows of ``segmentation.frames`` frames and the last window is
right-padded with silence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config


def load_audio(path: str, cfg: Config) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=cfg.audio.sample_rate, mono=cfg.audio.mono)
    if cfg.audio.trim_silence:
        y, _ = librosa.effects.trim(y, top_db=cfg.audio.trim_top_db)
    if cfg.audio.amplitude_norm == "peak":
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
    return y.astype(np.float32)


def log_mel(y: np.ndarray, cfg: Config) -> np.ndarray:
    """Eq. (1): 10 log10 of the Mel power spectrogram, referenced to the recording maximum."""
    import librosa

    s = cfg.spectrogram
    sr = cfg.audio.sample_rate
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=s.n_fft, hop_length=s.hop_length, win_length=s.win_length,
        window=s.window, center=s.center, n_mels=s.n_mels, fmin=s.fmin,
        fmax=s.fmax or sr / 2, power=s.power,
    )
    ref = np.max if s.db_ref == "max" else float(s.db_ref)
    return librosa.power_to_db(mel, ref=ref, top_db=s.top_db).astype(np.float32)


def segment(spec: np.ndarray, frames: int, hop: int, min_fill: float,
            pad_value: str | float = "min") -> tuple[np.ndarray, np.ndarray]:
    """Operator R: cut (n_mels, T) into windows of ``frames`` with hop ``hop``.

    Returns ``(segments, fill)`` where ``fill`` is the fraction of real (non-padded)
    frames per window. A recording shorter than one window yields one padded
    window whatever its length, so no recording is dropped.
    """
    n_mels, total = spec.shape
    value = float(spec.min()) if pad_value == "min" else float(pad_value)
    starts = list(range(0, max(total - frames, 0) + 1, hop))
    if total > frames and starts[-1] + frames < total:
        tail = total - (starts[-1] + hop)
        if tail / frames >= min_fill:
            starts.append(starts[-1] + hop)
    segs, fill = [], []
    for start in starts:
        chunk = spec[:, start:start + frames]
        real = chunk.shape[1]
        if real < frames:
            chunk = np.pad(chunk, ((0, 0), (0, frames - real)), constant_values=value)
        segs.append(chunk)
        fill.append(real / frames)
    return np.stack(segs).astype(np.float32), np.asarray(fill, dtype=np.float32)


def build_features(index: pd.DataFrame, cfg: Config, cache_path: str | Path | None = None,
                   verbose: bool = True) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute segments for every recording of ``index``.

    Returns ``X`` with shape (n_segments, n_mels, frames) stored as float16 to
    save memory, and a segment table carrying every column of ``index`` plus
    ``segment_idx``, ``fill`` and ``n_frames`` (recording length in frames).
    """
    if cache_path is not None and Path(cache_path).exists():
        return load_features(cache_path)

    seg_cfg = cfg.segmentation
    xs, rows = [], []
    for k, rec in enumerate(index.itertuples(index=False)):
        spec = log_mel(load_audio(rec.path, cfg), cfg)
        segs, fill = segment(spec, seg_cfg.frames, seg_cfg.hop_frames, seg_cfg.min_fill,
                             seg_cfg.pad_value)
        xs.append(segs.astype(np.float16))
        base = rec._asdict()
        for j, f in enumerate(fill):
            rows.append({**base, "segment_idx": j, "fill": float(f), "n_frames": spec.shape[1]})
        if verbose and (k + 1) % 250 == 0:
            print(f"[build_features] {k + 1}/{len(index)} recordings")
    X = np.concatenate(xs, axis=0)
    seg = pd.DataFrame(rows)
    if cache_path is not None:
        save_features(cache_path, X, seg)
    return X, seg


def save_features(path: str | Path, X: np.ndarray, seg: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), X)
    seg.to_csv(path.with_suffix(".csv"), index=False)


def load_features(path: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    path = Path(path)
    return np.load(path.with_suffix(".npy"), mmap_mode="r"), pd.read_csv(path.with_suffix(".csv"))


class BandNormalizer:
    """Per-Mel-band standardization fitted on the training partition only (Section 2.4).

    After this transform the Gaussian perturbation baseline of Section 2.11.1
    (mu_f + sigma_f * eps) is simply standard normal noise and the mean
    baseline is zero.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray, batch: int = 2048) -> "BandNormalizer":
        n_mels = X.shape[1]
        total = np.zeros(n_mels)
        total_sq = np.zeros(n_mels)
        count = 0
        for start in range(0, len(X), batch):
            chunk = np.asarray(X[start:start + batch], dtype=np.float64)
            total += chunk.sum(axis=(0, 2))
            total_sq += (chunk ** 2).sum(axis=(0, 2))
            count += chunk.shape[0] * chunk.shape[2]
        self.mean_ = (total / count).astype(np.float32)
        self.std_ = np.sqrt(np.maximum(total_sq / count - (total / count) ** 2, 0)).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_[None, :, None]) / (self.std_[None, :, None] + self.eps)

    def state(self) -> dict:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}
