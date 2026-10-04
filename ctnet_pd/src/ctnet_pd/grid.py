"""Mapping between the input spectrogram (128 x 229) and the token grid (14 x 25).

The same tile partition is used to downsample acoustic priors (operator D_RF,
Eq. 7), to upsample relevance maps for display, and to build perturbation
masks, so all three are consistent by construction.
"""

from __future__ import annotations

import numpy as np


def tile_edges(n_in: int, n_out: int) -> np.ndarray:
    """Edges of n_out contiguous tiles covering range(n_in) (adaptive-pooling convention)."""
    return np.floor(np.arange(n_out + 1) * n_in / n_out).astype(int)


def downsample(x: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Tile average of the last two axes: D_RF with uniform weights on the stride tile."""
    rows = tile_edges(x.shape[-2], out_shape[0])
    cols = tile_edges(x.shape[-1], out_shape[1])
    out = np.empty(x.shape[:-2] + tuple(out_shape), dtype=np.float32)
    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            out[..., i, j] = x[..., rows[i]:rows[i + 1], cols[j]:cols[j + 1]].mean(axis=(-2, -1))
    return out


def upsample(x: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Nearest (tile) upsampling of the last two axes, inverse of the tile partition."""
    rows = tile_edges(out_shape[0], x.shape[-2])
    cols = tile_edges(out_shape[1], x.shape[-1])
    row_idx = np.repeat(np.arange(x.shape[-2]), np.diff(rows))
    col_idx = np.repeat(np.arange(x.shape[-1]), np.diff(cols))
    return x[..., row_idx[:, None], col_idx[None, :]]


def mel_row_frequencies(n_rows: int, n_mels: int, sr: int, fmin: float = 0.0,
                        fmax: float | None = None) -> np.ndarray:
    """Approximate centre frequency (Hz) of each token row, for labelling profiles."""
    import librosa

    centres = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax or sr / 2)
    edges = tile_edges(n_mels, n_rows)
    return np.array([centres[edges[i]:edges[i + 1]].mean() for i in range(n_rows)])
