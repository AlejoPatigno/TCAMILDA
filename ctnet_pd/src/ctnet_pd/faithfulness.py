"""Deletion/insertion curves, matched random masks and ROAR masks (Section 2.11).

Inputs are assumed to be band-standardized (features.BandNormalizer), so the
Gaussian baseline B = mu_f + sigma_f * eps is standard normal noise and the
mean baseline is zero.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .grid import upsample


def make_baseline(shape, kind: str = "gaussian", rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    if kind == "gaussian":
        return rng.standard_normal(shape).astype(np.float32)
    if kind == "mean":
        return np.zeros(shape, dtype=np.float32)
    raise ValueError(kind)


def rank_cells(qmap: np.ndarray) -> np.ndarray:
    """Flat cell indices sorted from most to least relevant (stable ties)."""
    return np.argsort(-qmap.ravel(), kind="stable")


def cell_mask(order: np.ndarray, k: int, grid_shape) -> np.ndarray:
    m = np.zeros(int(np.prod(grid_shape)), dtype=np.float32)
    m[order[:k]] = 1.0
    return m.reshape(grid_shape)


def q_grid(step: float) -> np.ndarray:
    return np.round(np.arange(0.0, 1.0 + 1e-9, step), 6)


def shift_mask(mask: np.ndarray, shift: int) -> np.ndarray:
    """Frequency-preserving control: circular shift along the time (column) axis."""
    return np.roll(mask, shift, axis=-1)


def rowwise_permuted_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Secondary control: permute cells within each frequency row (exact frequency marginal)."""
    return np.stack([rng.permutation(row) for row in mask])


def perturbation_curves(prob_fn, x: np.ndarray, qmap: np.ndarray, baseline: np.ndarray,
                        qs: np.ndarray, masks: list[np.ndarray] | None = None) -> dict:
    """Deletion and insertion curves for one input (Eq. 11).

    ``prob_fn`` maps a batch (n, H, W) to the probability of the explained class.
    ``masks`` overrides the salient masks (used for the random controls).
    """
    grid = qmap.shape
    n_cells = int(np.prod(grid))
    if masks is None:
        order = rank_cells(qmap)
        masks = [cell_mask(order, int(round(q * n_cells)), grid) for q in qs]
    up = np.stack([upsample(m, x.shape) for m in masks])
    x_del = x[None] * (1 - up) + baseline[None] * up
    x_ins = baseline[None] * (1 - up) + x[None] * up
    p = prob_fn(np.concatenate([x_del, x_ins]))
    p_del, p_ins = p[: len(qs)], p[len(qs):]
    return {
        "q": qs, "deletion": p_del, "insertion": p_ins,
        "auc_del": float(np.trapezoid(p_del, qs)), "auc_ins": float(np.trapezoid(p_ins, qs)),
    }


def faithfulness(prob_fn, x: np.ndarray, qmap: np.ndarray, baseline: np.ndarray, qs: np.ndarray,
                 n_random: int, min_shift: int, rng: np.random.Generator,
                 q_points=(0.05, 0.10, 0.20)) -> dict:
    """Gap G = AUC_ins - AUC_del (Eq. 12) for the map and for matched random masks (Eq. 13)."""
    sal = perturbation_curves(prob_fn, x, qmap, baseline, qs)
    order = rank_cells(qmap)
    n_cols = qmap.shape[1]
    n_cells = qmap.size
    rand_gaps, rand_delta = [], {q: [] for q in q_points}
    for _ in range(n_random):
        shift = int(rng.integers(min_shift, n_cols - min_shift + 1))
        masks = [shift_mask(cell_mask(order, int(round(q * n_cells)), qmap.shape), shift) for q in qs]
        r = perturbation_curves(prob_fn, x, qmap, baseline, qs, masks=masks)
        rand_gaps.append(r["auc_ins"] - r["auc_del"])
        for q in q_points:
            i = int(np.argmin(np.abs(qs - q)))
            rand_delta[q].append(r["deletion"][0] - r["deletion"][i])
    p0 = sal["deletion"][0]
    out = {
        "auc_del": sal["auc_del"], "auc_ins": sal["auc_ins"],
        "gap": sal["auc_ins"] - sal["auc_del"], "gap_random": float(np.mean(rand_gaps)),
        "curve_del": sal["deletion"], "curve_ins": sal["insertion"], "p_full": float(p0),
    }
    for q in q_points:
        i = int(np.argmin(np.abs(qs - q)))
        out[f"delta_sal_{q:.2f}"] = float(p0 - sal["deletion"][i])
        out[f"delta_rand_{q:.2f}"] = float(np.mean(rand_delta[q]))
    return out


def model_prob_fn(model, target_class: int, batch_size: int = 128):
    """Probability of ``target_class`` for batches of (n, H, W) inputs."""

    def fn(batch: np.ndarray) -> np.ndarray:
        out = []
        for start in range(0, len(batch), batch_size):
            xb = tf.convert_to_tensor(batch[start:start + batch_size, ..., None], dtype=tf.float32)
            out.append(tf.nn.softmax(model(xb, training=False), axis=-1).numpy()[:, target_class])
        return np.concatenate(out)

    return fn


def roar_mask_inputs(X: np.ndarray, qmaps: np.ndarray, q: float, rng: np.random.Generator,
                     random_control: bool = False, min_shift: int = 6) -> np.ndarray:
    """Replace the top-q cells (or their time-shifted control) with Gaussian noise for ROAR."""
    out = np.array(X, dtype=np.float32, copy=True)
    for n in range(len(X)):
        qmap = qmaps[n]
        mask = cell_mask(rank_cells(qmap), int(round(q * qmap.size)), qmap.shape)
        if random_control:
            mask = shift_mask(mask, int(rng.integers(min_shift, qmap.shape[1] - min_shift + 1)))
        up = upsample(mask, X.shape[1:3])
        out[n] = X[n] * (1 - up) + rng.standard_normal(X.shape[1:3]).astype(np.float32) * up
    return out
