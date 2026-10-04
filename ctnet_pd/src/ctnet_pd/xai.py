"""Pre-/post-attention relevance maps and comparators (Sections 2.8-2.9).

All primary maps live on the native token grid (14 x 25). The explained
quantity is the logit contrast o_c - o_{c'} for the binary task.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .grid import upsample
from .model import explainer_model

EPS = 1e-8


def logit_contrast(logits: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    """o_target - o_other for a binary classifier; ``target`` is an int vector (batch,)."""
    target = tf.cast(target, tf.int32)
    sign = tf.cast(2 * target - 1, logits.dtype)  # +1 for class 1, -1 for class 0
    return sign * (logits[:, 1] - logits[:, 0])


def relevance_maps(model, X: np.ndarray, target=None, batch_size: int = 64, explainer=None) -> dict:
    """R_pre (Grad-CAM on the stem, Eq. 4) and R_post (exact token contribution, Eq. 5).

    ``target`` defaults to the predicted class. Returns non-negative maps of shape
    (n, rows, cols) plus the logits and explained targets.
    """
    explainer = explainer or explainer_model(model)
    pre, post, logits_all, targets = [], [], [], []
    for start in range(0, len(X), batch_size):
        xb = tf.convert_to_tensor(_as_input(X[start:start + batch_size]))
        with tf.GradientTape(persistent=True) as tape:
            stem, tokens, logits = explainer(xb, training=False)
            tb = (tf.argmax(logits, axis=-1) if target is None
                  else tf.convert_to_tensor(np.asarray(target)[start:start + batch_size]))
            od = logit_contrast(logits, tb)
        g_stem = tape.gradient(od, stem)
        g_tok = tape.gradient(od, tokens)
        del tape
        alpha = tf.reduce_mean(g_stem, axis=(1, 2), keepdims=True)
        r_pre = tf.nn.relu(tf.reduce_sum(alpha * stem, axis=-1))
        rows, cols = stem.shape[1], stem.shape[2]
        # grad x activation on the encoder tokens; exact (sums to o_delta - b_delta) for the GAP head
        phi = tf.reduce_sum(g_tok * tokens, axis=-1)
        r_post = tf.nn.relu(tf.reshape(phi, (-1, rows, cols)))
        pre.append(r_pre.numpy())
        post.append(r_post.numpy())
        logits_all.append(logits.numpy())
        targets.append(np.asarray(tb))
    return {"pre": np.concatenate(pre), "post": np.concatenate(post),
            "logits": np.concatenate(logits_all), "target": np.concatenate(targets)}


def token_contributions(model, X: np.ndarray, target, explainer=None) -> tuple[np.ndarray, np.ndarray]:
    """Signed phi_i of Eq. (5) and the logit contrast, for checking the exact decomposition."""
    explainer = explainer or explainer_model(model)
    xb = tf.convert_to_tensor(_as_input(X))
    with tf.GradientTape() as tape:
        _, tokens, logits = explainer(xb, training=False)
        od = logit_contrast(logits, tf.convert_to_tensor(target))
    g = tape.gradient(od, tokens)
    return tf.reduce_sum(g * tokens, axis=-1).numpy(), od.numpy()


def sum_normalize(R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """N_1: map -> probability distribution over the grid cells."""
    R = np.asarray(R, dtype=np.float64) + eps
    return (R / R.sum(axis=(-2, -1), keepdims=True)).astype(np.float32)


def joint_map(r_pre: np.ndarray, r_post: np.ndarray) -> np.ndarray:
    """Eq. (6): q_J = (q_pre + q_post) / 2 — no learnable fusion."""
    return 0.5 * (sum_normalize(r_pre) + sum_normalize(r_post))


def minmax(M: np.ndarray) -> np.ndarray:
    lo = M.min(axis=(-2, -1), keepdims=True)
    hi = M.max(axis=(-2, -1), keepdims=True)
    return (M - lo) / (hi - lo + EPS)


def explain(model, X: np.ndarray, target=None, batch_size: int = 64) -> dict:
    """Convenience wrapper returning q_pre, q_post and q_J (sum-normalized, grid resolution)."""
    maps = relevance_maps(model, X, target, batch_size)
    maps["q_pre"] = sum_normalize(maps["pre"])
    maps["q_post"] = sum_normalize(maps["post"])
    maps["q_joint"] = 0.5 * (maps["q_pre"] + maps["q_post"])
    return maps


def to_input_resolution(M: np.ndarray, input_hw=(128, 229)) -> np.ndarray:
    """Tile upsampling for display, then min-max scaling to [0, 1]."""
    return minmax(upsample(M, input_hw))


# ----------------------------------------------------------------- comparators (2.9)

def gradient_saliency(model, X: np.ndarray, target=None, batch_size: int = 64) -> np.ndarray:
    """|d o_delta / d X| at input resolution.

    This is what ``tf_keras_vis.saliency.Saliency`` computed in the preliminary
    study; it is vanilla gradient saliency, not LRP, and is reported as such.
    """
    out = []
    for start in range(0, len(X), batch_size):
        xb = tf.convert_to_tensor(_as_input(X[start:start + batch_size]))
        with tf.GradientTape() as tape:
            tape.watch(xb)
            logits = model(xb, training=False)
            tb = (tf.argmax(logits, axis=-1) if target is None
                  else tf.convert_to_tensor(np.asarray(target)[start:start + batch_size]))
            od = logit_contrast(logits, tb)
        out.append(tf.abs(tape.gradient(od, xb))[..., 0].numpy())
    return np.concatenate(out)


def score_cam(model, X: np.ndarray, target=None, baseline: float = 0.0) -> np.ndarray:
    """Score-CAM on the stem output, at grid resolution (one forward pass per channel)."""
    explainer = explainer_model(model)
    maps = []
    for n in range(len(X)):
        x = _as_input(X[n:n + 1])
        stem, _, logits = explainer(x, training=False)
        c = int(np.argmax(logits[0])) if target is None else int(np.asarray(target)[n])
        acts = stem[0].numpy()  # (rows, cols, channels)
        acts = np.moveaxis(acts, -1, 0)
        masks = minmax(upsample(acts, x.shape[1:3]))  # (channels, H, W)
        perturbed = x[0, None, :, :, 0] * masks + baseline * (1 - masks)
        probs = tf.nn.softmax(model(perturbed[..., None], training=False), axis=-1).numpy()[:, c]
        weights = np.exp(probs - probs.max())
        weights /= weights.sum()
        maps.append(np.maximum((weights[:, None, None] * acts).sum(axis=0), 0))
    return np.stack(maps)


def _as_input(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X[..., None] if X.ndim == 3 else X
