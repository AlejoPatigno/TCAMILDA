"""Model- and label-randomization sanity checks (Adebayo et al., 2018; Section 2.11.5)."""

from __future__ import annotations

import re

import keras
import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter


def cascade_order(model: keras.Model) -> list[list[str]]:
    """Top-down randomization stages: head -> Transformer layers (L..1) -> conv blocks (2..1)."""
    names = [l.name for l in model.layers]
    stages = [["logits"] + [n for n in ("head_dense",) if n in names]]
    enc = sorted({n.split("_")[0] for n in names if re.match(r"^enc\d+_", n)}, reverse=True)
    for e in enc:
        stages.append([n for n in names if n.startswith(f"{e}_")])
    if "pos_embed" in names:
        stages[-1].append("pos_embed")
    convs = sorted({n[-1] for n in names if n.startswith("conv")}, reverse=True)
    for c in convs:
        stages.append([f"conv{c}", f"bn{c}"])
    return stages


def randomize_layers(model: keras.Model, layer_names, rng: np.random.Generator) -> None:
    """Re-initialize the weights of the named layers in place.

    Kernels use the layer's own initializer when it exposes one; every other
    weight is redrawn from a normal distribution with the weight's current SD.
    """
    for name in layer_names:
        layer = model.get_layer(name)
        init = getattr(layer, "kernel_initializer", None)
        for w in layer.weights:
            value = w.numpy()
            if init is not None and "kernel" in w.name:
                new = keras.ops.convert_to_numpy(init(value.shape))
            else:
                new = rng.normal(0.0, value.std() + 1e-3, size=value.shape)
            w.assign(new.astype(value.dtype))


def cascading_randomization(model: keras.Model, explain_fn, X: np.ndarray, seed: int = 0) -> list[dict]:
    """Explanation similarity to the trained model after each cumulative stage.

    ``explain_fn(model, X)`` returns maps (n, rows, cols). The trained model is
    not modified (a copy is randomized).
    """
    rng = np.random.default_rng(seed)
    reference = explain_fn(model, X)
    work = keras.models.clone_model(model)
    work.set_weights(model.get_weights())
    results, done = [], []
    for stage in cascade_order(model):
        randomize_layers(work, stage, rng)
        done += stage
        maps = explain_fn(work, X)
        results.append({"randomized_up_to": stage[0], "layers": list(done),
                        **map_similarity(reference, maps)})
    return results


def map_similarity(A: np.ndarray, B: np.ndarray, topk_q: float = 0.10) -> dict:
    """Mean over samples of |Spearman|, SSIM and top-k IoU between two stacks of maps."""
    sp, ss, iou = [], [], []
    for a, b in zip(A, B):
        r = stats.spearmanr(a.ravel(), b.ravel()).statistic
        sp.append(abs(r) if np.isfinite(r) else 0.0)
        ss.append(ssim(a, b))
        iou.append(topk_iou(a, b, topk_q))
    return {"spearman_abs": float(np.mean(sp)), "ssim": float(np.mean(ss)), "topk_iou": float(np.mean(iou))}


def topk_iou(a: np.ndarray, b: np.ndarray, q: float) -> float:
    k = max(1, int(round(q * a.size)))
    sa = set(np.argsort(-a.ravel(), kind="stable")[:k])
    sb = set(np.argsort(-b.ravel(), kind="stable")[:k])
    return len(sa & sb) / len(sa | sb)


def ssim(a: np.ndarray, b: np.ndarray, win: int = 3) -> float:
    """SSIM with a uniform window on min-max scaled maps (data range 1)."""
    a = _scale(a)
    b = _scale(b)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = uniform_filter(a, win), uniform_filter(b, win)
    va = uniform_filter(a * a, win) - mu_a ** 2
    vb = uniform_filter(b * b, win) - mu_b ** 2
    cov = uniform_filter(a * b, win) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    return float(np.mean(num / den))


def _scale(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def permute_subject_labels(subject_labels, rng: np.random.Generator):
    """Label-randomization control: permute diagnoses across training subjects."""
    perm = subject_labels.copy()
    perm[:] = rng.permutation(subject_labels.to_numpy())
    return perm


def label_randomization_passes(sim_perm: np.ndarray, sim_seed: np.ndarray, n_boot: int = 2000,
                               level: float = 0.95, seed: int = 0) -> dict:
    """Pass if the upper CI bound of true-vs-permuted similarity < lower CI bound of seed-to-seed."""
    rng = np.random.default_rng(seed)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2

    def ci(v):
        boots = [np.mean(rng.choice(v, len(v))) for _ in range(n_boot)]
        return np.quantile(boots, [lo, hi])

    perm_ci, seed_ci = ci(np.asarray(sim_perm)), ci(np.asarray(sim_seed))
    return {"perm_ci": perm_ci.tolist(), "seed_ci": seed_ci.tolist(), "passes": bool(perm_ci[1] < seed_ci[0])}
