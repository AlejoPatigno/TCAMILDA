"""Training: subject-balanced loss, callbacks, hyperparameter search and the prior trainer.

Fixes with respect to the preliminary notebook:
- validation data come from an inner, subject-disjoint split, never from the test fold;
- ``ReduceLROnPlateau`` had ``min_lr = 0.001`` (= the default Adam rate), so the
  rate never decreased; ``min_lr`` is now 1e-6;
- the learning rate is actually searched (it was listed but never used);
- the checkpoint is selected on the inner-validation loss, not on test accuracy.
"""

from __future__ import annotations

import gc

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from .config import Config
from .model import TUNABLE, build_model, explainer_model
from .vision import TorchModelHandle
from .vision import predict_proba as _torch_predict
from .xai import logit_contrast


# ------------------------------------------------------------------ sample weights

def sample_weights(seg: pd.DataFrame, subject_balanced: bool = True, class_balanced: bool = True) -> np.ndarray:
    """Weights implementing the subject-balanced loss of Eq. (10).

    Each subject's segments share a total weight of 1 (so subjects with many
    recordings do not dominate); optionally each class gets equal total weight.
    Weights are rescaled to mean 1.
    """
    w = np.ones(len(seg), dtype=np.float64)
    if subject_balanced:
        counts = seg.groupby("subject_id")["subject_id"].transform("size").to_numpy()
        w /= counts
    if class_balanced:
        subj = seg.drop_duplicates("subject_id")
        n_class = subj["label"].value_counts()
        w *= seg["label"].map(lambda c: len(subj) / (2 * n_class[c])).to_numpy()
    return (w / w.mean()).astype(np.float32)


# ---------------------------------------------------------------------- datasets

def make_dataset(X, y, w=None, prior=None, batch_size=32, shuffle=False, seed=0) -> tf.data.Dataset:
    """tf.data pipeline from in-memory arrays; X is (n, H, W) and is given a channel axis."""
    X = np.asarray(X, dtype=np.float32)[..., None]
    y = np.asarray(y, dtype=np.int32)
    feats = X if prior is None else (X, np.asarray(prior, dtype=np.float32))
    items = (feats, y) if w is None else (feats, y, np.asarray(w, dtype=np.float32))
    ds = tf.data.Dataset.from_tensor_slices(items)
    if shuffle:
        ds = ds.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------- training

def compile_model(model: keras.Model, learning_rate: float) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model


def callbacks(cfg: Config, early_stopping: bool = True) -> list:
    t = cfg.training
    cbs = [keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=t.reduce_lr_factor,
                                             patience=t.reduce_lr_patience, min_lr=t.min_lr)]
    if early_stopping:
        cbs.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=t.early_stopping_patience,
                                                 restore_best_weights=True))
    return cbs


def build_and_compile(cfg: Config, hp: dict | None = None, seed: int | None = None) -> keras.Model:
    if seed is not None:
        keras.utils.set_random_seed(seed)
    hp = hp or {}
    model = build_model(cfg, hp)
    lr = hp.get("learning_rate", cfg.training.learning_rate)
    if cfg.prior.enabled:
        trainer = PriorRegularizedTrainer(model, lambda_prior=hp.get("lambda_prior", cfg.prior.lambda_prior),
                                          map_mode=cfg.prior.map)
        trainer.compile(optimizer=keras.optimizers.Adam(lr))
        return trainer
    return compile_model(model, lr)


def fit(model, X_tr, y_tr, w_tr, X_val, y_val, w_val, cfg: Config, epochs: int | None = None,
        prior_tr=None, prior_val=None, early_stopping: bool = True, seed: int = 0, verbose: int = 0):
    """Train with early stopping on the inner-validation (subject-balanced) loss.

    Returns the Keras history and the best epoch (1-based).
    """
    t = cfg.training
    ds_tr = make_dataset(X_tr, y_tr, w_tr, prior_tr, t.batch_size, shuffle=True, seed=seed)
    ds_val = None if X_val is None else make_dataset(X_val, y_val, w_val, prior_val, t.batch_size)
    hist = model.fit(ds_tr, validation_data=ds_val, epochs=epochs or t.max_epochs,
                     callbacks=callbacks(cfg, early_stopping and ds_val is not None) if ds_val is not None else [],
                     verbose=verbose)
    val_loss = hist.history.get("val_loss")
    best_epoch = int(np.argmin(val_loss)) + 1 if val_loss else len(hist.history["loss"])
    return hist, best_epoch


def predict_proba(model, X, batch_size: int = 128) -> np.ndarray:
    """P(PD) for each segment."""
    if isinstance(model, TorchModelHandle):
        return _torch_predict(model, X)
    base = model.base if isinstance(model, PriorRegularizedTrainer) else model
    out = []
    for start in range(0, len(X), batch_size):
        xb = np.asarray(X[start:start + batch_size], dtype=np.float32)[..., None]
        out.append(tf.nn.softmax(base(xb, training=False), axis=-1).numpy()[:, 1])
    return np.concatenate(out)


def clear_session():
    keras.backend.clear_session()
    gc.collect()


# --------------------------------------------------------- hyperparameter search

def tune(cfg: Config, X_tr, y_tr, w_tr, X_val, y_val, w_val, directory: str, project: str,
         seed: int = 0, prior_tr=None, prior_val=None) -> dict:
    """Bayesian optimization on the inner split only (KerasTuner). Returns the best values."""
    import keras_tuner as kt

    space = cfg.tuning.search_space

    tunable = TUNABLE[cfg.model.architecture]
    if cfg.model.architecture == "ctnet" and cfg.model.n_transformer_layers == 0:
        tunable = ("learning_rate", "activation")  # CNN-only ablation (no dropout layer)

    def build(hp):
        values = {key: hp.Choice(key, space[key]) for key in tunable}
        if cfg.prior.enabled:
            values["lambda_prior"] = hp.Choice("lambda_prior", space["lambda_prior"])
        return build_and_compile(cfg, values)

    tuner = kt.BayesianOptimization(build, objective=kt.Objective("val_loss", "min"),
                                    max_trials=cfg.tuning.max_trials, seed=seed, directory=directory,
                                    project_name=project, overwrite=True)
    t = cfg.training
    tuner.search(make_dataset(X_tr, y_tr, w_tr, prior_tr, t.batch_size, shuffle=True, seed=seed),
                 validation_data=make_dataset(X_val, y_val, w_val, prior_val, t.batch_size),
                 epochs=cfg.tuning.max_epochs, callbacks=callbacks(cfg), verbose=0)
    return dict(tuner.get_best_hyperparameters(1)[0].values)


# ------------------------------------------------------- prior-regularized trainer

def js_divergence(p: tf.Tensor, q: tf.Tensor, eps: float = 1e-8) -> tf.Tensor:
    """Jensen-Shannon divergence between distributions over the last two axes."""
    m = 0.5 * (p + q)
    kl = lambda a, b: tf.reduce_sum(a * (tf.math.log(a + eps) - tf.math.log(b + eps)), axis=(-2, -1))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _sum_norm(R: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    R = R + eps
    return R / tf.reduce_sum(R, axis=(-2, -1), keepdims=True)


class PriorRegularizedTrainer(keras.Model):
    """Wraps CTNet and adds lambda_P * L_prior (Eq. 8) to the subject-balanced CE.

    The prior is applied to PD samples only, explaining the PD logit contrast.
    ``map_mode`` selects q_pre (Grad-CAM, second-order gradients), q_post (exact
    token contribution, first order) or q_J. Batches are ((X, prior), y, w)
    with ``prior`` = p (already sum-normalized, grid shape).
    """

    def __init__(self, base: keras.Model, lambda_prior: float = 0.1, map_mode: str = "joint", **kwargs):
        super().__init__(**kwargs)
        self.base = base
        self.explainer = explainer_model(base)
        self.lambda_prior = float(lambda_prior)
        self.map_mode = map_mode
        self.ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.cls_tracker = keras.metrics.Mean(name="cls_loss")
        self.prior_tracker = keras.metrics.Mean(name="prior_loss")

    @property
    def metrics(self):
        return [self.loss_tracker, self.cls_tracker, self.prior_tracker]

    def call(self, x, training=False):
        if isinstance(x, (tuple, list)):
            x = x[0]
        return self.base(x, training=training)

    def _prior_losses(self, data, training):
        (x, prior), y, w = data
        y = tf.cast(y, tf.int32)
        w = tf.cast(w, tf.float32)
        target = tf.ones_like(y)  # explain the PD contrast
        with tf.GradientTape() as inner:
            stem, tokens, logits = self.explainer(x, training=training)
            od = logit_contrast(logits, target)
        g_stem, g_tok = inner.gradient(od, [stem, tokens])
        cls = tf.reduce_sum(w * self.ce(y, logits)) / tf.reduce_sum(w)
        rows, cols = stem.shape[1], stem.shape[2]
        maps = []
        if self.map_mode in ("pre", "joint"):
            alpha = tf.reduce_mean(g_stem, axis=(1, 2), keepdims=True)
            maps.append(_sum_norm(tf.nn.relu(tf.reduce_sum(alpha * stem, axis=-1))))
        if self.map_mode in ("post", "joint"):
            phi = tf.reshape(tf.reduce_sum(g_tok * tokens, axis=-1), (-1, rows, cols))
            maps.append(_sum_norm(tf.nn.relu(phi)))
        q = tf.add_n(maps) / len(maps)
        pd_w = w * tf.cast(tf.equal(y, 1), tf.float32)
        js = js_divergence(q, prior)
        prior_loss = tf.math.divide_no_nan(tf.reduce_sum(pd_w * js), tf.reduce_sum(pd_w))
        return cls, prior_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            cls, prior_loss = self._prior_losses(data, training=True)
            loss = cls + self.lambda_prior * prior_loss
        variables = self.base.trainable_variables
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))
        return self._update(loss, cls, prior_loss)

    def test_step(self, data):
        cls, prior_loss = self._prior_losses(data, training=False)
        return self._update(cls + self.lambda_prior * prior_loss, cls, prior_loss)

    def _update(self, loss, cls, prior_loss):
        self.loss_tracker.update_state(loss)
        self.cls_tracker.update_state(cls)
        self.prior_tracker.update_state(prior_loss)
        return {m.name: m.result() for m in self.metrics}
