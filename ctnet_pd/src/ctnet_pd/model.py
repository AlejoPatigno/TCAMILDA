"""CTNet: convolutional stem -> token Transformer -> GAP + linear head (Section 2.6).

Layer names are part of the interface used by the explanation code:
``stem_out`` (C, 14x25x64), ``encoder_out`` (H^(L), 350x64) and ``logits`` (o).
The model outputs logits; probabilities are ``softmax(logits)``.
"""

from __future__ import annotations

import keras
from keras import layers

from .config import Config


@keras.saving.register_keras_serializable(package="ctnet_pd")
class PositionalEmbedding(layers.Layer):
    """Learned positional embedding added to the token sequence (Eq. 2)."""

    def build(self, input_shape):
        self.pos = self.add_weight(
            name="pos", shape=(input_shape[1], input_shape[2]),
            initializer=keras.initializers.RandomNormal(stddev=0.02), trainable=True,
        )

    def call(self, x):
        return x + self.pos


def build_ctnet(input_shape=(128, 229, 1), n_conv_blocks=2, conv_filters=64, kernel_size=3,
                pool_size=3, activation="gelu", positional_encoding="learned",
                n_transformer_layers=1, num_heads=2, key_dim=None, ffn_dim=128, dropout=0.1,
                head="gap", flatten_dense_units=128, n_classes=2, name="ctnet") -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="spec")
    x = inputs
    for b in range(n_conv_blocks):
        x = layers.Conv2D(conv_filters, kernel_size, padding="same", name=f"conv{b + 1}")(x)
        x = layers.BatchNormalization(name=f"bn{b + 1}")(x)
        x = layers.Activation(activation, name=f"act{b + 1}")(x)
        pool_name = "stem_out" if b == n_conv_blocks - 1 else f"pool{b + 1}"
        x = layers.AveragePooling2D(pool_size, name=pool_name)(x)

    rows, cols, d = x.shape[1], x.shape[2], x.shape[3]
    x = layers.Reshape((rows * cols, d), name="tokens")(x)
    if positional_encoding == "learned":
        x = PositionalEmbedding(name="pos_embed")(x)

    kd = key_dim or d // num_heads
    for l in range(n_transformer_layers):
        p = f"enc{l + 1}"
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=kd, dropout=dropout,
                                         name=f"{p}_mha")(x, x)
        attn = layers.Dropout(dropout, name=f"{p}_drop1")(attn)
        x = layers.LayerNormalization(name=f"{p}_ln1")(layers.Add(name=f"{p}_add1")([x, attn]))
        ff = layers.Dense(ffn_dim, activation=activation, name=f"{p}_ffn1")(x)
        ff = layers.Dense(d, name=f"{p}_ffn2")(ff)
        ff = layers.Dropout(dropout, name=f"{p}_drop2")(ff)
        x = layers.LayerNormalization(name=f"{p}_ln2")(layers.Add(name=f"{p}_add2")([x, ff]))
    x = layers.Identity(name="encoder_out")(x)

    if head == "gap":
        z = layers.GlobalAveragePooling1D(name="gap")(x)
    elif head == "flatten":
        z = layers.Flatten(name="flatten")(x)
        z = layers.Dense(flatten_dense_units, activation="relu", name="head_dense")(z)
        z = layers.Dropout(dropout, name="head_drop")(z)
    else:
        raise ValueError(head)
    logits = layers.Dense(n_classes, name="logits")(z)
    return keras.Model(inputs, logits, name=name)


# ImageNet backbones for the Experiment II CNN baselines, with the scaling that maps
# band-standardized log-Mel values (roughly in [-3, 3]) to each backbone's expected range.
_BACKBONES = {
    # keras.applications.ResNet50 expects caffe-style mean-subtracted inputs (about +-128)
    "resnet50": ("ResNet50", 64.0, 0.0),
    # EfficientNetB0 rescales [0, 255] internally
    "efficientnetb0": ("EfficientNetB0", 64.0, 128.0),
}


def build_pretrained_cnn(input_shape=(128, 229, 1), backbone="resnet50", weights="imagenet",
                         dropout=0.2, n_classes=2, name=None) -> keras.Model:
    """ImageNet CNN fine-tuned on log-Mel segments (Experiment II).

    The single log-Mel channel is replicated to three channels; the head is
    GAP -> dropout -> linear, as in CTNet, and the output layer is also named
    ``logits``.
    """
    if backbone not in _BACKBONES:
        raise ValueError(f"backbone must be one of {sorted(_BACKBONES)}")
    cls_name, scale, offset = _BACKBONES[backbone]
    inputs = keras.Input(shape=input_shape, name="spec")
    x = layers.Concatenate(name="to_rgb")([inputs, inputs, inputs])
    x = layers.Rescaling(scale, offset, name="to_backbone_range")(x)
    base = getattr(keras.applications, cls_name)(include_top=False, weights=weights,
                                                 input_shape=input_shape[:2] + (3,))
    x = base(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_drop")(x)
    logits = layers.Dense(n_classes, name="logits")(x)
    return keras.Model(inputs, logits, name=name or backbone)


ARCHITECTURES = ("ctnet",) + tuple(_BACKBONES)

# Hyperparameters that each architecture actually uses (the rest of the search space is skipped).
TUNABLE = {
    "ctnet": ("learning_rate", "dropout", "num_heads", "activation", "ffn_dim"),
    "resnet50": ("learning_rate", "dropout"),
    "efficientnetb0": ("learning_rate", "dropout"),
    "vit": ("learning_rate",),   # PyTorch models: see vision.py
    "swin": ("learning_rate",),
}


def build_model(cfg: Config, hp: dict | None = None) -> keras.Model:
    """Model registry: CTNet (and its ablations) or an ImageNet CNN baseline."""
    arch = cfg.model.architecture
    if arch in ("vit", "swin"):
        raise ValueError("ViT and Swin are PyTorch models; they are trained through vision.train")
    if arch == "ctnet":
        return build_ctnet(**model_kwargs(cfg, hp))
    return build_pretrained_cnn(
        input_shape=(cfg.spectrogram.n_mels, cfg.segmentation.frames, 1), backbone=arch,
        weights=cfg.model.pretrained_weights, dropout=(hp or {}).get("dropout", cfg.model.dropout),
        n_classes=cfg.model.n_classes,
    )


def model_kwargs(cfg: Config, hp: dict | None = None) -> dict:
    """Builder arguments from the config, with tuned hyperparameters taking precedence."""
    m = cfg.model
    kw = dict(
        input_shape=(cfg.spectrogram.n_mels, cfg.segmentation.frames, 1),
        n_conv_blocks=m.n_conv_blocks, conv_filters=m.conv_filters, kernel_size=m.kernel_size,
        pool_size=m.pool_size, activation=m.activation, positional_encoding=m.positional_encoding,
        n_transformer_layers=m.n_transformer_layers, num_heads=m.num_heads, key_dim=m.key_dim,
        ffn_dim=m.ffn_dim, dropout=m.dropout, head=m.head,
        flatten_dense_units=m.flatten_dense_units, n_classes=m.n_classes,
    )
    for key in ("activation", "num_heads", "ffn_dim", "dropout"):
        if hp and key in hp:
            kw[key] = hp[key]
    return kw


def explainer_model(model: keras.Model) -> keras.Model:
    """Same weights, three outputs: stem activations, encoder tokens and logits."""
    return keras.Model(
        model.inputs,
        [model.get_layer("stem_out").output, model.get_layer("encoder_out").output,
         model.get_layer("logits").output],
        name=f"{model.name}_explainer",
    )


def head_parameter_count(model: keras.Model) -> int:
    return int(sum(w.numpy().size for w in model.get_layer("logits").weights))
