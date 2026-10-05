"""Configuration loading, overrides and consistency checks."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class Config(dict):
    """A dict with attribute access for nested sections (cfg.model.num_heads)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(name) from err

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __deepcopy__(self, memo):
        return Config(copy.deepcopy(dict(self), memo))

    def to_dict(self) -> dict:
        return _to_plain(self)


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    return obj


def apply_overrides(cfg: Config, overrides: Iterable[str]) -> Config:
    """Apply "section.key=value" overrides; values are parsed as YAML scalars."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        *parents, leaf = key.strip().split(".")
        node = cfg
        for part in parents:
            if part not in node:
                raise KeyError(f"Unknown config section {part!r} in {item!r}")
            node = node[part]
        if leaf not in node:
            raise KeyError(f"Unknown config key {leaf!r} in {item!r}")
        node[leaf] = _wrap(yaml.safe_load(raw))
    return cfg


def load_config(path: str | Path | None = None, overrides: Iterable[str] | None = None) -> Config:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        cfg = _wrap(yaml.safe_load(fh))
    cfg = apply_overrides(cfg, overrides or [])
    validate_config(cfg)
    return cfg


def token_grid(cfg: Config) -> tuple[int, int]:
    """Token grid (rows = Mel, cols = time) produced by the convolutional stem."""
    n_mels = cfg.spectrogram.n_mels
    frames = cfg.segmentation.frames
    for _ in range(cfg.model.n_conv_blocks):
        n_mels //= cfg.model.pool_size
        frames //= cfg.model.pool_size
    return n_mels, frames


def validate_config(cfg: Config) -> None:
    rows, cols = token_grid(cfg)
    if rows < 1 or cols < 1:
        raise ValueError(f"Convolutional stem collapses the input to {rows}x{cols}")
    d = cfg.model.conv_filters
    heads = [cfg.model.num_heads] + list(cfg.tuning.search_space.get("num_heads", []))
    if cfg.model.key_dim is None:
        bad = [h for h in heads if d % h]
        if bad:
            raise ValueError(f"num_heads {bad} do not divide d={d}; set model.key_dim explicitly")
    if cfg.model.architecture not in ("ctnet", "resnet50", "efficientnetb0", "vit", "swin"):
        raise ValueError("model.architecture must be ctnet, resnet50, efficientnetb0, vit or swin")
    if cfg.prior.enabled and cfg.model.architecture != "ctnet":
        raise ValueError("The acoustic prior is defined for CTNet only")
    if cfg.model.head not in ("gap", "flatten"):
        raise ValueError("model.head must be 'gap' or 'flatten'")
    if cfg.prior.map not in ("pre", "post", "joint"):
        raise ValueError("prior.map must be 'pre', 'post' or 'joint'")
    if cfg.cv.inner not in ("holdout", "kfold"):
        raise ValueError("cv.inner must be 'holdout' or 'kfold'")
    if not 0 < cfg.segmentation.min_fill <= 1:
        raise ValueError("segmentation.min_fill must be in (0, 1]")
