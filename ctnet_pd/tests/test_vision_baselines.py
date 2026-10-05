"""ViT / Swin baselines: tiny random models, trained in the isolated PyTorch process."""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from ctnet_pd.config import apply_overrides
from ctnet_pd.isolation import run_isolated
from ctnet_pd.metrics import aggregate
from ctnet_pd.pipeline import run_external, run_within_cohort, variant_config
from tiny_models import save_tiny_image_classifier

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None,
    reason="torch and transformers are optional")


@pytest.fixture
def toy_data(toy_segments):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((len(toy_segments), 128, 229)).astype(np.float32)
    X += 0.8 * toy_segments["label"].to_numpy()[:, None, None]
    return X, toy_segments


@pytest.mark.parametrize("arch", ["vit", "swin"])
def test_vision_baseline_within_and_external(cfg, tmp_path, toy_data, arch):
    X, seg = toy_data
    model_dir = tmp_path / f"tiny_{arch}"
    run_isolated(save_tiny_image_classifier, str(model_dir), arch)
    c = apply_overrides(cfg, [
        f"vision_transformers.checkpoints.{arch}={model_dir}", "vision_transformers.image_size=32",
        "vision_transformers.max_epochs=2", "vision_transformers.learning_rates=[1.0e-3, 1.0e-4]",
        "cv.n_repeats=1", "cv.n_outer_folds=3", "cv.inner_val_fraction=0.25"])
    c = variant_config(c, arch)
    pred = run_within_cohort(c, X, seg, tmp_path / "out", variant=arch, save_models=True)
    assert pred["subject_id"].nunique() == seg["subject_id"].nunique()
    assert pred["prob"].between(0, 1).all()
    runs = [json.loads(l) for l in (tmp_path / "out" / arch / "runs.jsonl").read_text().splitlines()]
    assert len(runs) == 3 and all(r["hp"]["learning_rate"] in (1e-3, 1e-4) for r in runs)
    assert len(list((tmp_path / "out" / arch).glob("model_r0_f*.pt"))) == 3
    assert not any((tmp_path / "out" / arch / "tuning").iterdir())  # temporary weights removed
    assert aggregate(pred)["subject_id"].nunique() == 30

    res = run_external(c, X, seg, X[:30], seg.iloc[:30].reset_index(drop=True), tmp_path / "ext", variant=arch)
    assert res["hp"]["learning_rate"] in (1e-3, 1e-4) and 1 <= res["epochs"] <= 2
    assert (tmp_path / "ext" / arch / "model_external.pt").exists()
