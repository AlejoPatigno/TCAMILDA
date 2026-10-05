import numpy as np
import pytest

from ctnet_pd.config import apply_overrides
from ctnet_pd.model import build_model
from ctnet_pd.pipeline import VARIANTS, variant_config


@pytest.mark.parametrize("arch", ["resnet50", "efficientnetb0"])
def test_pretrained_cnn_shapes(cfg, arch):
    c = apply_overrides(cfg, [f"model.architecture={arch}", "model.pretrained_weights=null"])
    m = build_model(c)
    assert m.output_shape == (None, 2) and m.get_layer("logits") is not None
    out = m(np.zeros((2, 128, 229, 1), np.float32))
    assert out.shape == (2, 2)


def test_variant_registry(cfg):
    assert set(VARIANTS) == {"ctnet", "cnn_only", "ctnet_flatten", "resnet50", "efficientnetb0"}
    assert variant_config(cfg, "cnn_only").model.n_transformer_layers == 0
    with pytest.raises(KeyError):
        variant_config(cfg, "vit")
    bad = apply_overrides(cfg, ["prior.enabled=true"])
    with pytest.raises(ValueError):
        variant_config(bad, "resnet50")
