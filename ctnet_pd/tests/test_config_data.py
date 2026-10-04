import pytest

from ctnet_pd.config import apply_overrides, token_grid, validate_config
from ctnet_pd.data import assign_task_family


def test_default_grid_is_14x25(cfg):
    assert token_grid(cfg) == (14, 25)


def test_override_and_validation(cfg):
    c2 = apply_overrides(cfg, ["model.num_heads=4", "cv.n_repeats=1"])
    assert c2.model.num_heads == 4 and c2.cv.n_repeats == 1 and cfg.cv.n_repeats == 10
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["model.not_a_key=1"])
    bad = apply_overrides(cfg, ["model.num_heads=3"])
    with pytest.raises(ValueError):
        validate_config(bad)


@pytest.mark.parametrize("task,family", [("PATAKA", "ddk_pataka"), ("A1", "vowels"), ("U", "vowels"),
                                         ("PATATA_BLANDA", "repeat"), ("DIABLO", "repeat"), ("FREE", "monologue")])
def test_task_families(cfg, task, family):
    assert assign_task_family(task, cfg.task_families) == family
