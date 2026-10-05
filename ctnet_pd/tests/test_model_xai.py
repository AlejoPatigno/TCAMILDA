import numpy as np
import tensorflow as tf

from ctnet_pd.faithfulness import faithfulness, make_baseline, perturbation_curves, q_grid
from ctnet_pd.grid import downsample, tile_edges, upsample
from ctnet_pd.model import build_ctnet, head_parameter_count
from ctnet_pd.sanity import cascade_order, cascading_randomization, map_similarity
from ctnet_pd.xai import explain, joint_map, relevance_maps, token_contributions


def test_architecture_shapes():
    m = build_ctnet()
    assert m.get_layer("stem_out").output.shape[1:] == (14, 25, 64)
    assert m.get_layer("encoder_out").output.shape[1:] == (350, 64)
    assert head_parameter_count(m) == 130
    assert build_ctnet(head="flatten").count_params() > 2_800_000


def test_token_contributions_are_exact():
    m = build_ctnet()
    X = np.random.randn(3, 128, 229).astype(np.float32)
    target = np.array([1, 0, 1])
    phi, od = token_contributions(m, X, target)
    w, b = m.get_layer("logits").get_weights()
    b_delta = np.where(target == 1, b[1] - b[0], b[0] - b[1])
    np.testing.assert_allclose(phi.sum(axis=1), od - b_delta, rtol=1e-4, atol=1e-4)


def test_maps_are_distributions():
    m = build_ctnet()
    X = np.random.randn(4, 128, 229).astype(np.float32)
    maps = explain(m, X)
    for key in ("q_pre", "q_post", "q_joint"):
        assert maps[key].shape == (4, 14, 25)
        np.testing.assert_allclose(maps[key].sum(axis=(1, 2)), 1, rtol=1e-4)
    np.testing.assert_allclose(joint_map(maps["pre"], maps["post"]), maps["q_joint"], rtol=1e-5)


def test_tiles_partition_and_roundtrip():
    e = tile_edges(128, 14)
    assert e[0] == 0 and e[-1] == 128 and np.all(np.diff(e) > 0)
    low = np.random.rand(14, 25).astype(np.float32)
    np.testing.assert_allclose(downsample(upsample(low, (128, 229)), (14, 25)), low, rtol=1e-6)


def test_faithfulness_detects_relevant_region():
    """Toy classifier whose output depends only on the first 3 token columns."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((128, 229)).astype(np.float32)
    x[:, :27] += 3.0
    prob_fn = lambda b: 1 / (1 + np.exp(-(b[:, :, :27].mean(axis=(1, 2)))))
    good = np.zeros((14, 25)); good[:, :3] = 1.0
    bad = np.zeros((14, 25)); bad[:, -3:] = 1.0
    qs = q_grid(0.02)
    B = make_baseline(x.shape, "mean")
    g_good = perturbation_curves(prob_fn, x, good + 1e-3 * rng.random((14, 25)), B, qs)
    g_bad = perturbation_curves(prob_fn, x, bad + 1e-3 * rng.random((14, 25)), B, qs)
    assert (g_good["auc_ins"] - g_good["auc_del"]) > (g_bad["auc_ins"] - g_bad["auc_del"])
    res = faithfulness(prob_fn, x, good, B, qs, n_random=3, min_shift=6, rng=rng)
    assert res["gap"] > res["gap_random"]


def test_cascading_randomization_runs():
    m = build_ctnet()
    stages = cascade_order(m)
    assert stages[0][0] == "logits" and stages[-1] == ["conv1", "bn1"]
    assert [s[0] for s in stages] == ["logits", "enc1_mha", "conv2", "conv1"]
    X = np.random.randn(2, 128, 229).astype(np.float32)
    fn = lambda model, X: relevance_maps(model, X, target=np.ones(len(X), int))["post"]
    res = cascading_randomization(m, fn, X)
    assert len(res) == len(stages)
    sim = map_similarity(fn(m, X), fn(m, X))
    assert np.isclose(sim["spearman_abs"], 1.0) and np.isclose(sim["topk_iou"], 1.0)
