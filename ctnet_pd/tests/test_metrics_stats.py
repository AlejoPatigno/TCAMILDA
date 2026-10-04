import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ctnet_pd.metrics import aggregate, binary_metrics, bootstrap_ci, calibration, repeated_cv_summary
from ctnet_pd.stats import delong_test, holm
from ctnet_pd.training import sample_weights


def test_aggregate_task_balanced(toy_segments):
    pred = toy_segments.assign(prob=np.where(toy_segments["task_family"] == "ddk", 0.9, 0.1))
    # give subject S00 an extra ddk recording: plain mean shifts, task-balanced does not
    extra = pred[(pred.subject_id == "S00") & (pred.task_family == "ddk")].assign(recording_id=999)
    pred = pd.concat([pred, extra])
    plain = aggregate(pred).set_index("subject_id")["prob"]
    balanced = aggregate(pred, task_balanced=True).set_index("subject_id")["prob"]
    assert np.isclose(balanced["S00"], 0.5) and plain["S00"] > 0.5


def test_delong_matches_sklearn_and_holm():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p1, p2 = y + rng.normal(0, 1, 200), y + rng.normal(0, 2, 200)
    res = delong_test(y, p1, p2)
    assert np.isclose(res["auc1"], roc_auc_score(y, p1)) and np.isclose(res["auc2"], roc_auc_score(y, p2))
    assert 0 <= res["p"] <= 1
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_calibration_of_calibrated_predictions():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    intercept, slope = calibration(y, p)
    assert abs(intercept) < 0.05 and abs(slope - 1) < 0.05


def test_metrics_and_bootstrap():
    rng = np.random.default_rng(2)
    rows = []
    for rep in range(3):
        for s in range(40):
            y = int(s < 20)
            rows.append({"repeat": rep, "subject_id": f"S{s}", "label": y,
                         "prob": float(np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2), 0, 1))})
    subj = pd.DataFrame(rows)
    per_rep, summary = repeated_cv_summary(subj)
    assert len(per_rep) == 3 and summary.loc["auroc", "mean"] > 0.8
    point, lo, hi = bootstrap_ci(subj, "auroc", n_boot=200)
    assert lo <= point <= hi
    m = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.4, 0.9]))
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.5


def test_subject_balanced_weights(toy_segments):
    seg = pd.concat([toy_segments, toy_segments[toy_segments.subject_id == "S00"]])  # S00 doubled
    w = sample_weights(seg)
    per_subject = pd.Series(w, index=seg["subject_id"].values).groupby(level=0).sum()
    assert np.allclose(per_subject, per_subject.iloc[0])
