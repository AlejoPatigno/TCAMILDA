"""End-to-end smoke test on synthetic audio laid out like the NeuroVoz corpus."""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from ctnet_pd.config import apply_overrides
from ctnet_pd.data import check_index_against_metadata, demographics_table, index_cohort, load_metadata, task_summary
from ctnet_pd.features import build_features
from ctnet_pd.metrics import aggregate, repeated_cv_summary
from ctnet_pd.pipeline import run_external, run_within_cohort


def _fake_corpus(root, n_per_group=6, seed=0):
    rng = np.random.default_rng(seed)
    sr = 22050
    (root / "audios").mkdir(parents=True)
    (root / "metadata").mkdir()
    meta = {"HC": [], "PD": []}
    sid = 1
    for group, f0 in (("HC", 120.0), ("PD", 180.0)):
        for _ in range(n_per_group):
            for task, dur in (("PATAKA", 3.0), ("A1", 1.0), ("DIABLO", 0.8)):
                t = np.arange(int(dur * sr)) / sr
                y = 0.3 * np.sin(2 * np.pi * (f0 + rng.normal(0, 5)) * t) + 0.01 * rng.standard_normal(t.size)
                sf.write(root / "audios" / f"{group}_{task}_{sid:04d}.wav", y.astype(np.float32), sr)
            meta[group].append({"ID": sid, "Age": int(rng.integers(50, 80)), "Sex": rng.choice(["M", "F"])})
            sid += 1
    for group, rows in meta.items():
        pd.DataFrame(rows).to_csv(root / "metadata" / f"metadata_{group.lower()}.csv", index=False)


@pytest.mark.parametrize("tuning", [False, True])
def test_pipeline_end_to_end(cfg, tmp_path, tuning):
    _fake_corpus(tmp_path / "corpus")
    c = apply_overrides(cfg, [
        f"cohorts.neurovoz.root={tmp_path / 'corpus'}", "cv.n_repeats=1", "cv.n_outer_folds=3",
        "training.max_epochs=2", f"tuning.enabled={str(tuning).lower()}", "tuning.max_trials=1",
        "tuning.max_epochs=1", "cv.inner_val_fraction=0.25",
    ])
    index = index_cohort(c, "neurovoz")
    assert len(index) == 36 and index["subject_id"].nunique() == 12
    summary = task_summary(index)
    assert set(summary["task_family"]) == {"ddk_pataka", "vowels", "repeat"}
    meta = load_metadata(c, "neurovoz")
    report = check_index_against_metadata(index, meta)
    assert report["audio_only"][0] == [] and report["metadata_only"][0] == []
    assert demographics_table(meta).shape[1] == 2

    X, seg = build_features(index, c, cache_path=tmp_path / "cache" / "nv")
    assert X.shape[1:] == (128, 229) and len(seg) == len(X)
    pred = run_within_cohort(c, X, seg, tmp_path / "out", folds=[0, 1, 2], save_models=True)
    assert pred["subject_id"].nunique() == 12  # every subject predicted once
    subj = aggregate(pred, task_balanced=True)
    per_rep, stats = repeated_cv_summary(subj)
    assert 0 <= stats.loc["auroc", "mean"] <= 1

    # resuming does not retrain finished folds
    again = run_within_cohort(c, X, seg, tmp_path / "out", folds=[0, 1, 2])
    assert len(again) == len(pred)

    if not tuning:
        res = run_external(c, X, seg, X, seg, tmp_path / "ext", source_oof=pred)
        assert "0.5" in res and "youden_source" in res

        from ctnet_pd.analysis import fold_explanations, fold_faithfulness, h1_test, load_fold
        model, stats_, test_subjects = load_fold(tmp_path / "out" / "ctnet", 0, 0)
        maps = fold_explanations(model, stats_, X, seg, test_subjects)
        assert maps["q_joint"].shape[1:] == (14, 25) and len(maps["seg"]) == len(maps["q_joint"])
        c2 = apply_overrides(c, ["xai.n_random_masks=2", "xai.q_step=0.1"])
        faith = fold_faithfulness(model, maps, c2, max_segments_per_subject=1)
        assert set(faith["subject_id"]) == set(test_subjects)
        assert {"gap", "gap_random", "delta_sal_0.10"} <= set(faith.columns)
        h1_test(faith)
