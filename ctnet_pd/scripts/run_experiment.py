"""Command-line entry point for the experiments.

Examples
--------
python scripts/run_experiment.py within --cohort neurovoz --out outputs
python scripts/run_experiment.py within --cohort neurovoz --task-family ddk_pataka --set cv.n_repeats=1
python scripts/run_experiment.py within --cohort neurovoz --variant resnet50
python scripts/run_experiment.py baseline --cohort neurovoz --kind praat --task-family ddk_pataka
python scripts/run_experiment.py external --source pcgita --target neurovoz --task-family monologue

Spectrogram variants (--variant): ctnet, cnn_only, ctnet_flatten, resnet50, efficientnetb0.
Other baselines (--kind): egemaps, praat (SVM) and wavlm, xlsr, ast (frozen-encoder probe);
they reuse outputs/<cohort>/<family>/ctnet/folds.csv, so run CTNet first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctnet_pd.config import load_config  # noqa: E402
from ctnet_pd.data import index_cohort  # noqa: E402
from ctnet_pd.features import build_features  # noqa: E402
from ctnet_pd.metrics import aggregate, repeated_cv_summary  # noqa: E402
from ctnet_pd.pipeline import run_external, run_within_cohort, variant_config  # noqa: E402


def load_cohort(cfg, cohort, out, family):
    index = index_cohort(cfg, cohort)
    X, seg = build_features(index, cfg, cache_path=Path(out) / "cache" / f"{cohort}_features")
    if family:
        keep = (seg["task_family"] == family).to_numpy()
        X, seg = X[keep], seg[keep].reset_index(drop=True)
    return X, seg


def run_baseline(cfg, args, suffix):
    import pandas as pd

    from ctnet_pd import baselines

    index = index_cohort(cfg, args.cohort)
    out = Path(args.out) / args.cohort / suffix
    assign = pd.read_csv(out / "ctnet" / "folds.csv")
    idx_f = (index if args.task_family is None else index[index.task_family == args.task_family]).reset_index(drop=True)
    cache = Path(args.out) / "cache" / f"{args.cohort}_{args.kind}"
    if args.kind in ("egemaps", "praat"):
        extract = baselines.egemaps_features if args.kind == "egemaps" else baselines.praat_features
        table = extract(index, cfg, cache_path=cache)
        variant = f"{args.kind}_svm"
        pred = baselines.run_feature_baseline(baselines.feature_matrix(table, idx_f), idx_f, assign, cfg, "svm",
                                              variant=variant)
    else:
        baselines.pretrained_embeddings(index, cfg, args.kind, cache_path=cache)
        E = baselines.pretrained_embeddings(idx_f, cfg, args.kind, cache_path=cache)
        variant = f"{args.kind}_probe"
        pred = baselines.run_embedding_probe(E, idx_f, assign, cfg, variant=variant)
    (out / variant).mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / variant / "predictions.csv", index=False)
    per_rep, summary = repeated_cv_summary(aggregate(pred, task_balanced=args.task_family is None),
                                           cfg.evaluation.threshold)
    print(summary)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["within", "external", "baseline"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--cohort", default="neurovoz")
    ap.add_argument("--source")
    ap.add_argument("--target")
    ap.add_argument("--task-family", default=None, help="restrict to one task family (Table 2)")
    ap.add_argument("--variant", default="ctnet")
    ap.add_argument("--kind", choices=["egemaps", "praat", "wavlm", "xlsr", "ast"], default="praat")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--set", action="append", default=[], help="config override section.key=value")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    suffix = args.task_family or "all_tasks"
    if args.experiment == "baseline":
        run_baseline(cfg, args, suffix)
        return
    if args.experiment == "within":
        X, seg = load_cohort(cfg, args.cohort, args.out, args.task_family)
        pred = run_within_cohort(variant_config(cfg, args.variant), X, seg, Path(args.out) / args.cohort / suffix,
                                 variant=args.variant, save_models=args.variant == "ctnet")
        per_rep, summary = repeated_cv_summary(aggregate(pred, task_balanced=args.task_family is None),
                                               cfg.evaluation.threshold)
        print(summary)
    else:
        Xs, ss = load_cohort(cfg, args.source, args.out, args.task_family)
        Xt, st = load_cohort(cfg, args.target, args.out, args.task_family)
        res = run_external(variant_config(cfg, args.variant), Xs, ss, Xt, st,
                           Path(args.out) / f"{args.source}_to_{args.target}" / suffix, variant=args.variant)
        print(res)


if __name__ == "__main__":
    main()
