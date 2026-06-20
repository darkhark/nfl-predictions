"""Train + evaluate the Bayesian logistic regression against the BART champion.

Run from repo root:
    conda run --no-capture-output -n nfl-predictions \
        python scripts/experiments/bayes_logistic.py
Fast check:  ... python scripts/experiments/bayes_logistic.py --smoke --priors normal
"""
import argparse
import json
import os
import sys
import time

import numpy as np

# Repo root on sys.path so first-party namespace packages import when run by path.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_science_utilities.models.bayes_logistic import (  # noqa: E402
    data_prep, evaluate, model, posterior,
)

# Reference rows for the side-by-side table (README run log).
REFERENCE = {
    "BART Run-6 (same 54 feats)": {"auroc": 0.705, "brier": 0.2194},
    "BART Run-10 (champion)": {"auroc": 0.708, "brier": 0.2185},
    "XGBoost Run-11 (best)": {"auroc": 0.707, "brier": 0.2206},
}
DEFAULT_OUTDIR = os.path.join(REPO_ROOT, "data", "predict_games", "bayes_logistic")
TARGET_ACCEPT = {"normal": 0.9, "horseshoe": 0.95}


def run(prior, split, feature_names, results_dir, *, draws, tune, chains, seed):
    pre = data_prep.TrainFitPreprocessor(feature_names).fit(split.train)
    X_train = pre.transform(split.train)
    y_train = data_prep.get_target(split.train)
    X_holdout = pre.transform(split.holdout)
    y_holdout = data_prep.get_target(split.holdout)

    t0 = time.time()
    pm_model = model.build_model(X_train, y_train, prior=prior)
    idata = model.sample(
        pm_model, draws=draws, tune=tune, chains=chains, seed=seed,
        target_accept=TARGET_ACCEPT[prior],
    )
    mean_p, std_p = model.predict(pm_model, idata, X_holdout, seed=seed)
    elapsed = time.time() - t0

    metrics = evaluate.evaluate(
        y_holdout, mean_p, p_std=std_p, weeks=split.holdout["week"].to_numpy()
    )
    diag = model.diagnostics(idata)

    posterior.save_trace(idata, os.path.join(results_dir, f"posterior_{prior}.nc"))
    summary = posterior.coefficient_summary(idata, pre.feature_names_out_, pre, prior=prior)
    posterior.save_summary(summary, os.path.join(results_dir, f"coef_summary_{prior}.json"))

    row = dict(metrics)
    row["diagnostics"] = diag
    row["elapsed_sec"] = round(elapsed, 1)
    print(f"[{prior}] AUROC {row['auroc']:.4f} | Brier {row['brier']:.4f} | "
          f"acc {row['accuracy']:.4f} | rhat {diag['max_rhat']:.3f} | "
          f"div {diag['n_divergences']} | ess {diag['min_ess_bulk']:.0f} | "
          f"{row['elapsed_sec']}s | passed={diag['passed']}")
    return row


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=data_prep.RANDOM_SEED)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--priors", nargs="+", default=["normal", "horseshoe"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    feature_names = data_prep.load_feature_list()
    split = data_prep.load_and_split(seed=args.seed)

    draws, tune, chains = args.draws, args.tune, args.chains
    if args.smoke:
        draws, tune, chains = 60, 60, 2
        split = data_prep.Split(
            train=split.train.head(800), valid=split.valid, holdout=split.holdout.head(200),
        )

    variants = {}
    for prior in args.priors:
        variants[prior] = run(
            prior, split, feature_names, args.outdir,
            draws=draws, tune=tune, chains=chains, seed=args.seed,
        )

    results = {"variants": variants, "reference": REFERENCE,
               "n_features": len(feature_names), "smoke": args.smoke}
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Side-by-side table + a ready-to-paste README run-log line for the best variant.
    print("\n=== Side-by-side (holdout 2024+2025) ===")
    for name, m in {**{f"Bayes-{k}": v for k, v in variants.items()}, **REFERENCE}.items():
        print(f"  {name:32s} AUROC {m['auroc']:.4f}  Brier {m['brier']:.4f}")
    best = max(variants, key=lambda k: variants[k]["auroc"])
    b = variants[best]
    print(f"\nREADME Run-27 line (best = {best}):")
    print(f"| 27 | 2026-06-20 | **Bayesian logistic regression** ({best} prior), "
          f"frozen 54-feature Run-6 set | {len(feature_names)} | "
          f"{b['auroc']:.3f} | {b['accuracy']:.3f} | {b['brier']:.4f} |")
    return results


if __name__ == "__main__":
    main()
