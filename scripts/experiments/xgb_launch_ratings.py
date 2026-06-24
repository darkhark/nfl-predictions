"""Phase 1 — first RFE/grid-search test of the Madden launch-rating features.

Reuses the in-repo CV RFE classifier, the random hyperparameter search, and the
bayes_logistic.evaluate hold-out metric suite, so Run 28 is directly comparable to the
BART/Bayesian runs. Rank-only pool + the 188 madden_* columns; seed-32 protocol."""
import json
import os
import numpy as np
import pandas as pd

from data_science_utilities.feature_groups import partition
from data_science_utilities.models.xgb.feature_selection.recursive.classifier_cross_validation import (
    ClassifierCrossValidationRecursiveFeatureSelection,
)

RANDOM_SEED = 32
TARGET = 'target_win'


def build_rfe_pool(candidate_features):
    """Rank-only candidate pool (partition.is_rank_only_kept) minus the target.
    Every madden_* column is rank-only-kept, so all 188 enter the pool."""
    return [f for f in candidate_features
            if f != TARGET and partition.is_rank_only_kept(f)]


def madden_columns(features):
    return [f for f in features if 'madden' in f]


def madden_ovr_columns(df):
    return [c for c in df.columns if 'madden' in c and c.endswith('_ovr')]


def season_2025_ovr_coverage(df):
    ovr = madden_ovr_columns(df)
    rows = df[df['season'] == 2025]
    if not ovr or rows.empty:
        return 0.0
    return float(rows[ovr].notna().to_numpy().mean())


def assert_madden_2025_coverage(df, min_cov=0.30):
    cov = season_2025_ovr_coverage(df)
    if cov < min_cov:
        raise RuntimeError(
            f'2025 madden _ovr coverage {cov:.3f} < {min_cov}: rebuild the parquet with '
            f'MADDEN_TOOLS_CDN_BASE set (see the plan precondition).')


def selected_features_from_rfe_csv(path, best_num_feats):
    fdf = pd.read_csv(path, index_col=0)
    return list(fdf.loc[best_num_feats, :].dropna().values)


RFE_XGB_PARAMS = dict(
    n_estimators=5000, n_jobs=-1, learning_rate=.15, early_stopping_rounds=10,
    max_depth=5, eval_metric='auc', importance_type='total_gain', random_state=RANDOM_SEED,
)
_RFE_PROGRESS_LOG = '/tmp/xgb_launch_rfe_progress.log'


def _log_rfe_progress(row):
    with open(_RFE_PROGRESS_LOG, 'a') as fh:
        fh.write(f"iter {row['iteration']}/{row['max_iter']}: "
                 f"{row['num_features']} features, cv {row['score']:.4f}\n")


def run_rfe(df, candidate_features, out_csv, *, rfe_params=RFE_XGB_PARAMS,
            max_iter=60, min_features=5, n_folds=5):
    """CV RFE on season<2024 over the rank-only pool; writes the features-by-count CSV
    and returns (best_num_feats via 1-SE, the fitted rfe object)."""
    pool = build_rfe_pool(candidate_features)
    inputs = (df[pool + [TARGET, 'season']]
              .sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True))
    train = inputs[inputs['season'] < 2024]
    rfe = ClassifierCrossValidationRecursiveFeatureSelection(
        train[pool], train[TARGET], rfe_params, model_score_metric='brier')
    rfe.get_optimal_features_no_grouped_records(
        max_iter=max_iter, min_features=min_features, n_folds=n_folds,
        verbose=1, on_iteration=_log_rfe_progress)
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    rfe.get_features_in_dataframe().to_csv(out_csv)
    return rfe.get_best_num_features_1se(), rfe
