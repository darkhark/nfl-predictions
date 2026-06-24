"""Phase 1 — first RFE/grid-search test of the Madden launch-rating features.

Reuses the in-repo CV RFE classifier, the random hyperparameter search, and the
bayes_logistic.evaluate hold-out metric suite, so Run 28 is directly comparable to the
BART/Bayesian runs. Rank-only pool + the 188 madden_* columns; seed-32 protocol."""
import json
import os
import numpy as np
import pandas as pd

from data_science_utilities.feature_groups import partition

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
