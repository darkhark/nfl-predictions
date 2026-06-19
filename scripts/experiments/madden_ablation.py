"""Targeted base-vs-madden_ratings ablation.

Compares two configurations per cross-validation fold:
  A) All groups (context_rest + all toggle groups)
  B) All groups MINUS madden_ratings

Reports Brier and ROC-AUC hold-out metrics for each. Writes results to
data/predict_games/group_ablation/madden_ablation.json.

Usage:
    conda run -n nfl-predictions python scripts/experiments/madden_ablation.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, REPO_ROOT)

from data_science_utilities.feature_groups.partition import partition_features, TARGET
from data_science_utilities.feature_groups.scorers import (
    season_rolling_origin_folds, make_xgb_fold_scorer)

RANDOM_SEED = 32
FIRST_TEST_SEASON = int(os.environ.get('FIRST_TEST_SEASON', '2019'))
LAST_TEST_SEASON = int(os.environ.get('LAST_TEST_SEASON', '2025'))
VALID_YEARS = 2
RANK_ONLY = True

PARQUET = os.path.join(REPO_ROOT, 'data', 'predict_games', 'input_data',
                       'schedule_and_weekly.parquet')
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'predict_games', 'group_ablation')
os.makedirs(OUT_DIR, exist_ok=True)
RESULT_JSON = os.path.join(OUT_DIR, 'madden_ablation.json')


def make_roc_scorer(train_df, valid_df, test_df):
    from data_science_utilities.feature_groups.scorers import make_xgb_fold_scorer
    return make_xgb_fold_scorer(train_df, valid_df, test_df, target=TARGET, metric='roc_auc')


def make_brier_scorer(train_df, valid_df, test_df):
    from data_science_utilities.feature_groups.scorers import make_xgb_fold_scorer
    return make_xgb_fold_scorer(train_df, valid_df, test_df, target=TARGET, metric='brier')


def main():
    t0 = time.time()
    np.random.seed(RANDOM_SEED)
    print(f'Loading {PARQUET}')
    df = (pd.read_parquet(PARQUET)
          .sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True))

    groups = partition_features(list(df.columns), rank_only=RANK_ONLY)
    base_features = groups['context_rest']
    toggle_groups = {n: c for n, c in groups.items()
                     if n != 'context_rest' and len(c) > 0}

    all_features = base_features + [f for cols in toggle_groups.values() for f in cols]
    no_madden = base_features + [f for name, cols in toggle_groups.items()
                                  if name != 'madden_ratings'
                                  for f in cols]

    print(f'All features: {len(all_features)}')
    print(f'Without madden_ratings: {len(no_madden)}')
    madden_features = toggle_groups.get('madden_ratings', [])
    print(f'madden_ratings group: {len(madden_features)} features')

    folds = season_rolling_origin_folds(
        df, first_test_season=FIRST_TEST_SEASON, last_test_season=LAST_TEST_SEASON,
        valid_years=VALID_YEARS)
    print(f'{len(folds)} folds: test seasons {[f["test_season"] for f in folds]}')

    results = []
    for i, fold in enumerate(folds):
        train_df = df.loc[fold['train']]
        valid_df = df.loc[fold['valid']]
        test_df = df.loc[fold['test']]

        brier_scorer = make_brier_scorer(train_df, valid_df, test_df)
        roc_scorer = make_roc_scorer(train_df, valid_df, test_df)

        brier_all = brier_scorer(all_features)
        brier_no_madden = brier_scorer(no_madden)
        roc_all = roc_scorer(all_features)
        roc_no_madden = roc_scorer(no_madden)

        row = {
            'fold': i + 1,
            'test_season': fold['test_season'],
            'n_train': len(fold['train']),
            'n_test': len(fold['test']),
            'brier_all': round(brier_all, 6),
            'brier_no_madden': round(brier_no_madden, 6),
            'brier_delta': round(brier_all - brier_no_madden, 6),
            'roc_auc_all': round(roc_all, 6),
            'roc_auc_no_madden': round(roc_no_madden, 6),
            'roc_auc_delta': round(roc_all - roc_no_madden, 6),
        }
        results.append(row)
        print(f"  fold {i+1}/{len(folds)} s{fold['test_season']}: "
              f"brier all={brier_all:.4f} no-madden={brier_no_madden:.4f} "
              f"delta={row['brier_delta']:+.4f} | "
              f"AUROC all={roc_all:.4f} no-madden={roc_no_madden:.4f} "
              f"delta={row['roc_auc_delta']:+.4f}")

    # Aggregate
    brier_delta_mean = np.mean([r['brier_delta'] for r in results])
    roc_delta_mean = np.mean([r['roc_auc_delta'] for r in results])
    brier_all_mean = np.mean([r['brier_all'] for r in results])
    brier_no_madden_mean = np.mean([r['brier_no_madden'] for r in results])
    roc_all_mean = np.mean([r['roc_auc_all'] for r in results])
    roc_no_madden_mean = np.mean([r['roc_auc_no_madden'] for r in results])

    out = {
        'config': {
            'rank_only': RANK_ONLY, 'random_seed': RANDOM_SEED,
            'first_test_season': FIRST_TEST_SEASON, 'last_test_season': LAST_TEST_SEASON,
            'all_features': len(all_features), 'no_madden_features': len(no_madden),
            'madden_ratings_features': len(madden_features),
        },
        'folds': results,
        'summary': {
            'brier_all_mean': round(brier_all_mean, 6),
            'brier_no_madden_mean': round(brier_no_madden_mean, 6),
            'brier_delta_mean': round(brier_delta_mean, 6),
            'roc_auc_all_mean': round(roc_all_mean, 6),
            'roc_auc_no_madden_mean': round(roc_no_madden_mean, 6),
            'roc_auc_delta_mean': round(roc_delta_mean, 6),
        },
        'elapsed_s': round(time.time() - t0, 1),
    }
    json.dump(out, open(RESULT_JSON, 'w'), indent=2)
    print(f'\nWrote {RESULT_JSON}')
    print(f'\n=== SUMMARY ===')
    print(f'Brier:   all={brier_all_mean:.4f}  no-madden={brier_no_madden_mean:.4f}  '
          f'delta={brier_delta_mean:+.4f}  '
          f'(negative=madden hurts, positive=madden helps [lower Brier is better])')
    print(f'ROC-AUC: all={roc_all_mean:.4f}  no-madden={roc_no_madden_mean:.4f}  '
          f'delta={roc_delta_mean:+.4f}  '
          f'(positive=madden helps, negative=madden hurts [higher AUROC is better])')
    print(f'Elapsed: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
