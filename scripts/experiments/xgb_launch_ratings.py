"""Phase 1 — first RFE/grid-search test of the Madden launch-rating features.

Reuses the in-repo CV RFE classifier, the random hyperparameter search, and the
bayes_logistic.evaluate hold-out metric suite, so Run 28 is directly comparable to the
BART/Bayesian runs. Rank-only pool + the 188 madden_* columns; seed-32 protocol."""
import argparse
import json
import os
import numpy as np
import pandas as pd

from data_science_utilities.feature_groups import partition
from data_science_utilities.models.xgb.feature_selection.recursive.classifier_cross_validation import (
    ClassifierCrossValidationRecursiveFeatureSelection,
)
from data_science_utilities.models.xgb.hyperparameter_search.random_search import (
    OptimalXGBHyperparameterSearch,
)
from data_science_utilities.models.bayes_logistic.evaluate import evaluate

RANDOM_SEED = 32
TARGET = 'target_win'


def build_rfe_pool(candidate_features):
    """Rank-only candidate pool (partition.is_rank_only_kept) minus the target.
    Every madden_* column is rank-only-kept, so all 188 enter the pool."""
    return [f for f in candidate_features
            if f != TARGET and partition.is_rank_only_kept(f)]


def restrict_to_families(features, keep_families):
    """Keep only features whose content family is in keep_families (Run 29's
    ablation-informed smaller pool). Names that don't classify (e.g. the target) drop."""
    keep = set(keep_families)
    out = []
    for f in features:
        try:
            fam = partition.content_family(f)
        except ValueError:
            continue
        if fam in keep:
            out.append(f)
    return out


def drop_madden_diffs(features):
    """Drop the Madden momentum/diff columns, keeping only the talent-level _ovr (and
    matchup) features. The internal sub-ablation showed the diffs are redundant
    (negative leave-one-out), so Run 29 pairs the clean level signal only."""
    return [f for f in features if not ('madden' in f and '_diff_' in f)]


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


RANDOM_XGB_PARAMS = {
    'learning_rate': [.03, .05, .1, .15, .2], 'max_depth': [2, 3],
    'subsample': [.5, .7, .9], 'min_child_weight': [10, 20, 50, 100],
    'gamma': [.5, 1, 5, 10, 100], 'n_estimators': [10, 20, 30, 40, 50],
    'early_stopping_rounds': [10, 20, 50], 'importance_type': ['total_gain'],
    'eval_metric': ['auc'],
}
RANDOM_SEARCH_PARAMS = dict(n_iter=100, cv=5, n_jobs=-1, random_state=RANDOM_SEED,
                            scoring='neg_brier_score')
CHAMPIONS = {
    'xgb_run11': {'auroc': 0.707, 'brier': 0.2206},
    'bart_run6': {'auroc': 0.705, 'brier': 0.2194},
    'bart_run10': {'auroc': 0.708, 'brier': 0.2185},
}


def run_grid_and_eval(df, selected_features, *, search_params=RANDOM_XGB_PARAMS,
                      search_kwargs=RANDOM_SEARCH_PARAMS):
    """Seed-32 split (train<2022 / valid22-23 / holdout>=2024), random search on the
    selected set, hold-out metrics via bayes_logistic.evaluate. Returns
    (best_model, metrics, holdout_df)."""
    data = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    train = data[data['season'] < 2022]
    valid = data[(data['season'] >= 2022) & (data['season'] < 2024)]
    holdout = data[data['season'] >= 2024].copy()
    search = OptimalXGBHyperparameterSearch(
        train[selected_features], train[TARGET],
        valid[selected_features], valid[TARGET],
    ).search(search_params, **search_kwargs)
    best = search.best_estimator_
    preds = best.predict_proba(holdout[selected_features])[:, 1]
    metrics = evaluate(holdout[TARGET], preds, p_std=None, weeks=holdout['week'])
    return best, metrics, holdout


def build_results(metrics, selected_features, best_model, best_num_feats):
    mads = madden_columns(selected_features)
    importances = best_model.get_booster().get_score()  # {feat: total_gain}
    ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    rank_of = {f: i + 1 for i, (f, _) in enumerate(ranked)}
    deltas = {}
    for name, champ in CHAMPIONS.items():
        deltas[f'{name}_auroc_delta'] = round(metrics['auroc'] - champ['auroc'], 4)
        deltas[f'{name}_brier_delta'] = round(metrics['brier'] - champ['brier'], 4)
    return {
        'config': {'seed': RANDOM_SEED, 'selection_metric': 'brier',
                   'rank_only': True, 'best_num_feats': int(best_num_feats)},
        'n_selected': len(selected_features),
        'madden_selected': mads,
        'madden_importance_rank': {m: rank_of.get(m) for m in mads},
        'metrics': metrics,
        'champion_deltas': deltas,
    }


PARQUET = 'data/predict_games/input_data/schedule_and_weekly.parquet'
FEATURES_LIST = 'data/predict_games/model_features_in/xgb_features_list.csv'
RFE_OUT = 'data/predict_games/model_features_in/rfe_features_kfolds_brier_madden.csv'
RESULTS_DIR = 'data/predict_games/xgb_launch_ratings'
MODEL_OUT = 'models/best_random_xgb_model_launch_ratings.json'


def _tagged(path, tag):
    """Suffix a file path before its extension (or a dir path) with `tag`."""
    if not tag:
        return path
    root, ext = os.path.splitext(path)
    return f'{root}{tag}{ext}'


def main(stage='all', *, parquet=PARQUET, features_list=FEATURES_LIST, rfe_out=RFE_OUT,
         results_dir=RESULTS_DIR, model_out=MODEL_OUT, best_num_feats=None,
         families=None, tag='', madden_levels_only=False):
    rfe_out = _tagged(rfe_out, tag)
    model_out = _tagged(model_out, tag)
    results_dir = results_dir + tag
    df = pd.read_parquet(parquet)
    assert_madden_2025_coverage(df)
    candidates = list(pd.read_csv(features_list)['feature'])
    if families:
        candidates = restrict_to_families(candidates, families)
    if madden_levels_only:
        candidates = drop_madden_diffs(candidates)

    if stage in ('rfe', 'all'):
        best_num_feats, _ = run_rfe(df, candidates, rfe_out)
        print(f'1-SE selected feature count (brier): {best_num_feats}')
    if stage in ('grid', 'all'):
        if best_num_feats is None:
            raise SystemExit('stage=grid requires --best-num-feats from a prior rfe run')
        selected = selected_features_from_rfe_csv(rfe_out, best_num_feats)
        model, metrics, _ = run_grid_and_eval(df, selected)
        os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
        model.save_model(model_out)
        os.makedirs(results_dir, exist_ok=True)
        results = build_results(metrics, selected, model, best_num_feats)
        with open(os.path.join(results_dir, 'results.json'), 'w') as fh:
            json.dump(results, fh, indent=2)
            fh.write('\n')
        print(json.dumps(results['metrics'], indent=2))
        print('champion deltas:', results['champion_deltas'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['rfe', 'grid', 'all'], default='all')
    ap.add_argument('--best-num-feats', type=int, default=None)
    ap.add_argument('--families', type=lambda s: s.split(','), default=None,
                    help='comma-separated content families to restrict the RFE pool to')
    ap.add_argument('--tag', type=str, default='',
                    help='suffix for the rfe-csv / results-dir / model output paths')
    ap.add_argument('--madden-levels-only', action='store_true',
                    help='drop the 138 madden _ovr_diff_* columns, keeping only talent levels')
    args = ap.parse_args()
    main(stage=args.stage, best_num_feats=args.best_num_feats,
         families=args.families, tag=args.tag, madden_levels_only=args.madden_levels_only)
