"""Headless runner for the feature-group ablation sweep.

Mirrors group_ablation.ipynb but is headless-friendly (the notebook buffers stdout
under nbconvert). Writes LIVE progress to PROGRESS_LOG (``tail -f`` it), saves the full
decomposition to RESULT_JSON, renders the interaction heatmap to HEATMAP_PNG, and prints
a summary. Config via env vars (SELECTION_METRIC, FIRST_TEST_SEASON, LAST_TEST_SEASON).

    conda run -n nfl-predictions python \
        notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/run_group_ablation.py
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_repo_root(start=None):
    path = os.path.abspath(start or os.path.dirname(__file__))
    while path != os.path.dirname(path):
        if os.path.isdir(os.path.join(path, 'data_science_utilities')):
            return path
        path = os.path.dirname(path)
    raise RuntimeError('repo root (with data_science_utilities/) not found')


REPO_ROOT = find_repo_root()
sys.path.insert(0, REPO_ROOT)

from data_science_utilities.feature_groups.partition import partition_features, TARGET
from data_science_utilities.feature_groups.ablation import cross_validated_group_ablation
from data_science_utilities.feature_groups.scorers import (
    season_rolling_origin_folds, make_xgb_fold_scorer)

RANDOM_SEED = 32
SELECTION_METRIC = os.environ.get('SELECTION_METRIC', 'brier')
HIGHER_IS_BETTER = SELECTION_METRIC == 'roc_auc'
RANK_ONLY = True
FIRST_TEST_SEASON = int(os.environ.get('FIRST_TEST_SEASON', '2019'))
LAST_TEST_SEASON = int(os.environ.get('LAST_TEST_SEASON', '2025'))
VALID_YEARS = 2

PARQUET = os.path.join(REPO_ROOT, 'data', 'predict_games', 'input_data',
                       'schedule_and_weekly.parquet')
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'predict_games', 'group_ablation')
os.makedirs(OUT_DIR, exist_ok=True)
PROGRESS_LOG = os.environ.get('PROGRESS_LOG', '/tmp/group_ablation_progress.log')
RESULT_JSON = os.path.join(OUT_DIR, f'group_ablation_{SELECTION_METRIC}.json')
HEATMAP_PNG = os.path.join(OUT_DIR, f'group_interaction_{SELECTION_METRIC}.png')


def log(msg):
    with open(PROGRESS_LOG, 'a') as fh:
        print(time.strftime('%H:%M:%S'), msg, file=fh)
    print(msg, flush=True)


def main():
    open(PROGRESS_LOG, 'w').close()
    np.random.seed(RANDOM_SEED)
    log(f'loading {PARQUET}')
    df = (pd.read_parquet(PARQUET)
          .sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True))

    groups = partition_features(list(df.columns), rank_only=RANK_ONLY)
    base_features = groups['context_rest']
    toggle_groups = {n: c for n, c in groups.items()
                     if n != 'context_rest' and len(c) > 0}
    folds = season_rolling_origin_folds(
        df, first_test_season=FIRST_TEST_SEASON, last_test_season=LAST_TEST_SEASON,
        valid_years=VALID_YEARS)
    n_subsets = 2 ** len(toggle_groups)
    log(f'{len(toggle_groups)} groups -> {n_subsets} subsets x {len(folds)} folds '
        f'= {n_subsets * len(folds)} fits; metric={SELECTION_METRIC}; '
        f'base={len(base_features)} cols')
    for name, cols in toggle_groups.items():
        log(f'  group {name:24s} {len(cols)} cols')
    for fold in folds:
        log(f"  fold test {fold['test_season']}: train {len(fold['train'])} "
            f"valid {len(fold['valid'])} test {len(fold['test'])}")

    t0 = time.time()

    def make_scorer(fold, pos):
        train_df, valid_df, test_df = (df.loc[fold['train']], df.loc[fold['valid']],
                                       df.loc[fold['test']])
        base_scorer = make_xgb_fold_scorer(
            train_df, valid_df, test_df, target=TARGET, metric=SELECTION_METRIC)
        state = {'n': 0}

        def scorer(features):
            score = base_scorer(features)
            state['n'] += 1
            log(f"fold {pos + 1}/{len(folds)} s{fold['test_season']} "
                f"subset {state['n']}/{n_subsets} nfeat={len(features)} "
                f"{SELECTION_METRIC}={score:.4f} [{time.time() - t0:.0f}s]")
            return score
        return scorer

    scorers = [make_scorer(f, i) for i, f in enumerate(folds)]
    result = cross_validated_group_ablation(
        toggle_groups, base_features, scorers, metric=SELECTION_METRIC,
        on_progress=lambda r: log(
            f"=== fold {r['fold'] + 1}/{len(folds)} complete @ "
            f"{time.time() - t0:.0f}s ==="))
    log(f'sweep done in {time.time() - t0:.0f}s')

    out = {
        'metric': SELECTION_METRIC, 'higher_is_better': HIGHER_IS_BETTER,
        'group_names': result['group_names'], 'num_folds': result['num_folds'],
        'base_features_n': len(base_features),
        'group_sizes': {n: len(c) for n, c in toggle_groups.items()},
        'shapley': result['shapley'], 'standalone': result['standalone'],
        'leave_one_out': result['leave_one_out'],
        'interactions': [{'pair': sorted(k), 'mean': v['mean'], 'se': v['se']}
                         for k, v in result['interactions'].items()],
        'mean_scores': [{'groups': sorted(k), 'n': len(k), 'mean': v}
                        for k, v in result['mean_scores'].items()],
    }
    json.dump(out, open(RESULT_JSON, 'w'), indent=2)
    log(f'wrote {RESULT_JSON}')

    names = result['group_names']
    mat = pd.DataFrame(0.0, index=names, columns=names)
    for pair, v in result['interactions'].items():
        i, j = sorted(pair)
        mat.loc[i, j] = v['mean']
        mat.loc[j, i] = v['mean']
    for k in range(len(names)):
        mat.iloc[k, k] = np.nan
    arr = mat.to_numpy()
    lim = float(np.nanmax(np.abs(arr))) or 1e-6
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(arr, cmap='RdBu', vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for a in range(len(names)):
        for b in range(len(names)):
            if a != b:
                ax.text(b, a, format(arr[a, b], '+.4f'),
                        ha='center', va='center', fontsize=7)
    ax.set_title(f'Pairwise interaction ({SELECTION_METRIC}): '
                 'blue=super-additive, red=sub-additive')
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(HEATMAP_PNG, dpi=120)
    log(f'wrote {HEATMAP_PNG}')

    log('SHAPLEY main effect (utility units; + = beneficial):')
    for name, v in sorted(result['shapley'].items(),
                          key=lambda kv: kv[1]['mean'], reverse=True):
        log(f"  {name:24s} {v['mean']:+.4f} +/- {v['se']:.4f}")
    log('PAIRWISE INTERACTION (most sub-additive first):')
    for d in sorted(out['interactions'], key=lambda x: x['mean']):
        log(f"  {' + '.join(d['pair']):44s} {d['mean']:+.4f} +/- {d['se']:.4f}")
    log('DONE')


if __name__ == '__main__':
    main()
