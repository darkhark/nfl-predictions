"""Madden-internal sub-family ablation.

Runs a feature-group ablation OVER THE MADDEN SUB-FAMILIES to learn which
Madden units / measure-types carry the signal.  Two sweeps are produced:
  1. madden_internal_units_{metric}.json   — one toggle group per positional unit
  2. madden_internal_measures_{metric}.json — two toggle groups: level vs diff

The script deliberately does NOT train; call it headless:

    conda run -n nfl-predictions python scripts/experiments/madden_internal_ablation.py

Config via env vars: SELECTION_METRIC (default 'brier'), FIRST_TEST_SEASON (default 2019),
LAST_TEST_SEASON (default 2025).

Pattern mirrors:
    notebooks/model_training/predict_games/schedule_and_weekly/
        cross_validation/run_group_ablation.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Repo-root bootstrap (mirrors run_group_ablation.py)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_science_utilities.feature_groups.partition import partition_features, TARGET
from data_science_utilities.feature_groups.ablation import cross_validated_group_ablation
from data_science_utilities.feature_groups.scorers import (
    season_rolling_origin_folds, make_xgb_fold_scorer)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 32
RANK_ONLY = True
SELECTION_METRIC = os.environ.get('SELECTION_METRIC', 'brier')
FIRST_TEST_SEASON = int(os.environ.get('FIRST_TEST_SEASON', '2019'))
LAST_TEST_SEASON = int(os.environ.get('LAST_TEST_SEASON', '2025'))
VALID_YEARS = 2

PARQUET = os.path.join(REPO_ROOT, 'data', 'predict_games', 'input_data',
                       'schedule_and_weekly.parquet')
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'predict_games', 'group_ablation')

# ---------------------------------------------------------------------------
# Base → unit mapping
# ---------------------------------------------------------------------------
_UNIT_OF_BASE = {
    'qb': 'qb',
    'rb': 'offense_skill',
    'wr1': 'offense_skill',
    'wr2': 'offense_skill',
    'wr3': 'offense_skill',
    'te': 'offense_skill',
    'backfield': 'offense_skill',
    'receivers': 'offense_skill',
    'tight_end': 'offense_skill',
    'lt': 'oline',
    'lg': 'oline',
    'c': 'oline',
    'rg': 'oline',
    'rt': 'oline',
    'interior_ol': 'oline',
    'exterior_ol': 'oline',
    'edge': 'dfront',
    'edge_left': 'dfront',
    'edge_right': 'dfront',
    'interior_dl': 'dfront',
    'linebacker': 'coverage',
    'cornerback': 'coverage',
    'safety': 'coverage',
    'matchup_interior': 'matchup',
    'matchup_pass_pro': 'matchup',
    'matchup_pass_rush': 'matchup',
    'matchup_skill_cover': 'matchup',
}

# Ordered from longest to shortest to avoid prefix-matching bugs
_PREFIXES = ('target_madden_', 'opp_madden_', 'madden_')
_MEASURES = ('_ovr_diff_prev_season', '_ovr_diff_prev', '_ovr_diff_4g', '_ovr')


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def _madden_base(col: str) -> str:
    """Strip perspective prefix and measure suffix to get the base token.

    Examples
    --------
    >>> _madden_base('target_madden_qb_ovr')
    'qb'
    >>> _madden_base('opp_madden_edge_left_ovr_diff_4g')
    'edge_left'
    >>> _madden_base('madden_matchup_pass_pro')
    'matchup_pass_pro'
    """
    base = col
    for prefix in _PREFIXES:
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    # Strip measure suffix (check longest first to avoid partial matches)
    for measure in _MEASURES:
        if base.endswith(measure):
            return base[: -len(measure)]
    return base  # matchup_* have no measure suffix


def madden_unit(col: str) -> str:
    """Return the positional-unit family for a Madden column.

    Parameters
    ----------
    col:
        A Madden feature column name such as ``'target_madden_qb_ovr'`` or
        ``'madden_matchup_pass_pro'``.

    Returns
    -------
    str
        One of: ``'qb'``, ``'offense_skill'``, ``'oline'``, ``'dfront'``,
        ``'coverage'``, ``'matchup'``.

    Raises
    ------
    KeyError
        If the base token is not recognised in ``_UNIT_OF_BASE``.
    """
    return _UNIT_OF_BASE[_madden_base(col)]


def madden_measure(col: str) -> str:
    """Return ``'diff'`` for change measures, ``'level'`` for raw OVR and matchup deltas.

    Diff measures end with ``_ovr_diff_prev_season``, ``_ovr_diff_prev``, or
    ``_ovr_diff_4g``.  Everything else (raw ``_ovr`` levels and the four
    ``madden_matchup_*`` columns) is classified as ``'level'``.
    """
    if col.endswith(('_ovr_diff_prev_season', '_ovr_diff_prev', '_ovr_diff_4g')):
        return 'diff'
    return 'level'


def madden_unit_groups(madden_cols: list) -> dict:
    """Partition *madden_cols* into a dict keyed by positional-unit family.

    Parameters
    ----------
    madden_cols:
        List of Madden feature column names.

    Returns
    -------
    dict[str, list[str]]
        Keys are unit family names (e.g. ``'qb'``, ``'oline'``); values are
        the corresponding column lists.
    """
    groups: dict = {}
    for col in madden_cols:
        groups.setdefault(madden_unit(col), []).append(col)
    return groups


def madden_measure_groups(madden_cols: list) -> dict:
    """Partition *madden_cols* into ``{'level': [...], 'diff': [...]}``.

    Parameters
    ----------
    madden_cols:
        List of Madden feature column names.

    Returns
    -------
    dict[str, list[str]]
    """
    groups: dict = {}
    for col in madden_cols:
        groups.setdefault(madden_measure(col), []).append(col)
    return groups


# ---------------------------------------------------------------------------
# Ablation runner (not unit-tested with heavy compute)
# ---------------------------------------------------------------------------

def run_internal_ablation(df, toggle_groups, base_features, folds, metric, out_json,
                          progress_log):
    """Run cross-validated group ablation for the given toggle groups.

    Parameters
    ----------
    df : pd.DataFrame
        Full shuffled dataframe (all rows; fold index arrays index into it).
    toggle_groups : dict[str, list[str]]
        Mapping of group name → list of feature column names to toggle.
    base_features : list[str]
        Always-included context feature columns.
    folds : list[dict]
        Output of ``season_rolling_origin_folds``.
    metric : str
        Scoring metric passed to ``make_xgb_fold_scorer``.
    out_json : str
        Destination file path for the JSON result.
    progress_log : str
        File path for live progress messages.

    Returns
    -------
    dict
        The raw result dict from ``cross_validated_group_ablation`` plus a
        ``'group_sizes'`` key — same shape as ``run_group_ablation.py`` output.
    """
    import time

    def _log(msg):
        with open(progress_log, 'a') as fh:
            print(time.strftime('%H:%M:%S'), msg, file=fh)
        print(msg, flush=True)

    def _make_scorer(fold):
        train_df = df.loc[fold['train']]
        valid_df = df.loc[fold['valid']]
        test_df = df.loc[fold['test']]
        return make_xgb_fold_scorer(train_df, valid_df, test_df, target=TARGET, metric=metric)

    scorers = [_make_scorer(f) for f in folds]

    result = cross_validated_group_ablation(
        toggle_groups, base_features, scorers, metric=metric)

    out = {
        'metric': metric,
        'group_names': result['group_names'],
        'num_folds': result['num_folds'],
        'base_features_n': len(base_features),
        'group_sizes': {n: len(c) for n, c in toggle_groups.items()},
        'shapley': result['shapley'],
        'standalone': result['standalone'],
        'leave_one_out': result['leave_one_out'],
        'interactions': [
            {'pair': sorted(k), 'mean': v['mean'], 'se': v['se']}
            for k, v in result['interactions'].items()
        ],
        'mean_scores': [
            {'groups': sorted(k), 'n': len(k), 'mean': v}
            for k, v in result['mean_scores'].items()
        ],
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(out, open(out_json, 'w'), indent=2)
    _log(f'wrote {out_json}')
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    np.random.seed(RANDOM_SEED)
    print(f'Loading {PARQUET}')
    df = (pd.read_parquet(PARQUET)
          .sample(frac=1, random_state=RANDOM_SEED)
          .reset_index(drop=True))

    groups = partition_features(list(df.columns), rank_only=RANK_ONLY)

    if 'madden_ratings' not in groups:
        raise RuntimeError(
            "'madden_ratings' group not found — parquet was not built with Madden features. "
            "Re-build the parquet with Madden ratings included before running this script."
        )

    base_features = groups['context_rest']
    madden_cols = groups['madden_ratings']

    # Build sub-family toggle groups
    unit_groups = {k: v for k, v in madden_unit_groups(madden_cols).items() if v}
    measure_groups = {k: v for k, v in madden_measure_groups(madden_cols).items() if v}

    metric = SELECTION_METRIC

    # Single shared fold set for both ablations
    folds = season_rolling_origin_folds(
        df,
        first_test_season=FIRST_TEST_SEASON,
        last_test_season=LAST_TEST_SEASON,
        valid_years=VALID_YEARS,
    )
    print(f'{len(folds)} folds; metric={metric}')
    print(f'Madden cols: {len(madden_cols)} | unit groups: {list(unit_groups)} '
          f'| measure groups: {list(measure_groups)}')

    os.makedirs(OUT_DIR, exist_ok=True)
    progress_log = os.environ.get('PROGRESS_LOG', '/tmp/madden_internal_ablation_progress.log')
    open(progress_log, 'w').close()

    # --- Units ablation ---
    units_json = os.path.join(OUT_DIR, f'madden_internal_units_{metric}.json')
    print(f'\n=== Units ablation ({len(unit_groups)} groups) -> {units_json}')
    run_internal_ablation(
        df, unit_groups, base_features, folds, metric, units_json, progress_log)

    # --- Measures ablation ---
    measures_json = os.path.join(OUT_DIR, f'madden_internal_measures_{metric}.json')
    print(f'\n=== Measures ablation ({len(measure_groups)} groups) -> {measures_json}')
    run_internal_ablation(
        df, measure_groups, base_features, folds, metric, measures_json, progress_log)

    print('\nDONE')


if __name__ == '__main__':
    main()
