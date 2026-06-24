# XGBoost with Launch Ratings Implementation Plan (Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the first RFE feature-selection of the Madden launch-rating features (rank-only pool + 188 `madden_*` cols), tune XGBoost on the selected set, evaluate the 2024+2025 hold-out, and report — honestly — whether Madden survives selection and beats the champion (Run 28).

**Architecture:** A single reproducible module `scripts/experiments/xgb_launch_ratings.py` with pure, unit-tested helpers (rank-only pool builder, 2025-coverage guard, RFE-CSV reader, Madden-survival analyzer) and two orchestration functions — `run_rfe` (reuses `ClassifierCrossValidationRecursiveFeatureSelection`) and `run_grid_and_eval` (reuses `OptimalXGBHyperparameterSearch` + the already-tested `bayes_logistic.evaluate`). A final controller task rebuilds the parquet, runs the heavy RFE + grid search live, and records Run 28.

**Tech Stack:** Python 3.11, pandas, xgboost, scikit-learn, the in-repo RFE/search/eval utilities. Conda env `nfl-predictions`.

## Global Constraints

- **Seed-32 protocol (verbatim from grid_search.ipynb):** `df.sample(frac=1, random_state=32).reset_index(drop=True)`, then `np.random.seed(32)`. Splits: train `season<2022`, valid `2022–2023` (early stopping), hold-out `season>=2024`. RFE selection uses `season<2024`.
- **Reuse, do not reimplement:** `data_science_utilities.models.xgb.feature_selection.recursive.classifier_cross_validation.ClassifierCrossValidationRecursiveFeatureSelection`; `data_science_utilities.models.xgb.hyperparameter_search.random_search.OptimalXGBHyperparameterSearch`; `data_science_utilities.models.bayes_logistic.evaluate.evaluate`; `data_science_utilities.feature_groups.partition.is_rank_only_kept`.
- **RFE config (verbatim from rfe.ipynb):** `RFE_XGB_PARAMS = dict(n_estimators=5000, n_jobs=-1, learning_rate=.15, early_stopping_rounds=10, max_depth=5, eval_metric='auc', importance_type='total_gain', random_state=32)`; `model_score_metric='brier'`; `get_optimal_features_no_grouped_records(max_iter=60, min_features=5, verbose=1, on_iteration=...)`; selection via `get_best_num_features_1se()`.
- **Grid config (verbatim from grid_search.ipynb):** `RANDOM_XGB_PARAMS = {'learning_rate':[.03,.05,.1,.15,.2], 'max_depth':[2,3], 'subsample':[.5,.7,.9], 'min_child_weight':[10,20,50,100], 'gamma':[.5,1,5,10,100], 'n_estimators':[10,20,30,40,50], 'early_stopping_rounds':[10,20,50], 'importance_type':['total_gain'], 'eval_metric':['auc']}`; `RANDOM_SEARCH_PARAMS = dict(n_iter=100, cv=5, n_jobs=-1, random_state=32, scoring='neg_brier_score')`.
- **RANK_ONLY keeps Madden:** `is_rank_only_kept` returns True for every `madden_*` column (verified). All 188 enter the RFE pool.
- **Leakage/meta cols NEVER fed as features:** `game_id, season, season_type, opp_team, opp_score, target_team, target_score, h_win`. `xgb_features_list.csv` already excludes them; the pool builder additionally drops `target_win` (the target). `season` is used only for splits, then dropped. One-week shift + within-season `_ovr` z-score are already in the parquet — do not re-apply.
- **Champion anchors to beat:** XGB Run 11 = 0.707 / Brier 0.2206; BART Run 6 = 0.705 / 0.2194; BART Run 10 = 0.708 / 0.2185.
- **Precondition:** the parquet is git-ignored. Rebuild with `MADDEN_TOOLS_CDN_BASE=https://cdn.madden.tools python -m scripts.data_assembly.predict_game_winner.schedule_and_weekly` before the live run.
- **Tests:** `unittest` TestCases, fast (tiny synthetic frames, tiny model params injected); run with `conda run --no-capture-output -n nfl-predictions python -m pytest <path> -v`.

---

## File Structure

- **Create** `scripts/experiments/xgb_launch_ratings.py` — constants, pure helpers, `run_rfe`, `run_grid_and_eval`, `build_results`, `main`.
- **Create** `tests/experiments/__init__.py` and `tests/experiments/test_xgb_launch_ratings.py`.
- **Generated at run time (not committed except where noted):** `data/predict_games/model_features_in/rfe_features_kfolds_brier_madden.csv` (committed), `data/predict_games/xgb_launch_ratings/results.json` (committed), `models/best_random_xgb_model_launch_ratings.json` (committed).
- **Modify (Task 5):** `README.md` (Run 28 entry).

---

## Task 1: Pure helpers — pool, coverage guard, RFE-CSV reader, survival

**Files:**
- Create: `scripts/experiments/xgb_launch_ratings.py`
- Create: `tests/experiments/__init__.py` (empty), `tests/experiments/test_xgb_launch_ratings.py`

**Interfaces:**
- Consumes: `partition.is_rank_only_kept`.
- Produces: `TARGET`, `RANDOM_SEED`, `build_rfe_pool(candidate_features: list[str]) -> list[str]`; `madden_columns(features: list[str]) -> list[str]`; `madden_ovr_columns(df) -> list[str]`; `season_2025_ovr_coverage(df) -> float`; `assert_madden_2025_coverage(df, min_cov=0.30) -> None`; `selected_features_from_rfe_csv(path, best_num_feats: int) -> list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/experiments/test_xgb_launch_ratings.py
import os
import tempfile
import unittest
import numpy as np
import pandas as pd
from scripts.experiments import xgb_launch_ratings as xlr


class TestPureHelpers(unittest.TestCase):
    def test_build_rfe_pool_keeps_madden_drops_cumulative_and_target(self):
        cols = ['target_madden_qb_ovr', 'opp_madden_edge_ovr',
                'off_target_epa_per_play_cumulative_average',  # dropped (rank-only)
                'off_target_epa_per_play_rank',                # kept
                'week', 'target_win']                          # target_win dropped
        pool = xlr.build_rfe_pool(cols)
        self.assertIn('target_madden_qb_ovr', pool)
        self.assertIn('opp_madden_edge_ovr', pool)
        self.assertIn('off_target_epa_per_play_rank', pool)
        self.assertIn('week', pool)
        self.assertNotIn('off_target_epa_per_play_cumulative_average', pool)
        self.assertNotIn('target_win', pool)

    def test_madden_columns(self):
        feats = ['target_madden_qb_ovr', 'week', 'opp_madden_edge_ovr']
        self.assertEqual(xlr.madden_columns(feats),
                         ['target_madden_qb_ovr', 'opp_madden_edge_ovr'])

    def test_2025_coverage_and_guard(self):
        df = pd.DataFrame({
            'season': [2024, 2025, 2025],
            'target_madden_qb_ovr': [1.0, 2.0, np.nan],
            'opp_madden_qb_ovr': [0.5, np.nan, np.nan],
            'week': [1, 1, 2],
        })
        # 2025 rows: 4 _ovr cells, 1 non-null -> 0.25
        self.assertAlmostEqual(xlr.season_2025_ovr_coverage(df), 0.25)
        with self.assertRaises(RuntimeError):
            xlr.assert_madden_2025_coverage(df, min_cov=0.30)
        xlr.assert_madden_2025_coverage(df, min_cov=0.10)  # passes

    def test_selected_features_from_rfe_csv(self):
        # mimic get_features_in_dataframe().to_csv(): index = num_features, cols 0..N
        fdf = pd.DataFrame.from_dict(
            {3: ['a', 'b', 'c'], 2: ['a', 'b', None]}, orient='index')
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'rfe.csv')
            fdf.to_csv(p)
            self.assertEqual(xlr.selected_features_from_rfe_csv(p, 2), ['a', 'b'])
            self.assertEqual(xlr.selected_features_from_rfe_csv(p, 3), ['a', 'b', 'c'])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/experiments/test_xgb_launch_ratings.py::TestPureHelpers -v`
Expected: FAIL — `ModuleNotFoundError: scripts.experiments.xgb_launch_ratings`.

- [ ] **Step 3: Write minimal implementation** (create `scripts/experiments/xgb_launch_ratings.py`)

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/experiments/test_xgb_launch_ratings.py::TestPureHelpers -v`
Expected: PASS (4 tests). (Create the empty `tests/experiments/__init__.py` so discovery works.)

- [ ] **Step 5: Commit**

```bash
git add scripts/experiments/xgb_launch_ratings.py tests/experiments/__init__.py tests/experiments/test_xgb_launch_ratings.py
git commit -m "feat(xgb): Phase 1 pure helpers — rank-only pool, 2025 coverage guard, RFE reader"
```

---

## Task 2: `run_rfe` orchestration

**Files:**
- Modify: `scripts/experiments/xgb_launch_ratings.py`
- Modify: `tests/experiments/test_xgb_launch_ratings.py`

**Interfaces:**
- Consumes: `build_rfe_pool` (Task 1); `ClassifierCrossValidationRecursiveFeatureSelection`.
- Produces: `RFE_XGB_PARAMS` (constant); `run_rfe(df, candidate_features, out_csv, *, rfe_params=RFE_XGB_PARAMS, max_iter=60, min_features=5, n_folds=5) -> tuple[int, object]` — writes the features CSV, returns `(best_num_feats, rfe_object)`.

- [ ] **Step 1: Write the failing test** (append)

```python
def _synthetic(n_per_season=30):
    rng = np.random.default_rng(0)
    seasons = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    rows = []
    for s in seasons:
        for i in range(n_per_season):
            signal = rng.normal()
            rows.append({
                'season': s, 'week': (i % 17) + 1,
                'target_madden_qb_ovr': signal + rng.normal(0, 0.1),
                'opp_madden_edge_ovr': rng.normal(),
                'off_target_epa_per_play_rank': rng.normal(),
                'off_target_epa_per_play_cumulative_average': rng.normal(),  # rank-only drop
                'target_win': int(signal + rng.normal(0, 0.5) > 0),
            })
    return pd.DataFrame(rows)


class TestRunRfe(unittest.TestCase):
    def test_run_rfe_writes_csv_and_returns_best(self):
        df = _synthetic()
        cands = ['target_madden_qb_ovr', 'opp_madden_edge_ovr',
                 'off_target_epa_per_play_rank',
                 'off_target_epa_per_play_cumulative_average', 'week', 'target_win']
        tiny = dict(xlr.RFE_XGB_PARAMS, n_estimators=40)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, 'rfe.csv')
            best, rfe = xlr.run_rfe(df, cands, out, rfe_params=tiny,
                                    max_iter=3, min_features=2, n_folds=2)
            self.assertTrue(os.path.exists(out))
            self.assertIsInstance(best, (int, np.integer))
            # the cumulative col must never be in any selected row (rank-only filtered out)
            written = pd.read_csv(out, index_col=0)
            flat = set(written.values.ravel().tolist())
            self.assertNotIn('off_target_epa_per_play_cumulative_average', flat)
            self.assertNotIn('target_win', flat)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/experiments/test_xgb_launch_ratings.py::TestRunRfe -v`
Expected: FAIL — `AttributeError: ... has no attribute 'run_rfe'`.

- [ ] **Step 3: Write minimal implementation** (append to the script; add the import at top)

```python
from data_science_utilities.models.xgb.feature_selection.recursive.classifier_cross_validation import (
    ClassifierCrossValidationRecursiveFeatureSelection,
)

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
    np.random.seed(RANDOM_SEED)
    train = inputs[inputs['season'] < 2024]
    rfe = ClassifierCrossValidationRecursiveFeatureSelection(
        train[pool], train[TARGET], rfe_params, model_score_metric='brier')
    rfe.get_optimal_features_no_grouped_records(
        max_iter=max_iter, min_features=min_features, n_folds=n_folds,
        verbose=1, on_iteration=_log_rfe_progress)
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    rfe.get_features_in_dataframe().to_csv(out_csv)
    return rfe.get_best_num_features_1se(), rfe
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/experiments/test_xgb_launch_ratings.py::TestRunRfe -v`
Expected: PASS (1 test; a few seconds — tiny params).

- [ ] **Step 5: Commit**

```bash
git add scripts/experiments/xgb_launch_ratings.py tests/experiments/test_xgb_launch_ratings.py
git commit -m "feat(xgb): run_rfe — CV RFE over rank-only pool incl. madden, brier-1se"
```

---

## Task 3: `run_grid_and_eval` + `build_results`

**Files:**
- Modify: `scripts/experiments/xgb_launch_ratings.py`
- Modify: `tests/experiments/test_xgb_launch_ratings.py`

**Interfaces:**
- Consumes: `madden_columns` (Task 1); `OptimalXGBHyperparameterSearch`; `bayes_logistic.evaluate`.
- Produces: `RANDOM_XGB_PARAMS`, `RANDOM_SEARCH_PARAMS`, `CHAMPIONS` (constants); `run_grid_and_eval(df, selected_features, *, search_params=RANDOM_XGB_PARAMS, search_kwargs=RANDOM_SEARCH_PARAMS) -> tuple[object, dict, DataFrame]` returning `(best_model, metrics, holdout_df)`; `build_results(metrics, selected_features, best_model, best_num_feats) -> dict`.

- [ ] **Step 1: Write the failing test** (append)

```python
class TestGridAndResults(unittest.TestCase):
    def test_grid_eval_and_results(self):
        df = _synthetic(n_per_season=40)
        selected = ['target_madden_qb_ovr', 'opp_madden_edge_ovr',
                    'off_target_epa_per_play_rank', 'week']
        tiny_params = dict(xlr.RANDOM_XGB_PARAMS, n_estimators=[20, 30])
        tiny_search = dict(xlr.RANDOM_SEARCH_PARAMS, n_iter=2, cv=2)
        best, metrics, holdout = xlr.run_grid_and_eval(
            df, selected, search_params=tiny_params, search_kwargs=tiny_search)
        for k in ('auroc', 'accuracy', 'brier', 'logloss',
                  'per_week_auroc_mean', 'reliability_curve'):
            self.assertIn(k, metrics)
        res = xlr.build_results(metrics, selected, best, best_num_feats=len(selected))
        self.assertEqual(res['n_selected'], 4)
        self.assertEqual(set(res['madden_selected']),
                         {'target_madden_qb_ovr', 'opp_madden_edge_ovr'})
        self.assertIn('champion_deltas', res)
        self.assertIn('xgb_run11_auroc_delta', res['champion_deltas'])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/experiments/test_xgb_launch_ratings.py::TestGridAndResults -v`
Expected: FAIL — `AttributeError: run_grid_and_eval`.

- [ ] **Step 3: Write minimal implementation** (append; add imports)

```python
from data_science_utilities.models.xgb.hyperparameter_search.random_search import (
    OptimalXGBHyperparameterSearch,
)
from data_science_utilities.models.bayes_logistic.evaluate import evaluate

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
    np.random.seed(RANDOM_SEED)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/experiments/test_xgb_launch_ratings.py::TestGridAndResults -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add scripts/experiments/xgb_launch_ratings.py tests/experiments/test_xgb_launch_ratings.py
git commit -m "feat(xgb): run_grid_and_eval + build_results (reuse bayes evaluate; madden survival)"
```

---

## Task 4: `main` orchestration + paths + coverage guard

**Files:**
- Modify: `scripts/experiments/xgb_launch_ratings.py`
- Modify: `tests/experiments/test_xgb_launch_ratings.py`

**Interfaces:**
- Consumes: all of the above.
- Produces: path constants (`PARQUET`, `FEATURES_LIST`, `RFE_OUT`, `RESULTS_DIR`, `MODEL_OUT`); `main(stage='all')` that loads the parquet + candidate list, guards 2025 coverage, runs the requested stage(s), and writes `results.json` + the model. `stage in {'rfe','grid','all'}`.

- [ ] **Step 1: Write the failing test** (append — verifies wiring/guard without heavy training)

```python
class TestMainGuard(unittest.TestCase):
    def test_main_raises_on_low_2025_coverage(self):
        df = _synthetic(n_per_season=5)
        # blank out 2025 madden so coverage is 0 -> guard must fire
        df.loc[df['season'] == 2025,
               ['target_madden_qb_ovr', 'opp_madden_edge_ovr']] = np.nan
        with tempfile.TemporaryDirectory() as d:
            pq = os.path.join(d, 'sw.parquet')
            df.to_parquet(pq, index=False)
            fl = os.path.join(d, 'feats.csv')
            pd.DataFrame({'feature': ['target_madden_qb_ovr', 'opp_madden_edge_ovr',
                                      'off_target_epa_per_play_rank', 'week',
                                      'target_win']}).to_csv(fl, index=False)
            with self.assertRaises(RuntimeError):
                xlr.main(stage='rfe', parquet=pq, features_list=fl)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/experiments/test_xgb_launch_ratings.py::TestMainGuard -v`
Expected: FAIL — `main` missing / wrong signature.

- [ ] **Step 3: Write minimal implementation** (append)

```python
import argparse

PARQUET = 'data/predict_games/input_data/schedule_and_weekly.parquet'
FEATURES_LIST = 'data/predict_games/model_features_in/xgb_features_list.csv'
RFE_OUT = 'data/predict_games/model_features_in/rfe_features_kfolds_brier_madden.csv'
RESULTS_DIR = 'data/predict_games/xgb_launch_ratings'
MODEL_OUT = 'models/best_random_xgb_model_launch_ratings.json'


def main(stage='all', *, parquet=PARQUET, features_list=FEATURES_LIST, rfe_out=RFE_OUT,
         results_dir=RESULTS_DIR, model_out=MODEL_OUT, best_num_feats=None):
    df = pd.read_parquet(parquet)
    assert_madden_2025_coverage(df)
    candidates = list(pd.read_csv(features_list)['feature'])

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
    args = ap.parse_args()
    main(stage=args.stage, best_num_feats=args.best_num_feats)
```

(`stage='all'` threads `best_num_feats` straight from `run_rfe`; a standalone `stage='grid'` requires `--best-num-feats` from the prior rfe run. The coverage guard fires before any training, which is what the Task-4 test asserts.)

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/experiments/test_xgb_launch_ratings.py -v`
Expected: PASS (all tests — the guard test plus Tasks 1–3).

- [ ] **Step 5: Commit**

```bash
git add scripts/experiments/xgb_launch_ratings.py tests/experiments/test_xgb_launch_ratings.py
git commit -m "feat(xgb): main() orchestration + CLI stages + 2025 coverage guard"
```

---

## Task 5: Live run + Run 28 (controller)

Heavy + live (needs the CDN + nflverse). The controller runs this; not a subagent task.

**Files:**
- Create (commit): `data/predict_games/model_features_in/rfe_features_kfolds_brier_madden.csv`, `data/predict_games/xgb_launch_ratings/results.json`, `models/best_random_xgb_model_launch_ratings.json`
- Modify: `README.md`

- [ ] **Step 1: Rebuild the parquet with the Madden CDN source**

Run: `MADDEN_TOOLS_CDN_BASE=https://cdn.madden.tools conda run --no-capture-output -n nfl-predictions python -m scripts.data_assembly.predict_game_winner.schedule_and_weekly`
Expected: regenerates `schedule_and_weekly.parquet` + `xgb_features_list.csv`; sanity-check `xgb_features_list.csv` now contains `madden_*` columns and that 2025 `_ovr` coverage > 0.30 (the script's guard enforces this on the next step).

- [ ] **Step 2: Run RFE (background — the long pole)**

Run: `MADDEN_TOOLS_CDN_BASE=https://cdn.madden.tools conda run --no-capture-output -n nfl-predictions python -m scripts.experiments.xgb_launch_ratings --stage rfe`
Watch `/tmp/xgb_launch_rfe_progress.log`. Records the brier-1SE selected count and writes `rfe_features_kfolds_brier_madden.csv`.

- [ ] **Step 3: Run grid + eval (background)**

Run: `conda run --no-capture-output -n nfl-predictions python -m scripts.experiments.xgb_launch_ratings --stage grid --best-num-feats <N from Step 2>`
Produces `data/predict_games/xgb_launch_ratings/results.json` + `models/best_random_xgb_model_launch_ratings.json`.

- [ ] **Step 4: Record Run 28 in the README run-log**

Add the Run 28 table row + a narrative entry in the existing format: the hold-out AUROC / accuracy / Brier; how many of the 188 `madden_*` survived RFE and their importance ranks; and the honest verdict vs the champion (XGB 0.707, BART 0.705/0.708) — **a null result (Madden doesn't survive or doesn't beat the champion) is reported plainly.**

- [ ] **Step 5: Commit**

```bash
git add data/predict_games/model_features_in/rfe_features_kfolds_brier_madden.csv \
        data/predict_games/xgb_launch_ratings/results.json \
        models/best_random_xgb_model_launch_ratings.json README.md
git commit -m "experiment(xgb): Run 28 — XGBoost RFE/grid with launch ratings vs champion"
```

---

## Self-Review

**Spec coverage:**
- Reproducible script reusing RFE classifier + random_search + bayes evaluate → Tasks 1–4. ✓
- Rebuild-parquet precondition + 2025 coverage guard → Task 1 (guard) + Task 5 Step 1. ✓
- RFE on rank-only pool incl. madden, brier-1se → Task 2 (`build_rfe_pool`, `run_rfe`). ✓
- Grid search + champion protocol split + holdout eval → Task 3. ✓
- Madden survival analysis (count, columns, importance ranks) → Task 3 (`build_results`). ✓
- Single madden-inclusive run vs recorded champion (no A/B) → `CHAMPIONS` deltas, Task 3. ✓
- Run 28 README entry + honest verdict; outputs (CSV/results/model) → Task 5. ✓
- Unit tests for the Phase-1 glue → Tasks 1–4; reuse of the tested `evaluate` (not re-tested). ✓

**Placeholder scan:** Clean — `main` uses a normal `if/raise SystemExit` guard; the default `stage='all'` path threads `best_num_feats` from `run_rfe`. No TBD/TODO; every code/test step is complete.

**Type consistency:** `run_rfe` returns `(best_num_feats, rfe)`; `selected_features_from_rfe_csv(path, best_num_feats)` consumes that int; `run_grid_and_eval(df, selected_features)` returns `(best_model, metrics, holdout)`; `build_results(metrics, selected_features, best_model, best_num_feats)` consumes them. `RFE_XGB_PARAMS`/`RANDOM_XGB_PARAMS`/`RANDOM_SEARCH_PARAMS`/`CHAMPIONS` names are consistent across tasks and tests. `madden_columns` (list→list) vs `madden_ovr_columns` (df→list) are distinct and used correctly.
```
