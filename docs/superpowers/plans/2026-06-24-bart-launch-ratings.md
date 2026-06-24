# BART with Launch Ratings Implementation Plan (Phase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run BART backward-elimination over the full 188-column Madden family + the BART champion's start features, then a final 4-chain BART, and report the 2024+2025 hold-out vs the BART champion (Run 30).

**Architecture:** A reproducible `scripts/experiments/bart_launch_ratings.py` with pure, unit-tested helpers (start-pool builder, sentinel matrix, results) and BART run functions reusing `BartBackwardElimination`, the `bart_fit` pattern (probit BART, `variable_inclusion` ranking), and the tested `bayes_logistic.evaluate` (with `p_std` → width-stratified Brier). The heavy BART fits are the live experiment, run by the controller in the background.

**Tech Stack:** Python 3.11, PyMC 5.23 + pymc_bart 0.9.2, pandas, scikit-learn, the in-repo BART backward-elimination + eval utilities. Conda env `nfl-predictions`; 14 cores.

## Global Constraints

- **Seed-32 protocol verbatim:** `df.sample(frac=1, random_state=32).reset_index(drop=True)`; split train `season<2022` / valid `2022–2023` / hold-out `season>=2024` (~1,088 rows).
- **Reuse (no reimplementation):** `data_science_utilities.models.bart.feature_selection.backward_elimination.BartBackwardElimination`; `data_science_utilities.models.bayes_logistic.evaluate.evaluate`; `scripts.experiments.xgb_launch_ratings.madden_columns` + `assert_madden_2025_coverage`.
- **Start pool = champion-start + full Madden:** the 90 non-Madden features at row 90 of `data/predict_games/model_features_in/rfe_features_kfolds_brier.csv` + all 188 `madden_*` columns (levels AND diffs) = 278 candidates.
- **NaN → `-100` sentinel** (`BART_NAN_SENTINEL = -100.0`): BART cannot ingest NaN; fill then assert no NaN remains (the `bart.ipynb` stale-output guard).
- **BART selection fit (verbatim from `bart_rfe.ipynb`):** probit BART `pmb.BART('mu', X, y, m=50)`, `pm.sample(draws=500, tune=1000, chains=2, cores=2, random_seed=seed, progressbar=False)`; validation = `brier_score_loss(y_valid, valid_preds)`; `variable_inclusion` = chain-averaged, a `pd.Series` **indexed by feature name** (BartBackwardElimination raises if any candidate is missing).
- **Backward elimination:** `BartBackwardElimination(fit, drop_rate=0.2, min_features=10, replicates=6, max_workers=6, base_seed=32)`; select via `get_best_features_1se(higher_is_better=False)` (Brier is lower-is-better).
- **Final BART (verbatim from `bart.ipynb`):** probit BART `m=50`, `pm.sample(draws=1000, tune=1000, chains=4, cores=4, random_seed=32, progressbar=False)`; hold-out preds = posterior-mean P(win); posterior **std** retained for width-stratified Brier.
- **Eval:** `evaluate(y_true, preds, p_std=p_std, weeks=weeks)` — auroc, accuracy, brier, logloss, per-week-auroc, reliability, **width-stratified Brier**. Compare to BART Run 10 = 0.708/0.2185, Run 6 = 0.705/0.2194.
- **Never feed leakage/meta cols:** game_id, season, season_type, opp_team, opp_score, target_team, target_score, h_win, target_win. `season` used only for splits.
- **Tests:** `unittest` TestCases. Pure helpers + stub-driven RFE wiring are fast/hermetic; BART-touching functions get ONE tiny real fit each (synthetic data, `draws=20, tune=20, chains=1`). Run with `conda run --no-capture-output -n nfl-predictions python -m pytest <path> -v`.

---

## File Structure
- **Create** `scripts/experiments/bart_launch_ratings.py` — constants, pure helpers, `make_bart_fit`, `run_bart_rfe`, `run_final_bart`, `build_results`, `main`.
- **Create** `tests/experiments/test_bart_launch_ratings.py`.
- **Generated at run time (committed):** `data/predict_games/model_features_in/bart_rfe_features_brier_madden.csv`, `data/predict_games/bart_launch_ratings/results.json`.
- **Modify (Task 5):** `README.md` (Run 30).

---

## Task 1: Pure helpers — start pool, sentinel matrix, split, results

**Files:**
- Create: `scripts/experiments/bart_launch_ratings.py`, `tests/experiments/test_bart_launch_ratings.py`

**Interfaces:**
- Consumes: `xgb_launch_ratings.madden_columns`.
- Produces: constants (`RANDOM_SEED=32`, `TARGET`, `BART_NAN_SENTINEL=-100.0`, `META`, `CHAMPIONS`); `build_start_pool(champion_start, madden_cols) -> list[str]`; `to_bart_matrix(df, features) -> np.ndarray`; `split_seasons(df) -> (train, valid, holdout)`; `build_results(metrics, selected_features, history, best_num_feats) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/experiments/test_bart_launch_ratings.py
import unittest
import numpy as np
import pandas as pd
from scripts.experiments import bart_launch_ratings as blr


class TestPureHelpers(unittest.TestCase):
    def test_build_start_pool(self):
        champ = ['off_target_epa_per_play_rank', 'week', 'game_id']  # game_id is meta -> dropped
        madden = ['target_madden_qb_ovr', 'target_madden_qb_ovr_diff_prev', 'target_madden_qb_ovr']  # dup
        pool = blr.build_start_pool(champ, madden)
        self.assertEqual(pool, ['off_target_epa_per_play_rank', 'week',
                                'target_madden_qb_ovr', 'target_madden_qb_ovr_diff_prev'])
        self.assertNotIn('game_id', pool)        # meta dropped
        self.assertEqual(len(pool), len(set(pool)))  # deduped

    def test_to_bart_matrix_fills_sentinel(self):
        df = pd.DataFrame({'a': [1.0, np.nan], 'b': [np.nan, 2.0]})
        X = blr.to_bart_matrix(df, ['a', 'b'])
        self.assertFalse(np.isnan(X).any())
        self.assertEqual(X[1, 0], -100.0)
        self.assertEqual(X.dtype, float)

    def test_split_seasons(self):
        df = pd.DataFrame({'season': [2019, 2021, 2022, 2023, 2024, 2025],
                           'target_win': [0, 1, 0, 1, 0, 1], 'x': range(6)})
        tr, va, ho = blr.split_seasons(df)
        self.assertEqual(set(tr['season']), {2019, 2021})
        self.assertEqual(set(va['season']), {2022, 2023})
        self.assertEqual(set(ho['season']), {2024, 2025})

    def test_build_results(self):
        metrics = {'auroc': 0.71, 'brier': 0.218}
        hist = pd.DataFrame({'num_features': [20, 16], 'validation_score': [0.222, 0.220]})
        res = blr.build_results(metrics, ['target_madden_qb_ovr', 'week'], hist, best_num_feats=16)
        self.assertEqual(res['n_selected'], 2)
        self.assertEqual(res['madden_selected'], ['target_madden_qb_ovr'])
        self.assertEqual(res['n_madden'], 1)
        self.assertIn('bart_run10_auroc_delta', res['champion_deltas'])
        self.assertAlmostEqual(res['champion_deltas']['bart_run10_auroc_delta'], 0.71 - 0.708, places=4)
        self.assertEqual(len(res['validation_curve']), 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/experiments/test_bart_launch_ratings.py::TestPureHelpers -v`
Expected: FAIL — `ModuleNotFoundError: scripts.experiments.bart_launch_ratings`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/experiments/bart_launch_ratings.py
"""Phase 2 — BART with the full Madden launch-rating family.

BART backward-elimination over the champion-start features + all 188 madden_* columns,
then a final 4-chain BART, comparable to the BART champion (Run 10 = 0.708). Reuses the
in-repo BartBackwardElimination + bayes_logistic.evaluate. NaN -> -100 sentinel."""
import argparse
import json
import os
import pandas as pd

from scripts.experiments.xgb_launch_ratings import madden_columns

RANDOM_SEED = 32
TARGET = 'target_win'
BART_NAN_SENTINEL = -100.0
META = {'game_id', 'season', 'season_type', 'opp_team', 'opp_score', 'target_team',
        'target_score', 'h_win', TARGET}
CHAMPIONS = {'bart_run6': {'auroc': 0.705, 'brier': 0.2194},
             'bart_run10': {'auroc': 0.708, 'brier': 0.2185}}


def build_start_pool(champion_start, madden_cols):
    """90 non-madden champion-start features + all madden columns; meta/target dropped,
    de-duplicated, order preserved (champion-start first, then madden)."""
    out, seen = [], set()
    for f in list(champion_start) + list(madden_cols):
        if f in META or f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def to_bart_matrix(df, features):
    """df[features] with NaN -> -100 sentinel as a float matrix (BART cannot ingest NaN)."""
    import numpy as np
    X = df[features].fillna(BART_NAN_SENTINEL).to_numpy(dtype=float)
    assert not np.isnan(X).any(), 'NaN remains after sentinel fill'
    return X


def split_seasons(df):
    """Seed-32 shuffle, then (train <2022, valid 2022-2023, holdout >=2024)."""
    data = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    train = data[data['season'] < 2022]
    valid = data[(data['season'] >= 2022) & (data['season'] < 2024)]
    holdout = data[data['season'] >= 2024].copy()
    return train, valid, holdout


def build_results(metrics, selected_features, history, best_num_feats):
    mads = madden_columns(selected_features)
    deltas = {}
    for name, champ in CHAMPIONS.items():
        deltas[f'{name}_auroc_delta'] = round(metrics['auroc'] - champ['auroc'], 4)
        deltas[f'{name}_brier_delta'] = round(metrics['brier'] - champ['brier'], 4)
    return {
        'config': {'seed': RANDOM_SEED, 'selection_metric': 'brier',
                   'nan_sentinel': BART_NAN_SENTINEL, 'best_num_feats': int(best_num_feats)},
        'n_selected': len(selected_features),
        'madden_selected': mads,
        'n_madden': len(mads),
        'metrics': metrics,
        'champion_deltas': deltas,
        'validation_curve': [{'n': int(r['num_features']),
                              'val_brier': float(r['validation_score'])}
                             for _, r in history.iterrows()],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/experiments/test_bart_launch_ratings.py::TestPureHelpers -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/experiments/bart_launch_ratings.py tests/experiments/test_bart_launch_ratings.py
git commit -m "feat(bart): Phase 2 pure helpers — start pool, -100 matrix, split, results"
```

---

## Task 2: BART selection — `make_bart_fit` + `run_bart_rfe`

**Files:**
- Modify: `scripts/experiments/bart_launch_ratings.py`, `tests/experiments/test_bart_launch_ratings.py`

**Interfaces:**
- Consumes: `to_bart_matrix`, `split_seasons` (Task 1); `BartBackwardElimination`; PyMC/pymc_bart.
- Produces: `make_bart_fit(train_df, valid_df, y_train, y_valid, *, m=50, draws=500, tune=1000, chains=2, cores=2) -> fit_fn`; `run_bart_rfe(df, start_features, out_csv, *, fit_fn=None, replicates=6, max_workers=6, drop_rate=0.2, min_features=10, fit_kwargs=None) -> (best_features, history)`. `fit_fn(features, seed) -> {'validation_score': float, 'variable_inclusion': pd.Series indexed by feature name}`.

- [ ] **Step 1: Write the failing test** (append; add a `_synthetic` helper)

```python
def _synthetic(n_per_season=40):
    rng = np.random.default_rng(0)
    rows = []
    for s in (2019, 2020, 2021, 2022, 2023, 2024, 2025):
        for i in range(n_per_season):
            sig = rng.normal()
            rows.append({'season': s, 'week': (i % 5) + 1,
                         'f_qb': sig + rng.normal(0, 0.2),
                         'f_b': rng.normal(), 'f_c': rng.normal(),
                         'target_win': int(sig + rng.normal(0, 0.5) > 0)})
    return pd.DataFrame(rows)


class TestRunBartRfeWiring(unittest.TestCase):
    def test_rfe_uses_fit_fn_and_writes_csv(self):
        # Stub fit_fn: instant, deterministic; importance favors f_qb so it survives.
        def stub_fit(features, seed):
            incl = pd.Series({f: (3.0 if f == 'f_qb' else 1.0) for f in features})
            return {'validation_score': 0.22 - 0.001 * (3 - len(features)),
                    'variable_inclusion': incl}
        df = _synthetic()
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, 'bart_rfe.csv')
            best, hist = blr.run_bart_rfe(
                df, ['f_qb', 'f_b', 'f_c'], out, fit_fn=stub_fit,
                replicates=1, max_workers=1, min_features=1)
            self.assertTrue(os.path.exists(out))
            self.assertIn('f_qb', best)                 # highest importance survives
            self.assertIn('feature', pd.read_csv(out).columns)
            self.assertGreaterEqual(len(hist), 1)


class TestBartFitContract(unittest.TestCase):
    def test_tiny_real_fit_returns_contract(self):
        # ONE tiny real PyMC BART fit (slow ~20-40s) to prove the model + return shape.
        df = _synthetic(n_per_season=20)
        tr, va, _ = blr.split_seasons(df)
        fit = blr.make_bart_fit(tr, va, tr['target_win'].to_numpy(int),
                                va['target_win'].to_numpy(int),
                                draws=20, tune=20, chains=1, cores=1)
        out = fit(['f_qb', 'f_b', 'f_c'], seed=32)
        self.assertIn('validation_score', out)
        self.assertIsInstance(out['validation_score'], float)
        self.assertIsInstance(out['variable_inclusion'], pd.Series)
        self.assertEqual(set(out['variable_inclusion'].index), {'f_qb', 'f_b', 'f_c'})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/experiments/test_bart_launch_ratings.py::TestRunBartRfeWiring -v`
Expected: FAIL — `AttributeError: ... has no attribute 'run_bart_rfe'`.

- [ ] **Step 3: Write minimal implementation** (append; add imports at top)

```python
import time
import numpy as np
import pymc as pm
import pymc_bart as pmb
from sklearn.metrics import brier_score_loss
from data_science_utilities.models.bart.feature_selection.backward_elimination import (
    BartBackwardElimination,
)

_RFE_PROGRESS_LOG = '/tmp/bart_launch_rfe_progress.log'


def _log_iter(row):
    with open(_RFE_PROGRESS_LOG, 'a') as fh:
        fh.write(f"iter: {row['num_features']} features, "
                 f"val_brier {row['validation_score']:.4f}\n")


def make_bart_fit(train_df, valid_df, y_train, y_valid, *,
                  m=50, draws=500, tune=1000, chains=2, cores=2):
    """Build a selection-grade BART fit_fn: train probit BART on <2022, score the
    2022-2023 validation slice with Brier, return chain-averaged variable_inclusion."""
    def bart_fit(features, seed):
        t0 = time.perf_counter()
        X_train = to_bart_matrix(train_df, features)
        X_valid = to_bart_matrix(valid_df, features)
        with pm.Model():
            X_data = pm.Data('X', X_train)
            mu = pmb.BART('mu', X_data, y_train, m=m)
            p = pm.Deterministic('p', pm.math.invprobit(mu))
            pm.Bernoulli('y', p=p, observed=y_train, shape=mu.shape)
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                              random_seed=seed, progressbar=False)
            pm.set_data({'X': X_valid})
            ppc = pm.sample_posterior_predictive(
                idata, var_names=['p'], random_seed=seed, progressbar=False)
        valid_preds = ppc.posterior_predictive['p'].mean(dim=['chain', 'draw']).to_numpy()
        inclusion = pd.Series(
            idata.sample_stats['variable_inclusion'].mean(dim=['chain', 'draw']).to_numpy(),
            index=features)
        with open(_RFE_PROGRESS_LOG, 'a') as fh:
            fh.write(f"  fit: {len(features)} feat seed {seed} "
                     f"{time.perf_counter() - t0:.0f}s\n")
        return {'validation_score': float(brier_score_loss(y_valid, valid_preds)),
                'variable_inclusion': inclusion}
    return bart_fit


def run_bart_rfe(df, start_features, out_csv, *, fit_fn=None, replicates=6, max_workers=6,
                 drop_rate=0.2, min_features=10, fit_kwargs=None):
    """BART backward-elimination from start_features; writes the selected set (brier-1SE)
    and returns (best_features, history). Inject fit_fn for tests; else build a real one."""
    train, valid, _ = split_seasons(df)
    if fit_fn is None:
        fit_fn = make_bart_fit(train, valid, train[TARGET].to_numpy(dtype=int),
                               valid[TARGET].to_numpy(dtype=int), **(fit_kwargs or {}))
    rfe = BartBackwardElimination(fit_fn, drop_rate=drop_rate, min_features=min_features,
                                  replicates=replicates, max_workers=max_workers,
                                  base_seed=RANDOM_SEED, on_iteration=_log_iter)
    history = rfe.run(start_features)
    best = rfe.get_best_features_1se(higher_is_better=False)
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    pd.DataFrame({'feature': best}).to_csv(out_csv, index=False)
    return best, history
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/experiments/test_bart_launch_ratings.py::TestRunBartRfeWiring tests/experiments/test_bart_launch_ratings.py::TestBartFitContract -v`
Expected: PASS (2 tests; the contract test takes ~20-40s for one tiny real BART fit).

- [ ] **Step 5: Commit**

```bash
git add scripts/experiments/bart_launch_ratings.py tests/experiments/test_bart_launch_ratings.py
git commit -m "feat(bart): make_bart_fit + run_bart_rfe (backward elimination, brier-1SE)"
```

---

## Task 3: Final BART + eval — `run_final_bart`

**Files:**
- Modify: `scripts/experiments/bart_launch_ratings.py`, `tests/experiments/test_bart_launch_ratings.py`

**Interfaces:**
- Consumes: `to_bart_matrix`, `split_seasons` (Task 1); `bayes_logistic.evaluate`.
- Produces: `run_final_bart(df, selected_features, *, m=50, draws=1000, tune=1000, chains=4, cores=4) -> (preds, p_std, holdout_df)`.

- [ ] **Step 1: Write the failing test** (append)

```python
class TestRunFinalBart(unittest.TestCase):
    def test_tiny_real_final_fit_and_eval(self):
        from data_science_utilities.models.bayes_logistic.evaluate import evaluate
        df = _synthetic(n_per_season=25)
        preds, p_std, holdout = blr.run_final_bart(
            df, ['f_qb', 'f_b', 'f_c'], draws=20, tune=20, chains=1, cores=1)
        self.assertEqual(len(preds), len(holdout))
        self.assertEqual(len(p_std), len(holdout))
        self.assertTrue(((preds >= 0) & (preds <= 1)).all())
        metrics = evaluate(holdout['target_win'].to_numpy(int), preds,
                           p_std=p_std, weeks=holdout['week'])
        for k in ('auroc', 'brier', 'per_week_auroc_mean',
                  'width_stratified_brier', 'reliability_curve'):
            self.assertIn(k, metrics)   # p_std present -> width_stratified_brier emitted
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/experiments/test_bart_launch_ratings.py::TestRunFinalBart -v`
Expected: FAIL — `AttributeError: run_final_bart`.

- [ ] **Step 3: Write minimal implementation** (append)

```python
def run_final_bart(df, selected_features, *, m=50, draws=1000, tune=1000, chains=4, cores=4):
    """Final probit BART on the selected set; returns (holdout posterior-mean preds,
    posterior std, holdout_df). The std feeds the width-stratified Brier."""
    train, _, holdout = split_seasons(df)
    y_train = train[TARGET].to_numpy(dtype=int)
    X_train = to_bart_matrix(train, selected_features)
    X_holdout = to_bart_matrix(holdout, selected_features)
    with pm.Model():
        X_data = pm.Data('X', X_train)
        mu = pmb.BART('mu', X_data, y_train, m=m)
        p = pm.Deterministic('p', pm.math.invprobit(mu))
        pm.Bernoulli('y', p=p, observed=y_train, shape=mu.shape)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                          random_seed=RANDOM_SEED, progressbar=False)
        pm.set_data({'X': X_holdout})
        ppc = pm.sample_posterior_predictive(
            idata, var_names=['p'], random_seed=RANDOM_SEED, progressbar=False)
    post = ppc.posterior_predictive['p']
    preds = post.mean(dim=['chain', 'draw']).to_numpy()
    p_std = post.std(dim=['chain', 'draw']).to_numpy()
    return preds, p_std, holdout
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/experiments/test_bart_launch_ratings.py::TestRunFinalBart -v`
Expected: PASS (1 test, ~20-40s tiny real fit).

- [ ] **Step 5: Commit**

```bash
git add scripts/experiments/bart_launch_ratings.py tests/experiments/test_bart_launch_ratings.py
git commit -m "feat(bart): run_final_bart — 4-chain holdout fit with posterior std"
```

---

## Task 4: `main` orchestration + CLI + coverage guard

**Files:**
- Modify: `scripts/experiments/bart_launch_ratings.py`, `tests/experiments/test_bart_launch_ratings.py`

**Interfaces:**
- Consumes: all of the above; `xgb_launch_ratings.assert_madden_2025_coverage`.
- Produces: path constants (`PARQUET`, `CHAMPION_START_CSV`, `CHAMPION_START_COUNT=90`, `RFE_OUT`, `RESULTS_DIR`); `load_champion_start(path, count) -> list[str]`; `main(stage='all', *, parquet=PARQUET, ..., selected=None)`. `stage in {'rfe','final','all'}`.

- [ ] **Step 1: Write the failing test** (append — guard + start-pool loader, no BART)

```python
class TestMainGuardAndStart(unittest.TestCase):
    def test_load_champion_start(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'rfe.csv')
            # RFE-by-count table: index = num_features, cols = feature slots
            pd.DataFrame.from_dict({90: ['a', 'b'], 50: ['a', None]},
                                   orient='index').to_csv(p)
            self.assertEqual(blr.load_champion_start(p, 90), ['a', 'b'])

    def test_main_rfe_raises_on_low_2025_coverage(self):
        import tempfile, os
        df = _synthetic(n_per_season=5)
        # rename so the coverage guard sees a madden _ovr col with no 2025 coverage
        df = df.rename(columns={'f_qb': 'target_madden_qb_ovr'})
        df.loc[df['season'] == 2025, 'target_madden_qb_ovr'] = np.nan
        with tempfile.TemporaryDirectory() as d:
            pq = os.path.join(d, 'sw.parquet'); df.to_parquet(pq, index=False)
            cs = os.path.join(d, 'cs.csv')
            pd.DataFrame.from_dict({90: ['f_b']}, orient='index').to_csv(cs)
            with self.assertRaises(RuntimeError):
                blr.main(stage='rfe', parquet=pq, champion_start_csv=cs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/experiments/test_bart_launch_ratings.py::TestMainGuardAndStart -v`
Expected: FAIL — `load_champion_start` / `main` missing.

- [ ] **Step 3: Write minimal implementation** (append)

```python
from scripts.experiments.xgb_launch_ratings import assert_madden_2025_coverage
from data_science_utilities.models.bayes_logistic.evaluate import evaluate

PARQUET = 'data/predict_games/input_data/schedule_and_weekly.parquet'
CHAMPION_START_CSV = 'data/predict_games/model_features_in/rfe_features_kfolds_brier.csv'
CHAMPION_START_COUNT = 90
RFE_OUT = 'data/predict_games/model_features_in/bart_rfe_features_brier_madden.csv'
RESULTS_DIR = 'data/predict_games/bart_launch_ratings'


def load_champion_start(path, count):
    """The non-madden champion start features (row `count` of the XGBoost RFE table)."""
    table = pd.read_csv(path, index_col=0)
    return list(table.loc[count, :].dropna().values)


def main(stage='all', *, parquet=PARQUET, champion_start_csv=CHAMPION_START_CSV,
         champion_start_count=CHAMPION_START_COUNT, rfe_out=RFE_OUT,
         results_dir=RESULTS_DIR, selected=None):
    df = pd.read_parquet(parquet)
    assert_madden_2025_coverage(df)
    best = selected
    history = None

    if stage in ('rfe', 'all'):
        champ = load_champion_start(champion_start_csv, champion_start_count)
        madden = madden_columns(list(df.columns))
        start = build_start_pool(champ, madden)
        print(f'BART RFE start pool: {len(start)} features ({len(madden)} madden)')
        best, history = run_bart_rfe(df, start, rfe_out)
        print(f'1-SE selected {len(best)} features ({len(madden_columns(best))} madden)')
    if stage in ('final', 'all'):
        if best is None:
            best = list(pd.read_csv(rfe_out)['feature'])
        if history is None:
            # validation curve is an rfe-only artifact; standalone --stage final has none
            history = pd.DataFrame({'num_features': [len(best)], 'validation_score': [float('nan')]})
        preds, p_std, holdout = run_final_bart(df, best)
        metrics = evaluate(holdout[TARGET].to_numpy(dtype=int), preds,
                           p_std=p_std, weeks=holdout['week'])
        os.makedirs(results_dir, exist_ok=True)
        results = build_results(metrics, best, history, len(best))
        with open(os.path.join(results_dir, 'results.json'), 'w') as fh:
            json.dump(results, fh, indent=2)
            fh.write('\n')
        print(json.dumps(results['metrics'], indent=2))
        print('champion deltas:', results['champion_deltas'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['rfe', 'final', 'all'], default='all')
    args = ap.parse_args()
    main(stage=args.stage)
```

NOTE for the implementer: `history` is initialised to `None` at the top of `main`; the rfe branch binds it, and a standalone `--stage final` (rfe didn't run) falls through to the 1-row placeholder — the validation curve is an rfe-only artifact. Implement exactly as written.

- [ ] **Step 4: Run test to verify it passes (and the whole file)**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/experiments/test_bart_launch_ratings.py -v`
Expected: PASS — all classes (pure helpers, rfe wiring, bart-fit contract, final bart, guard).

- [ ] **Step 5: Commit**

```bash
git add scripts/experiments/bart_launch_ratings.py tests/experiments/test_bart_launch_ratings.py
git commit -m "feat(bart): main() orchestration + CLI stages + 2025 coverage guard"
```

---

## Task 5: Live run + Run 30 (controller)

Heavy + live (BART RFE ~3 hr + final BART ~20-30 min). The controller runs this; not a subagent task.

**Files:**
- Create (commit): `data/predict_games/model_features_in/bart_rfe_features_brier_madden.csv`, `data/predict_games/bart_launch_ratings/results.json`
- Modify: `README.md`

- [ ] **Step 1: Confirm the parquet is Madden-built**

Run: `conda run --no-capture-output -n nfl-predictions python -c "import pandas as pd; from scripts.experiments import xgb_launch_ratings as x; df=pd.read_parquet('data/predict_games/input_data/schedule_and_weekly.parquet'); print('2025 cov', x.season_2025_ovr_coverage(df))"`
Expected: ~0.765. If absent/low, rebuild: `MADDEN_TOOLS_CDN_BASE=https://cdn.madden.tools python -m scripts.data_assembly.predict_game_winner.schedule_and_weekly`.

- [ ] **Step 2: Run BART backward-elimination (background — the long pole)**

Run: `conda run --no-capture-output -n nfl-predictions python -m scripts.experiments.bart_launch_ratings --stage rfe`
Watch `/tmp/bart_launch_rfe_progress.log`. Writes `bart_rfe_features_brier_madden.csv`. ~3 hr (278 → 10 features, ×6 replicates).

- [ ] **Step 3: Run final BART + eval (background)**

Run: `conda run --no-capture-output -n nfl-predictions python -m scripts.experiments.bart_launch_ratings --stage final`
Produces `data/predict_games/bart_launch_ratings/results.json`. ~20-30 min (4-chain).

- [ ] **Step 4: Record Run 30 in the README run-log**

Add the Run 30 table row + a narrative: hold-out ROC-AUC / accuracy / Brier; how many `madden_*` survived BART selection; the width-stratified Brier; and the honest verdict vs the BART champion (Run 10 0.708 / Run 6 0.705). **A null (Madden survives but doesn't beat 0.708) is reported plainly,** and compared to the Phase-1 XGBoost result (0.700).

- [ ] **Step 5: Commit**

```bash
git add data/predict_games/model_features_in/bart_rfe_features_brier_madden.csv \
        data/predict_games/bart_launch_ratings/results.json README.md
git commit -m "experiment(bart): Run 30 — BART with the full Madden family vs champion"
```

---

## Self-Review

**Spec coverage:**
- Start pool = champion-start + full Madden → Task 1 (`build_start_pool`) + Task 4 (`load_champion_start`, wired in `main`). ✓
- BART selection fit + backward elimination, brier-1SE → Task 2. ✓
- Final 4-chain BART + posterior std → Task 3. ✓
- `-100` sentinel + no-NaN guard → Task 1 (`to_bart_matrix`). ✓
- Eval via `bayes_logistic.evaluate` with `p_std` (width-stratified Brier) → Task 3 test + Task 4 `main`. ✓
- 2025 coverage guard → Task 4. ✓
- Run 30 README + honest verdict + outputs → Task 5. ✓
- Reuse (BartBackwardElimination, evaluate, madden_columns, assert_madden_2025_coverage) — imports, not reimplementation. ✓

**Placeholder scan:** No TBD/TODO. The only nuance (the `--stage final` standalone `history` fallback) is flagged with an explicit NOTE and the fallback code is shown. Every code/test step is complete.

**Type consistency:** `run_bart_rfe` returns `(best_features, history)`; `build_results(metrics, selected_features, history, best_num_feats)` consumes both; `run_final_bart` returns `(preds, p_std, holdout)`; `evaluate(..., p_std=p_std, weeks=...)` consumes them. `fit_fn` contract (`{'validation_score': float, 'variable_inclusion': pd.Series}`) is consistent between `make_bart_fit`, the stub test, and `BartBackwardElimination`'s requirement. `CHAMPIONS`/`META`/`BART_NAN_SENTINEL` names consistent across tasks and tests.
```
