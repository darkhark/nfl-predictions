# BART Backward-Elimination Feature Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimator-specific feature selection for BART — an iterative backward-elimination loop driven by BART's own `variable_inclusion` importances, re-measured each iteration, with parallel seed-replicate fits to tame PGBART sampler noise.

**Architecture:** A reusable, dependency-injected class in `data_science_utilities/models/bart/feature_selection/backward_elimination.py` — it knows nothing about PyMC; callers pass a `fit_fn(features, seed) -> {'validation_score', 'variable_inclusion'}`. Each iteration runs `replicates` fits in parallel threads (each PyMC fit spawns its own chain processes; 2 workers × 2 chains = 4 cores on the 14-core/48GB machine), averages validation scores and per-replicate inclusion RANK (rank-averaging, not raw-share averaging, so replicates with different inclusion scales weight equally), drops the bottom `drop_rate` fraction, and repeats. The experiment notebook starts from the 176-feature row of the run-13 RFE trace (gentle XGBoost pre-filter), selects by **validation-curve peak** (run-13 lesson: smallest-within-tolerance over-shrinks on flat curves), and reads the 2024+2025 hold-out exactly once for the winner (Run 14).

**Tech Stack:** Python 3.11 (conda env `nfl-predictions`), pandas, pymc/pymc-bart (notebook only — the class itself is pure pandas + concurrent.futures), unittest with a synthetic fit_fn (no PyMC in tests).

---

## Conventions

- Tests from repo root: `conda run -n nfl-predictions python -m unittest tests.data_science_utilities.test_bart_backward_elimination -v` (dotted module form; no `__init__.py` files, matching repo convention — note tests for `data_science_utilities` get a new `tests/data_science_utilities/` directory).
- Naming: the class is `BartBackwardElimination`; its history frame has columns `num_features`, `validation_score`, `replicate_scores`, `features`.
- Threading note: two concurrent `pm.sample` calls from threads work (pytensor compilation is lock-serialized; sampling subprocesses are independent), but the CLASS must not depend on that — `max_workers=1` must degrade gracefully to sequential. The notebook can drop to `max_workers=1` if concurrent sampling proves flaky on this machine.
- Hold-out discipline: the notebook touches `season >= 2024` exactly once, in the final cell, for the selected set only. All selection decisions use the 2022–2023 validation slice.

## File structure

| File | Action | Responsibility |
|---|---|---|
| `data_science_utilities/models/bart/feature_selection/backward_elimination.py` | Create | The loop: parallel replicate fits, rank-averaged inclusion, drop schedule, history, peak selection |
| `tests/data_science_utilities/test_bart_backward_elimination.py` | Create | Synthetic-fit_fn unit tests (no PyMC) |
| `notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/bart_rfe.ipynb` | Create | The Run 14 experiment |
| `README.md` | Modify | Run 14 row + bullet |

---

### Task 1: The backward-elimination class (TDD, no PyMC)

**Files:**
- Create: `data_science_utilities/models/bart/feature_selection/backward_elimination.py`
- Test: `tests/data_science_utilities/test_bart_backward_elimination.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/data_science_utilities/test_bart_backward_elimination.py`:

```python
import threading
import unittest

import pandas as pd

from data_science_utilities.models.bart.feature_selection.backward_elimination import (
    BartBackwardElimination,
)


def make_fit_fn(score_by_size=None):
    """A deterministic synthetic fit_fn. Importance is alphabetical: features earlier
    in the alphabet get HIGHER inclusion, so elimination order is fully predictable
    (z drops first). validation_score defaults to 0.6 + 0.001 * num_features unless
    score_by_size overrides a specific size. The seed nudges inclusion by a tiny,
    rank-preserving epsilon so replicate averaging is exercised without changing
    the elimination order."""
    calls = []

    def fit_fn(features, seed):
        calls.append((tuple(features), seed))
        ordered = sorted(features)  # alphabetical
        inclusion = pd.Series(
            {f: (len(ordered) - i) + seed * 1e-6 for i, f in enumerate(ordered)}
        )
        size = len(features)
        score = (score_by_size or {}).get(size, 0.6 + 0.001 * size)
        return {'validation_score': score, 'variable_inclusion': inclusion}

    fit_fn.calls = calls
    return fit_fn


FEATURES_8 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


class TestBartBackwardElimination(unittest.TestCase):

    def test_drop_schedule_and_history(self):
        fit_fn = make_fit_fn()
        rfe = BartBackwardElimination(fit_fn, drop_rate=0.25, min_features=3,
                                      replicates=2, max_workers=1)
        history = rfe.run(FEATURES_8)
        # 8 -> drop 2 -> 6 -> drop 1 (25% of 6 floored to 1, min 1) -> wait: 25% of 6
        # is 1.5 -> floor 1? The spec: drop max(1, floor(drop_rate * n)). 8->6->5->4->3.
        self.assertEqual(list(history['num_features']), [8, 6, 5, 4, 3])
        # alphabetical importance means the LAST letters drop first
        self.assertEqual(sorted(history.iloc[1]['features']),
                         ['a', 'b', 'c', 'd', 'e', 'f'])
        self.assertEqual(sorted(history.iloc[-1]['features']), ['a', 'b', 'c'])

    def test_stops_at_min_features(self):
        fit_fn = make_fit_fn()
        rfe = BartBackwardElimination(fit_fn, drop_rate=0.5, min_features=4,
                                      replicates=1, max_workers=1)
        history = rfe.run(FEATURES_8)
        # 8 -> 4, then stop (4 == min_features is evaluated, no further drop)
        self.assertEqual(list(history['num_features']), [8, 4])

    def test_replicates_get_distinct_seeds_and_scores_average(self):
        fit_fn = make_fit_fn(score_by_size={8: 0.7})
        rfe = BartBackwardElimination(fit_fn, drop_rate=0.5, min_features=4,
                                      replicates=3, max_workers=1, base_seed=100)
        history = rfe.run(FEATURES_8)
        first_iter_calls = [c for c in fit_fn.calls if len(c[0]) == 8]
        self.assertEqual(len(first_iter_calls), 3)
        self.assertEqual(len({seed for _, seed in first_iter_calls}), 3)
        self.assertAlmostEqual(history.iloc[0]['validation_score'], 0.7)
        self.assertEqual(len(history.iloc[0]['replicate_scores']), 3)

    def test_parallel_workers_produce_same_result_as_sequential(self):
        sequential = BartBackwardElimination(make_fit_fn(), drop_rate=0.25,
                                             min_features=3, replicates=2,
                                             max_workers=1).run(FEATURES_8)
        parallel = BartBackwardElimination(make_fit_fn(), drop_rate=0.25,
                                           min_features=3, replicates=2,
                                           max_workers=2).run(FEATURES_8)
        self.assertEqual(list(sequential['num_features']),
                         list(parallel['num_features']))
        for s_feats, p_feats in zip(sequential['features'], parallel['features']):
            self.assertEqual(sorted(s_feats), sorted(p_feats))

    def test_parallel_actually_runs_concurrently(self):
        # two replicates that block until both have started prove real concurrency
        barrier = threading.Barrier(2, timeout=10)

        def blocking_fit(features, seed):
            barrier.wait()  # deadlocks (then Barrier raises) unless 2 run at once
            ordered = sorted(features)
            return {'validation_score': 0.6,
                    'variable_inclusion': pd.Series(
                        {f: len(ordered) - i for i, f in enumerate(ordered)})}

        rfe = BartBackwardElimination(blocking_fit, drop_rate=0.5, min_features=4,
                                      replicates=2, max_workers=2)
        history = rfe.run(FEATURES_8)  # raises BrokenBarrierError if sequential
        self.assertEqual(list(history['num_features']), [8, 4])

    def test_rank_averaged_inclusion_decides_drops(self):
        # replicate seeds disagree on raw scale but agree on order -> order wins;
        # engineered case: one replicate's raw inclusion would mislead a raw average
        def fit_fn(features, seed):
            ordered = sorted(features)
            if seed % 2 == 0:
                # huge scale, alphabetical order
                inclusion = {f: (len(ordered) - i) * 1000 for i, f in enumerate(ordered)}
            else:
                # tiny scale, alphabetical order
                inclusion = {f: (len(ordered) - i) * 0.001 for i, f in enumerate(ordered)}
            return {'validation_score': 0.6,
                    'variable_inclusion': pd.Series(inclusion)}

        rfe = BartBackwardElimination(fit_fn, drop_rate=0.25, min_features=6,
                                      replicates=2, max_workers=1)
        history = rfe.run(FEATURES_8)
        # rank-averaging keeps the alphabetical order regardless of scale
        self.assertEqual(sorted(history.iloc[-1]['features']),
                         ['a', 'b', 'c', 'd', 'e', 'f'])

    def test_best_features_at_validation_peak(self):
        # run-13 lesson: select the PEAK of the validation curve, not the smallest
        # set within a tolerance
        fit_fn = make_fit_fn(score_by_size={8: 0.60, 6: 0.65, 5: 0.64, 4: 0.61, 3: 0.58})
        rfe = BartBackwardElimination(fit_fn, drop_rate=0.25, min_features=3,
                                      replicates=1, max_workers=1)
        rfe.run(FEATURES_8)
        best = rfe.get_best_features()
        self.assertEqual(len(best), 6)
        self.assertEqual(sorted(best), ['a', 'b', 'c', 'd', 'e', 'f'])

    def test_run_before_best_raises(self):
        rfe = BartBackwardElimination(make_fit_fn())
        with self.assertRaises(RuntimeError):
            rfe.get_best_features()


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n nfl-predictions python -m unittest tests.data_science_utilities.test_bart_backward_elimination -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_science_utilities.models.bart'`

- [ ] **Step 3: Write the implementation**

Create `data_science_utilities/models/bart/feature_selection/backward_elimination.py`:

```python
import math
from concurrent.futures import ThreadPoolExecutor

import pandas as pd


class BartBackwardElimination:
    """
    Iterative backward elimination driven by an injected fit function — built for BART,
    whose variable_inclusion importances (like any importance measure) are conditional
    on the candidate set and must be RE-MEASURED after every drop.

    fit_fn(features: list[str], seed: int) -> dict with:
        'validation_score': float (higher is better; e.g. ROC-AUC on a validation slice)
        'variable_inclusion': pd.Series indexed by feature name (higher = more used)

    Each iteration runs `replicates` fits with distinct seeds — in parallel threads when
    max_workers > 1 (each PyMC fit spawns its own chain subprocesses, so threads only
    coordinate; max_workers=1 degrades to sequential with identical results). Validation
    scores are averaged across replicates; importances are RANK-averaged (each
    replicate's inclusion converted to ranks before averaging) so replicates with
    different inclusion scales weight equally. The bottom max(1, floor(drop_rate * n))
    features by averaged rank are dropped each iteration until min_features is reached.

    Selection follows the validation-curve PEAK, not smallest-within-tolerance: on flat
    curves the tolerance rule over-shrinks, and small sets carry hold-out variance the
    validation score does not price (see the run-13 experiment-log entry).
    """

    def __init__(self, fit_fn, drop_rate=0.2, min_features=10, replicates=2,
                 max_workers=2, base_seed=32):
        self.fit_fn = fit_fn
        self.drop_rate = drop_rate
        self.min_features = min_features
        self.replicates = replicates
        self.max_workers = max_workers
        self.base_seed = base_seed
        self.history = None

    def run(self, features):
        """
        Run the elimination from the given starting feature list. Returns (and stores
        on self.history) a frame with one row per evaluated set: num_features,
        validation_score (replicate mean), replicate_scores (list), features (list).
        """
        features = list(features)
        rows = []
        iteration = 0
        while True:
            mean_score, replicate_scores, mean_ranks = self._evaluate(features, iteration)
            rows.append({
                'num_features': len(features),
                'validation_score': mean_score,
                'replicate_scores': replicate_scores,
                'features': list(features),
            })
            if len(features) <= self.min_features:
                break
            drop_count = max(1, math.floor(self.drop_rate * len(features)))
            drop_count = min(drop_count, len(features) - self.min_features)
            # mean_ranks: higher rank value = more important; drop the lowest
            features = list(mean_ranks.sort_values(ascending=False)
                            .head(len(features) - drop_count).index)
            iteration += 1
        self.history = pd.DataFrame(rows)
        return self.history

    def get_best_features(self):
        """The feature list at the peak of the replicate-mean validation curve."""
        if self.history is None:
            raise RuntimeError('call run() before get_best_features()')
        best_row = self.history.loc[self.history['validation_score'].idxmax()]
        return list(best_row['features'])

    def _evaluate(self, features, iteration):
        seeds = [self.base_seed + iteration * self.replicates + r
                 for r in range(self.replicates)]
        if self.max_workers > 1 and self.replicates > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                results = list(pool.map(
                    lambda seed: self.fit_fn(list(features), seed), seeds))
        else:
            results = [self.fit_fn(list(features), seed) for seed in seeds]

        scores = [r['validation_score'] for r in results]
        rank_frames = [r['variable_inclusion'].rank() for r in results]
        mean_ranks = pd.concat(rank_frames, axis=1).mean(axis=1)
        return sum(scores) / len(scores), scores, mean_ranks
```

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n nfl-predictions python -m unittest tests.data_science_utilities.test_bart_backward_elimination -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add data_science_utilities/models/bart tests/data_science_utilities
git commit -m "Add BART backward-elimination feature selection with parallel replicates"
```

---

### Task 2: The Run 14 experiment notebook

**Files:**
- Create: `notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/bart_rfe.ipynb`

Controller-executed (notebook authoring + long-running execution, like runs 9–13).

- [ ] **Step 1: Author the notebook** with these cells (paths relative to the notebook dir, matching the sibling notebooks):

Cell 0 (markdown): purpose, the selection-discipline note (validation slice decides; hold-out read once at the end), and the run-13 lesson reference.

Cell 1 (code): imports + config + data load:

```python
from concurrent.futures import ThreadPoolExecutor  # noqa: F401 (used inside the class)
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb
import os, sys
module_path = os.path.abspath(os.path.join('../../../../../'))
if module_path not in sys.path:
    sys.path.append(module_path)
from data_science_utilities.models.bart.feature_selection.backward_elimination import (
    BartBackwardElimination,
)

RANDOM_SEED = 32
TARGET = 'target_win'
BART_NAN_SENTINEL = -100.0
START_POOL_SIZE = 176  # generous XGBoost pre-filter row from the run-13 RFE trace

START_FEATURES = list(pd.read_csv(
    '../../../../../data/predict_games/model_features_in/rfe_features_kfolds.csv',
    index_col=0,
).loc[START_POOL_SIZE].dropna().values)

MODEL_INPUTS_DF = pd.read_parquet(
    '../../../../../data/predict_games/input_data/schedule_and_weekly.parquet'
).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

train_df = MODEL_INPUTS_DF[MODEL_INPUTS_DF['season'] < 2022]
valid_df = MODEL_INPUTS_DF[
    (MODEL_INPUTS_DF['season'] >= 2022) & (MODEL_INPUTS_DF['season'] < 2024)
]
y_train = train_df[TARGET].to_numpy(dtype=int)
y_valid = valid_df[TARGET].to_numpy(dtype=int)
```

Cell 2 (code): the fit function (2 chains for selection speed; sentinel fill + no-NaN asserts per the bart.ipynb convention):

```python
def bart_fit(features, seed):
    """One BART fit: train on <2022, score ROC-AUC on the 2022-2023 validation slice,
    return the chain-averaged variable_inclusion. 2 chains / 2 cores per fit so two
    replicate fits can run in parallel on this machine."""
    X_train = train_df[features].fillna(BART_NAN_SENTINEL).to_numpy(dtype=float)
    X_valid = valid_df[features].fillna(BART_NAN_SENTINEL).to_numpy(dtype=float)
    assert not np.isnan(X_train).any() and not np.isnan(X_valid).any()

    with pm.Model() as model:
        X_data = pm.Data('X', X_train)
        mu = pmb.BART('mu', X_data, y_train, m=50)
        p = pm.Deterministic('p', pm.math.invprobit(mu))
        pm.Bernoulli('y', p=p, observed=y_train, shape=mu.shape)
        idata = pm.sample(draws=1000, tune=1000, chains=2, cores=2,
                          random_seed=seed, progressbar=False)
        pm.set_data({'X': X_valid})
        ppc = pm.sample_posterior_predictive(
            idata, var_names=['p'], random_seed=seed, progressbar=False)

    valid_preds = ppc.posterior_predictive['p'].mean(dim=['chain', 'draw']).to_numpy()
    inclusion = pd.Series(
        idata.sample_stats['variable_inclusion']
        .mean(dim=['chain', 'draw']).to_numpy(),
        index=features,
    )
    return {'validation_score': roc_auc_score(y_valid, valid_preds),
            'variable_inclusion': inclusion}
```

Cell 3 (code): run the loop and show the curve:

```python
rfe = BartBackwardElimination(bart_fit, drop_rate=0.2, min_features=10,
                              replicates=2, max_workers=2, base_seed=RANDOM_SEED)
history = rfe.run(START_FEATURES)
history[['num_features', 'validation_score', 'replicate_scores']]
```

Cell 4 (code): plot validation curve (matplotlib, num_features x-axis reversed) and print `rfe.get_best_features()` count.

Cell 5 (code): save the winner:

```python
best_features = rfe.get_best_features()
pd.DataFrame({'feature': best_features}).to_csv(
    '../../../../../data/predict_games/model_features_in/bart_rfe_features.csv',
    index=False,
)
len(best_features)
```

Cell 6 (code): **the single hold-out read** — final 4-chain fit on the winner, evaluated exactly as bart.ipynb does (pooled AUROC, accuracy, Brier, log loss, width-stratified Brier).

- [ ] **Step 2: Execute** with the verification protocol (no pipes without pipefail; `&& echo NBCONVERT_OK`; afterwards confirm the printed train shape matches the selected count and the history table is populated). Runtime estimate: ~11 iterations × ~90s wall (2 replicates in parallel) + final 4-chain fit ≈ 20–25 min.

- [ ] **Step 3: Record Run 14** in README (table row + changelog bullet: starting pool, curve shape, selected count, validation peak, hold-out result vs run 10 champion and run 12) and commit notebook + CSV + README.

### Task 3: PR

Push `extend-rfe-and-bart-rfe`; `gh pr create --base master` titled "Extended RFE iterations + BART backward-elimination selection (runs 13-14)"; body summarizes the run-13 methodology lesson, the new class, and run-14 results; ends with the Claude Code attribution line.
