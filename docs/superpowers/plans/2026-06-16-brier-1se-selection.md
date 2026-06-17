# Brier + 1-SE Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the feature-selection metric configurable (default Brier instead of ROC-AUC) and replace the over-shrinking 0.005 tolerance with a direction-aware 1-SE rule, in both the XGBoost RFE and BART backward-elimination classes; then re-run both pools × both estimators and record the results.

**Architecture:** Two small, tested additions to the `data_science_utilities` selection classes (a `brier` metric + per-fold score storage + a `get_best_num_features_1se()` on the XGBoost RFE; a `get_best_features_1se()` on `BartBackwardElimination`), then a single `SELECTION_METRIC` knob wired through three notebooks. Elimination *path* is unchanged — only count-selection changes.

**Tech Stack:** Python, scikit-learn (`brier_score_loss`), XGBoost, PyMC/pymc-bart, unittest (run via `python -m unittest`).

---

## Conventions

- Repo root: `/Users/joshuaharkness/ClaudeProjects/nfl-predictions` (abbreviated **ROOT**). `cd` there first.
- **Env Python** (abbreviated **EPY**): `/opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/python` (the bare `python3` lacks pandas/sklearn/pytest). There is **no pytest** — tests are `unittest`.
- **Run a single test file** (the `tests/` tree has no `__init__.py`, so `discover -p`
  finds 0 tests — use the dotted module path instead):
  ```bash
  cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
  PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/python \
    -m unittest tests.data_science_utilities.<test_file_without_.py> -v
  ```
  (`-m unittest` puts ROOT on `sys.path` so `import data_science_utilities` resolves.)
  Full suite: `... -m unittest discover -s tests -v` (discover works for a full run).
- **Notebook cells** are edited with **NotebookEdit** (`cell_id` = the `id` from a Read of the notebook). Read the notebook before editing.
- **Headless notebook runs** (the canonical form from the Run-15 plan):
  ```bash
  cd notebooks/model_training/predict_games/schedule_and_weekly/cross_validation
  PYTHONPATH=/Users/joshuaharkness/ClaudeProjects/nfl-predictions \
    /opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/jupyter nbconvert \
    --to notebook --execute --inplace <nb>.ipynb \
    --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1 > /tmp/<log>.out 2>&1
  ```
  Run long ones with `run_in_background: true`; keep the Mac on AC + `caffeinate -dimsu -w <nbconvert_pid>`.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `data_science_utilities/models/xgb/feature_selection/recursive/classifier_cross_validation.py` | XGBoost CV-RFE | Add `brier` metric, per-fold storage, `get_best_num_features_1se()` |
| `data_science_utilities/models/bart/feature_selection/backward_elimination.py` | BART backward elimination | Add `get_best_features_1se()` |
| `tests/data_science_utilities/test_xgb_rfe_min_features.py` | XGBoost RFE tests | Add brier + per-fold + 1-SE tests |
| `tests/data_science_utilities/test_bart_backward_elimination.py` | BART tests | Add 1-SE tests |
| `.../cross_validation/rfe.ipynb` | XGBoost RFE driver | `SELECTION_METRIC`, 1-SE select |
| `.../cross_validation/grid_search.ipynb` | XGBoost tune + hold-out | scoring map, `BEST_NUM_FEATS` from 1-SE |
| `.../cross_validation/bart_rfe.ipynb` | BART RFE driver | `SELECTION_METRIC`, Brier fit, 1-SE select, plot |

---

## Task 1: XGBoost RFE — add the `brier` metric

**Files:**
- Modify: `data_science_utilities/models/xgb/feature_selection/recursive/classifier_cross_validation.py`
- Test: `tests/data_science_utilities/test_xgb_rfe_min_features.py`

- [ ] **Step 1: Write the failing test** — append to the test file (inside a new class):
```python
class TestBrierMetric(unittest.TestCase):

    def test_brier_metric_runs_and_scores_are_probabilities(self):
        X, y = make_synthetic_frame()
        rfe = ClassifierCrossValidationRecursiveFeatureSelection(
            X, y, dict(FAST_XGB_PARAMS), model_score_metric='brier'
        )
        rfe.get_optimal_features_no_grouped_records(
            drop_rate=0.3, max_iter=3, n_folds=3
        )
        # brier_score_loss is in [0, 1]; a real model beats the 0.25 coin-flip ceiling loosely
        self.assertTrue(all(0.0 <= s <= 1.0 for s in rfe.all_model_scores))
        self.assertEqual(len(rfe.all_model_scores), 3)

    def test_brier_is_treated_as_lower_is_better(self):
        rfe = ClassifierCrossValidationRecursiveFeatureSelection(
            *make_synthetic_frame(), dict(FAST_XGB_PARAMS), model_score_metric='brier'
        )
        self.assertNotIn('brier', rfe.HIGHER_IS_BETTER_METRICS)
```

- [ ] **Step 2: Run it, verify it fails**
```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/python \
  -m unittest discover -s tests -p "test_xgb_rfe_min_features.py" -v
```
Expected: `TestBrierMetric.test_brier_metric_runs_and_scores_are_probabilities` FAILS with `ValueError: model_score_metric must be one of roc_auc, log_loss, ...` (brier not handled).

- [ ] **Step 3: Implement** — in `classifier_cross_validation.py`:
  - Extend the import (line 1-4) to include `brier_score_loss`:
```python
from sklearn.metrics import (
    roc_auc_score, log_loss, f1_score, precision_score,
    recall_score, accuracy_score, brier_score_loss
)
```
  - In `_get_test_scores`, add a branch after the `log_loss` branch:
```python
        elif self.model_score_metric == 'brier':
            score = brier_score_loss(y_test, preds)
```
  - Update the `ValueError` message to:
```python
            raise ValueError('model_score_metric must be one of roc_auc, log_loss, brier, f1, precision, recall, accuracy')
```

- [ ] **Step 4: Run tests, verify pass** (same command as Step 2). Expected: all tests in the file PASS.

- [ ] **Step 5: Commit**
```bash
git add data_science_utilities/models/xgb/feature_selection/recursive/classifier_cross_validation.py \
        tests/data_science_utilities/test_xgb_rfe_min_features.py
git commit -m "XGBoost RFE: add brier metric (lower-is-better) to _get_test_scores"
```

---

## Task 2: XGBoost RFE — store per-fold CV scores

**Files:**
- Modify: `classifier_cross_validation.py`
- Test: `tests/data_science_utilities/test_xgb_rfe_min_features.py`

- [ ] **Step 1: Write the failing test** — append a new class:
```python
class TestPerFoldStorage(unittest.TestCase):

    def test_per_fold_scores_stored_and_consistent_with_means(self):
        X, y = make_synthetic_frame()
        rfe = ClassifierCrossValidationRecursiveFeatureSelection(X, y, dict(FAST_XGB_PARAMS))
        rfe.get_optimal_features_no_grouped_records(drop_rate=0.3, max_iter=3, n_folds=3)
        # one fold-list per evaluated size, aligned with the means
        self.assertEqual(len(rfe.all_model_score_folds), len(rfe.all_model_scores))
        for folds, mean in zip(rfe.all_model_score_folds, rfe.all_model_scores):
            self.assertEqual(len(folds), 3)            # n_folds
            self.assertAlmostEqual(sum(folds) / len(folds), mean)
```

- [ ] **Step 2: Run it, verify it fails** (same command as Task 1 Step 2). Expected: `AttributeError: 'ClassifierCrossValidationRecursiveFeatureSelection' object has no attribute 'all_model_score_folds'`.

- [ ] **Step 3: Implement** — in `classifier_cross_validation.py`:
  - In `__init__`, after `self.all_model_scores = []` add:
```python
        self.all_model_score_folds = []
```
  - In `get_optimal_features_no_grouped_records`, immediately after the existing
    `self.all_model_scores.append(np.mean(model_scores))` line, add:
```python
            self.all_model_score_folds.append(list(model_scores))
```

- [ ] **Step 4: Run tests, verify pass** (same command). Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add data_science_utilities/models/xgb/feature_selection/recursive/classifier_cross_validation.py \
        tests/data_science_utilities/test_xgb_rfe_min_features.py
git commit -m "XGBoost RFE: store per-fold CV scores (all_model_score_folds) for 1-SE rule"
```

---

## Task 3: XGBoost RFE — `get_best_num_features_1se()`

**Files:**
- Modify: `classifier_cross_validation.py`
- Test: `tests/data_science_utilities/test_xgb_rfe_min_features.py`

- [ ] **Step 1: Write the failing test** — append a new class. These set the stored
  attributes by hand (no real fitting) to test the selection logic deterministically:
```python
class TestOneSESelection(unittest.TestCase):

    def _rfe(self, metric):
        X, y = make_synthetic_frame()
        return ClassifierCrossValidationRecursiveFeatureSelection(
            X, y, dict(FAST_XGB_PARAMS), model_score_metric=metric
        )

    def test_lower_is_better_picks_parsimonious_within_band(self):
        rfe = self._rfe('brier')
        rfe.all_features = {10: ['f'] * 10, 7: ['f'] * 7, 5: ['f'] * 5, 3: ['f'] * 3}
        rfe.all_model_scores = [0.200, 0.180, 0.181, 0.230]   # best (min) at size 7
        rfe.all_model_score_folds = [
            [0.200, 0.200, 0.200],
            [0.178, 0.180, 0.182],   # SE = std(ddof=1)/sqrt(3) ~= 0.00115; band <= 0.18115
            [0.181, 0.181, 0.181],   # 0.181 <= band -> the more parsimonious size 5 qualifies
            [0.230, 0.230, 0.230],
        ]
        self.assertEqual(rfe.get_best_num_features_1se(), 5)

    def test_tight_band_returns_the_optimum(self):
        rfe = self._rfe('brier')
        rfe.all_features = {10: ['f'] * 10, 7: ['f'] * 7, 5: ['f'] * 5}
        rfe.all_model_scores = [0.200, 0.180, 0.190]
        rfe.all_model_score_folds = [
            [0.200, 0.200, 0.200],
            [0.180, 0.180, 0.180],   # SE = 0 -> band is a point; only size 7 qualifies
            [0.190, 0.190, 0.190],
        ]
        self.assertEqual(rfe.get_best_num_features_1se(), 7)

    def test_higher_is_better_direction(self):
        rfe = self._rfe('roc_auc')
        rfe.all_features = {10: ['f'] * 10, 7: ['f'] * 7, 5: ['f'] * 5, 3: ['f'] * 3}
        rfe.all_model_scores = [0.690, 0.710, 0.709, 0.660]   # best (max) at size 7
        rfe.all_model_score_folds = [
            [0.690, 0.690, 0.690],
            [0.708, 0.710, 0.712],   # SE ~= 0.00115; band >= 0.70885
            [0.709, 0.709, 0.709],   # 0.709 >= band -> parsimonious size 5 qualifies
            [0.660, 0.660, 0.660],
        ]
        self.assertEqual(rfe.get_best_num_features_1se(), 5)
```

- [ ] **Step 2: Run it, verify it fails** (same command). Expected: `AttributeError: ... has no attribute 'get_best_num_features_1se'`.

- [ ] **Step 3: Implement** — add this method to the class (after `get_best_num_features`):
```python
    def get_best_num_features_1se(self):
        """Most parsimonious feature count within one standard error of the best CV
        score. SE is the standard error across folds at the best-scoring count.
        Direction-aware via HIGHER_IS_BETTER_METRICS, so it works for any metric
        (roc_auc, brier, log_loss, ...). Replaces the smallest-within-tolerance rule,
        which over-shrinks on flat curves (run-13 lesson)."""
        sizes = list(self.all_features.keys())
        means = list(self.all_model_scores)
        folds = list(self.all_model_score_folds)
        higher_is_better = self.model_score_metric in self.HIGHER_IS_BETTER_METRICS
        best_i = (max if higher_is_better else min)(
            range(len(means)), key=lambda i: means[i]
        )
        best_folds = folds[best_i]
        se = (float(np.std(best_folds, ddof=1) / np.sqrt(len(best_folds)))
              if len(best_folds) > 1 else 0.0)
        if higher_is_better:
            within = [sizes[i] for i in range(len(means)) if means[i] >= means[best_i] - se]
        else:
            within = [sizes[i] for i in range(len(means)) if means[i] <= means[best_i] + se]
        return min(within)
```

- [ ] **Step 4: Run tests, verify pass** (same command). Expected: all PASS.

- [ ] **Step 5: Commit**
```bash
git add data_science_utilities/models/xgb/feature_selection/recursive/classifier_cross_validation.py \
        tests/data_science_utilities/test_xgb_rfe_min_features.py
git commit -m "XGBoost RFE: add direction-aware get_best_num_features_1se()"
```

---

## Task 4: BART RFE — `get_best_features_1se()`

**Files:**
- Modify: `data_science_utilities/models/bart/feature_selection/backward_elimination.py`
- Test: `tests/data_science_utilities/test_bart_backward_elimination.py`

- [ ] **Step 1: Write the failing test** — append a new class (build `history` by hand
  for deterministic control of replicate spread):
```python
class TestBartOneSE(unittest.TestCase):

    def _history(self):
        return pd.DataFrame([
            {'num_features': 8, 'validation_score': 0.200,
             'replicate_scores': [0.200, 0.200, 0.200],
             'features': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']},
            {'num_features': 6, 'validation_score': 0.180,
             'replicate_scores': [0.178, 0.180, 0.182],   # SE ~= 0.00115; band <= 0.18115
             'features': ['a', 'b', 'c', 'd', 'e', 'f']},
            {'num_features': 5, 'validation_score': 0.181,
             'replicate_scores': [0.181, 0.181, 0.181],   # within band -> parsimonious wins
             'features': ['a', 'b', 'c', 'd', 'e']},
            {'num_features': 3, 'validation_score': 0.230,
             'replicate_scores': [0.230, 0.230, 0.230],
             'features': ['a', 'b', 'c']},
        ])

    def test_lower_is_better_picks_parsimonious_within_band(self):
        rfe = BartBackwardElimination(make_fit_fn())
        rfe.history = self._history()
        best = rfe.get_best_features_1se(higher_is_better=False)
        self.assertEqual(sorted(best), ['a', 'b', 'c', 'd', 'e'])   # 5 features

    def test_tight_band_returns_optimum(self):
        rfe = BartBackwardElimination(make_fit_fn())
        df = self._history()
        df.at[1, 'replicate_scores'] = [0.180, 0.180, 0.180]        # SE = 0 at the optimum
        rfe.history = df
        best = rfe.get_best_features_1se(higher_is_better=False)
        self.assertEqual(sorted(best), ['a', 'b', 'c', 'd', 'e', 'f'])  # the 6-feature optimum

    def test_higher_is_better_direction(self):
        rfe = BartBackwardElimination(make_fit_fn())
        rfe.history = pd.DataFrame([
            {'num_features': 8, 'validation_score': 0.690,
             'replicate_scores': [0.690, 0.690, 0.690], 'features': list('abcdefgh')},
            {'num_features': 6, 'validation_score': 0.710,
             'replicate_scores': [0.708, 0.710, 0.712], 'features': list('abcdef')},
            {'num_features': 5, 'validation_score': 0.709,
             'replicate_scores': [0.709, 0.709, 0.709], 'features': list('abcde')},
        ])
        best = rfe.get_best_features_1se(higher_is_better=True)
        self.assertEqual(sorted(best), list('abcde'))               # 5 features

    def test_run_before_1se_raises(self):
        rfe = BartBackwardElimination(make_fit_fn())
        with self.assertRaises(RuntimeError):
            rfe.get_best_features_1se()
```

- [ ] **Step 2: Run it, verify it fails**
```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/python \
  -m unittest tests.data_science_utilities.test_bart_backward_elimination -v
```
Expected: `AttributeError: 'BartBackwardElimination' object has no attribute 'get_best_features_1se'`.

- [ ] **Step 3: Implement** — in `backward_elimination.py`:
  - Add `import numpy as np` near the top (after `import pandas as pd`).
  - Add this method after `get_best_features`:
```python
    def get_best_features_1se(self, higher_is_better=False):
        """Most parsimonious feature set within one standard error of the best
        validation score. SE is the standard error across replicate scores at the
        best-scoring set. Pass higher_is_better=False for lower-is-better metrics
        (e.g. Brier), True for higher-is-better (e.g. ROC-AUC).

        Caveat: BART's RFE has no cross-validation -- the replicates are sampler-seed
        reruns on a fixed validation slice, so this SE captures SAMPLER noise (~0.003),
        not data/generalization variance. The band is therefore tight (near-peak)."""
        if self.history is None:
            raise RuntimeError('call run() before get_best_features_1se()')
        means = self.history['validation_score'].to_numpy()
        best_i = int(np.argmax(means) if higher_is_better else np.argmin(means))
        reps = list(self.history.iloc[best_i]['replicate_scores'])
        se = (float(np.std(reps, ddof=1) / np.sqrt(len(reps))) if len(reps) > 1 else 0.0)
        if higher_is_better:
            within = self.history[self.history['validation_score'] >= means[best_i] - se]
        else:
            within = self.history[self.history['validation_score'] <= means[best_i] + se]
        best_row = within.loc[within['num_features'].idxmin()]
        return list(best_row['features'])
```
  - Update the class docstring's "Selection follows the validation-curve PEAK" paragraph
    to add: "`get_best_features_1se()` offers a direction-aware 1-SE variant; see its
    docstring for the sampler-noise caveat."

- [ ] **Step 4: Run tests, verify pass** (same command as Step 2). Expected: all PASS.

- [ ] **Step 5: Run the FULL suite to confirm no regressions**
```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/python \
  -m unittest discover -s tests -v
```
Expected: all existing + new tests PASS.

- [ ] **Step 6: Commit**
```bash
git add data_science_utilities/models/bart/feature_selection/backward_elimination.py \
        tests/data_science_utilities/test_bart_backward_elimination.py
git commit -m "BART RFE: add direction-aware get_best_features_1se() (replicate-SE band)"
```

---

## Task 5: Wire `rfe.ipynb` to `SELECTION_METRIC` + 1-SE

**Files:** Modify `.../cross_validation/rfe.ipynb`

- [ ] **Step 1: Inspect cells** (get cell ids)
```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
python3 -c "
import json
nb=json.load(open('notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/rfe.ipynb'))
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code':
        s=''.join(c['source'])
        if 'RFE_XGB_PARAMS' in s or 'classifier(' in s or 'get_best_num_features' in s:
            print('=== cell',i,'===', repr(c.get('id'))); print(s); print()
"
```

- [ ] **Step 2: Add `SELECTION_METRIC`** — NotebookEdit the params cell (the one defining
  `RFE_XGB_PARAMS`, cell index 1), appending at the end of its source:
```python
# Selection metric: 'brier' (proper scoring rule, default) | 'roc_auc' | 'log_loss'.
# Drives the RFE scoring metric, the 1-SE selection direction, and grid_search scoring.
SELECTION_METRIC = 'brier'
```

- [ ] **Step 3: Use the metric in the classifier** — NotebookEdit the RFE cell (the one
  calling `classifier(... model_score_metric='roc_auc')`, cell index 9): change
  `model_score_metric='roc_auc'` to `model_score_metric=SELECTION_METRIC`. Leave the rest
  (the `_log_progress` callback, `max_iter=60`, etc.) unchanged.

- [ ] **Step 4: Switch selection to 1-SE** — NotebookEdit the best-count cell (the one with
  `best_num_feats = rfe.get_best_num_features(.005)`, cell index 12) to:
```python
best_num_feats = rfe.get_best_num_features_1se()
print(f'1-SE selected feature count ({SELECTION_METRIC}):', best_num_feats)
best_num_feats
```

- [ ] **Step 5: Verify the edits landed** (re-run Step 1's inspection); confirm
  `SELECTION_METRIC = 'brier'`, `model_score_metric=SELECTION_METRIC`,
  `get_best_num_features_1se()`. Do NOT execute the notebook here.

- [ ] **Step 6: Commit**
```bash
git add notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/rfe.ipynb
git commit -m "rfe.ipynb: configurable SELECTION_METRIC (default brier) + 1-SE selection"
```

---

## Task 6: Wire `grid_search.ipynb` to the metric

**Files:** Modify `.../cross_validation/grid_search.ipynb`

- [ ] **Step 1: Inspect the config cell** (cell index 1; get its id)
```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
python3 -c "
import json
nb=json.load(open('notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/grid_search.ipynb'))
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code' and ('RANDOM_SEARCH_PARAMS' in ''.join(c['source']) or 'BEST_NUM_FEATS' in ''.join(c['source'])):
        print('=== cell',i,'===', repr(c.get('id'))); print(''.join(c['source'])); print()
"
```

- [ ] **Step 2: Add the metric + scoring map** — NotebookEdit the config cell (index 1).
  Change the `BEST_NUM_FEATS = 31` line and the `RANDOM_SEARCH_PARAMS` `scoring=` value.
  Add near the top of the cell:
```python
# Must match rfe.ipynb's SELECTION_METRIC. Maps to the sklearn scoring name for tuning.
SELECTION_METRIC = 'brier'
_SCORING = {'brier': 'neg_brier_score', 'roc_auc': 'roc_auc', 'log_loss': 'neg_log_loss'}[SELECTION_METRIC]
```
  and change `RANDOM_SEARCH_PARAMS`'s `scoring='roc_auc'` to `scoring=_SCORING`. Leave
  `BEST_NUM_FEATS` as an int the engineer will set in Step 3 of the run tasks (it comes from
  rfe.ipynb's 1-SE output); the `SELECTED_RFE_CSV` path is also set per-run there.

- [ ] **Step 3: Verify** (re-run Step 1 inspection); confirm `SELECTION_METRIC`, `_SCORING`,
  and `scoring=_SCORING`. Do NOT execute.

- [ ] **Step 4: Commit**
```bash
git add notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/grid_search.ipynb
git commit -m "grid_search.ipynb: derive hyperparameter scoring from SELECTION_METRIC"
```

---

## Task 7: Wire `bart_rfe.ipynb` to the metric + 1-SE

**Files:** Modify `.../cross_validation/bart_rfe.ipynb`

- [ ] **Step 1: Inspect cells** (config cell `54edd439`, fit cell `60c2138e`, plot cell
  `d86e1852`)
```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
python3 -c "
import json
nb=json.load(open('notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/bart_rfe.ipynb'))
for i,c in enumerate(nb['cells']):
    s=''.join(c['source'])
    if c['cell_type']=='code' and ('START_POOL_SIZE' in s or 'def bart_fit' in s or 'get_best_features' in s):
        print('=== cell',i,'===', repr(c.get('id'))); print(s[:400]); print('...')
"
```

- [ ] **Step 2: Add `SELECTION_METRIC` + scorer map** — NotebookEdit the config cell
  (`54edd439`); add after the `BART_NAN_SENTINEL = -100.0` line:
```python
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss as _log_loss_fn
# Selection metric: 'brier' (default) | 'roc_auc' | 'log_loss'. _SCORER maps preds->score;
# _HIGHER_IS_BETTER drives the 1-SE direction. roc_auc higher-better; brier/log_loss lower.
SELECTION_METRIC = 'brier'
_SCORER = {'brier': brier_score_loss, 'roc_auc': roc_auc_score, 'log_loss': _log_loss_fn}[SELECTION_METRIC]
_HIGHER_IS_BETTER = SELECTION_METRIC == 'roc_auc'
```
  (Keep the existing `START_POOL_SIZE` and everything else in the cell.)

- [ ] **Step 3: Use the scorer in `bart_fit`** — NotebookEdit the fit cell (`60c2138e`):
  change the line `score = roc_auc_score(y_valid, valid_preds)` to:
```python
    score = _SCORER(y_valid, valid_preds)
```
  and update the `log_progress(...)` line in that cell to label it generically, e.g.
  `f'{SELECTION_METRIC} {score:.4f}'` instead of `f'valid auroc {score:.4f}'`.

- [ ] **Step 4: Switch selection to 1-SE + fix the plot** — NotebookEdit the plot/select
  cell (`d86e1852`) to:
```python
plt.plot(history['num_features'], history['validation_score'], marker='o')
plt.gca().invert_xaxis()
_best_score = history['validation_score'].max() if _HIGHER_IS_BETTER else history['validation_score'].min()
plt.axhline(_best_score, color='r', linestyle='--')
plt.xlabel('features')
plt.ylabel(f'validation {SELECTION_METRIC} (2022-2023)')
best_features = rfe.get_best_features_1se(higher_is_better=_HIGHER_IS_BETTER)
print('1-SE selected', len(best_features), 'features,',
      f"best {SELECTION_METRIC} {_best_score:.4f}")
```

- [ ] **Step 5: Verify** (re-run Step 1 inspection); confirm `SELECTION_METRIC`, `_SCORER`
  in `bart_fit`, and `get_best_features_1se`. Do NOT execute.

- [ ] **Step 6: Commit**
```bash
git add notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/bart_rfe.ipynb
git commit -m "bart_rfe.ipynb: configurable SELECTION_METRIC (default brier) + 1-SE selection"
```

---

## Task 8: Run A — XGBoost rank-only, Brier+1-SE

**Files:** executes `rfe.ipynb` + `grid_search.ipynb`; writes `rfe_features_kfolds_brier.csv`

- [ ] **Step 1: Set rank-only + brier output** — NotebookEdit `rfe.ipynb`: set
  `RANK_ONLY = True` (cell 5) and change cell 11's output filename so RANK_ONLY=True writes
  `rfe_features_kfolds_brier.csv` (not the AUROC champion `rfe_features_kfolds.csv`). Make
  cell 11:
```python
features_df = rfe.get_features_in_dataframe()
_suffix = '' if SELECTION_METRIC == 'roc_auc' else f'_{SELECTION_METRIC}'
_pool = '' if RANK_ONLY else '_full'
features_df.to_csv(f'../../../../../data/predict_games/model_features_in/rfe_features_kfolds{_pool}{_suffix}.csv')
features_df
```
  Confirm `SELECTION_METRIC = 'brier'`.

- [ ] **Step 2: Run `rfe.ipynb` headless** (background; canonical command, log
  `/tmp/xgb_rfe_brier_ro.out`). Verify clean exit, then capture the 1-SE count:
```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
python3 -c "
import json
nb=json.load(open('notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/rfe.ipynb'))
for c in nb['cells']:
    if c['cell_type']=='code' and 'get_best_num_features_1se' in ''.join(c['source']):
        for o in c.get('outputs',[]): print(''.join(o.get('text','')) or (o.get('data',{}) or {}).get('text/plain'))
"
```
Record the printed count as **A_BEST**. Confirm `rfe_features_kfolds_brier.csv` exists and its index max is ~1529 (rank-only full candidate set), proving the rank-only brier RFE ran.

- [ ] **Step 3: Point `grid_search.ipynb` at this run** — NotebookEdit `grid_search.ipynb`
  config cell: `SELECTION_METRIC = 'brier'`, `SELECTED_RFE_CSV = 'rfe_features_kfolds_brier.csv'`,
  `BEST_NUM_FEATS = <A_BEST>`.

- [ ] **Step 4: Run `grid_search.ipynb` headless** (background, log `/tmp/xgb_grid_brier_ro.out`).
  Verify clean exit; capture the overall hold-out AUROC + accuracy + Brier + log-loss lines
  from the executed notebook (the `2024+2025 hold-out ROC-AUC (overall)` print and the Brier
  cell). Record as **Run A results**.

- [ ] **Step 5: Commit**
```bash
git add notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/rfe.ipynb \
        notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/grid_search.ipynb \
        data/predict_games/model_features_in/rfe_features_kfolds_brier.csv
git commit -m "Run A: XGBoost rank-only, Brier+1-SE selection (<A_BEST> feats) hold-out <metrics>"
```

---

## Task 9: Run B — XGBoost full set, Brier+1-SE

**Files:** executes `rfe.ipynb` + `grid_search.ipynb`; writes `rfe_features_kfolds_full_brier.csv`

- [ ] **Step 1: Set full + brier** — NotebookEdit `rfe.ipynb` cell 5: `RANK_ONLY = False`
  (cell 11's filename logic from Task 8 Step 1 already routes to `..._full_brier.csv`).

- [ ] **Step 2: Run `rfe.ipynb` headless** (background, log `/tmp/xgb_rfe_brier_full.out`).
  Verify clean exit; confirm `rfe_features_kfolds_full_brier.csv` index max ~2348. Capture
  the 1-SE count → **B_BEST**.

- [ ] **Step 3: Point grid_search at it** — NotebookEdit `grid_search.ipynb`:
  `SELECTED_RFE_CSV = 'rfe_features_kfolds_full_brier.csv'`, `BEST_NUM_FEATS = <B_BEST>`.

- [ ] **Step 4: Run `grid_search.ipynb` headless** (background, log `/tmp/xgb_grid_brier_full.out`);
  capture hold-out AUROC/acc/Brier/log-loss → **Run B results**.

- [ ] **Step 5: Commit**
```bash
git add notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/rfe.ipynb \
        notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/grid_search.ipynb \
        data/predict_games/model_features_in/rfe_features_kfolds_full_brier.csv
git commit -m "Run B: XGBoost full set, Brier+1-SE selection (<B_BEST> feats) hold-out <metrics>"
```

---

## Task 10: Run C — BART rank-only, Brier+1-SE

**Files:** executes `bart_rfe.ipynb`; writes `bart_rfe_features_brier.csv`

- [ ] **Step 1: Point BART at the rank-only brier trace** — NotebookEdit `bart_rfe.ipynb`
  config cell (`54edd439`): confirm `SELECTION_METRIC = 'brier'`; set the
  `START_FEATURES = pd.read_csv(...)` source to `rfe_features_kfolds_brier.csv` and
  `START_POOL_SIZE` to the produced row nearest ~91 (find it the way Run 15 did:
  `min(index, key=lambda n: abs(n-91))`). NotebookEdit the output cell (`28b20867`) to write
  `bart_rfe_features_brier.csv`.

- [ ] **Step 2: Run headless** (background, log `/tmp/bart_rfe_brier_ro.out`; caffeinate on).
  Verify clean exit + `START run` echo. Capture the final-cell hold-out AUROC/acc/Brier/log-loss
  and the 1-SE selected count → **Run C results**.

- [ ] **Step 3: Commit**
```bash
git add notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/bart_rfe.ipynb \
        data/predict_games/model_features_in/bart_rfe_features_brier.csv
git commit -m "Run C: BART rank-only, Brier+1-SE selection hold-out <metrics>"
```

---

## Task 11: Run D — BART full set, Brier+1-SE

**Files:** executes `bart_rfe.ipynb`; writes `bart_rfe_features_full_brier.csv`

- [ ] **Step 1: Point BART at the full brier trace** — NotebookEdit `bart_rfe.ipynb` config
  cell: set `START_FEATURES` source to `rfe_features_kfolds_full_brier.csv`,
  `START_POOL_SIZE` to the produced row nearest ~148; output cell to
  `bart_rfe_features_full_brier.csv`.

- [ ] **Step 2: Run headless** (background, log `/tmp/bart_rfe_brier_full.out`; caffeinate on).
  Verify clean exit; capture hold-out AUROC/acc/Brier/log-loss + 1-SE count → **Run D results**.

- [ ] **Step 3: Commit**
```bash
git add notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/bart_rfe.ipynb \
        data/predict_games/model_features_in/bart_rfe_features_full_brier.csv
git commit -m "Run D: BART full set, Brier+1-SE selection hold-out <metrics>"
```

---

## Task 12: Record results in README

**Files:** Modify `README.md`

- [ ] **Step 1: Add table rows** (after the Run 16 row) for Runs A–D with their
  feature counts and hold-out ROC-AUC / accuracy / Brier, numbered 17–20, e.g.:
```
| 17 | 2026-06-16 | **Brier+1-SE selection**, rank-only pool, XGBoost | <A_BEST> | <auroc> | <acc> | <brier> |
| 18 | 2026-06-16 | **Brier+1-SE selection**, full set, XGBoost | <B_BEST> | <auroc> | <acc> | <brier> |
| 19 | 2026-06-16 | **Brier+1-SE selection**, rank-only pool, BART | <C_n> | <auroc> | <acc> | <brier> |
| 20 | 2026-06-16 | **Brier+1-SE selection**, full set, BART | <D_n> | <auroc> | <acc> | <brier> |
```

- [ ] **Step 2: Add a narrative bullet** after the Run 14→15/16 entry explaining: the
  selection metric switched from ranking (AUROC) to a proper scoring rule (Brier) with a
  1-SE rule replacing the 0.005 tolerance; the configurable `SELECTION_METRIC` knob; whether
  optimizing Brier beat the 0.2185 Brier bar (and the AUROC trade-off); and the BART
  replicate-SE caveat. Use the actual captured numbers.

- [ ] **Step 3: Commit**
```bash
git add README.md
git commit -m "Record Runs 17-20: Brier+1-SE selection vs AUROC-selected champions"
```

---

## Self-Review notes

- **Spec coverage:** brier metric → Task 1; per-fold storage → Task 2; XGBoost 1-SE → Task 3;
  BART 1-SE + caveat → Task 4; configurable `SELECTION_METRIC` in all three notebooks →
  Tasks 5–7; grid scoring map → Task 6; 4 runs × `_brier` artifacts → Tasks 8–11; README →
  Task 12. All covered.
- **Method-name consistency:** `get_best_num_features_1se()` (XGBoost) and
  `get_best_features_1se(higher_is_better=…)` (BART) — used identically in tests, notebooks,
  and the README. `all_model_score_folds`, `SELECTION_METRIC`, `_SCORER`, `_SCORING`,
  `_HIGHER_IS_BETTER` consistent throughout.
- **Placeholders:** `<A_BEST>`, `<B_BEST>`, `<C_n>`, `<D_n>`, `<metrics>`, `<auroc>` are
  runtime-captured values, each with an explicit capture step — expected for an experiment.
- **Direction handling:** XGBoost reuses `HIGHER_IS_BETTER_METRICS`; BART takes an explicit
  `higher_is_better` flag derived from `SELECTION_METRIC == 'roc_auc'`. Brier and log_loss are
  both lower-is-better and handled the same way.
