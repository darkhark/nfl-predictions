# Configurable selection metric + 1-SE rule for XGBoost & BART feature selection

**Date:** 2026-06-16
**Status:** Approved design — ready for implementation plan
**Motivation:** ROC-AUC is a pure *ranking* metric — it scores discrimination only and is
blind to calibration (the actual probability values). For this project's intended use
(probability quality / bet-sizing-style decisions), a **proper scoring rule** is the more
relevant selection target. This change switches feature/model selection from ROC-AUC to
**Brier** by default, makes the metric **configurable**, and replaces the over-shrinking
`0.005` tolerance with a **1-SE rule**.

## Goals

1. Make the selection metric a single configurable knob per notebook (`SELECTION_METRIC`),
   defaulting to `'brier'`, switchable to `'roc_auc'` / `'log_loss'` with no other edits.
2. Replace XGBoost's `get_best_num_features(0.005)` tolerance and BART's pure-peak with a
   shared **1-SE rule**: the most parsimonious feature count whose mean metric is within one
   standard error of the best.
3. Apply the new selection to **both pools** (rank-only and full ranked+aggregated) and
   **both estimators**, and compare to the AUROC-selected champions — primarily on Brier
   (the bar is BART run-10's **0.2185**), with AUROC reported for continuity.

Non-goal: changing the elimination *path*. XGBoost still drops features by `total_gain`;
BART still drops by `variable_inclusion`. Only the **count-selection metric and rule** change.

## Background — current selection mechanics

- **XGBoost** (`rfe.ipynb` → `ClassifierCrossValidationRecursiveFeatureSelection`):
  `model_score_metric='roc_auc'` scores each surviving feature count via 5-fold CV;
  `get_best_num_features(0.005)` picks the smallest size within 0.005 of the best score.
  `grid_search.ipynb` then tunes with `scoring='roc_auc'`.
- **BART** (`bart_rfe.ipynb` → `BartBackwardElimination`): `bart_fit` returns
  `roc_auc_score` on the fixed 2022–2023 validation slice; `get_best_features()` takes the
  validation-curve **peak** (`idxmax`), no tolerance. Six sampler-seed replicates per size.
- Run 13 lesson (recorded in README): the 0.005 tolerance **over-shrinks on flat curves**
  (picked 17 features / 0.688 hold-out vs the better 51-feature / 0.707 model). The 1-SE
  rule is the principled replacement.

## Design

### 1. `classifier_cross_validation.py` (XGBoost RFE) — source change + tests

- **Add `brier` metric** to `_get_test_scores`: `brier_score_loss(y_test, preds)`. It is
  correctly *absent* from `HIGHER_IS_BETTER_METRICS`, so the existing direction logic treats
  it as lower-is-better. Update the `ValueError` message to list `brier`.
- **Store per-fold scores**: add `self.all_model_score_folds = []` in `__init__` and append
  the per-fold `model_scores` list each iteration (alongside the existing
  `self.all_model_scores.append(np.mean(model_scores))`). Needed for the SE band.
- **Add `get_best_num_features_1se()`**: compute, per feature count, the mean metric and the
  standard error `SE = std(folds, ddof=1) / sqrt(n_folds)`. Let `best` be the optimum mean
  (max if metric in `HIGHER_IS_BETTER_METRICS`, else min) and `se_at_best` its SE. Return the
  **most parsimonious** (fewest-feature) count whose mean is within the 1-SE band of `best`
  (`mean <= best + se_at_best` for lower-is-better; `mean >= best - se_at_best` for
  higher-is-better). Direction-aware → works for any metric.
- **Unit tests**: brier branch returns expected values; per-fold storage shape; 1-SE
  selection on a synthetic curve (a) returns the optimum when the band is tight, (b) returns
  a smaller set when a flat tail is within the band, (c) correct direction for both a
  higher-is-better and a lower-is-better metric.

### 2. `backward_elimination.py` (BART RFE) — source change + tests

- **Add `get_best_features_1se(higher_is_better=False)`**: uses the already-stored
  `replicate_scores` per size. `SE = std(replicate_scores, ddof=1) / sqrt(n_replicates)` at
  the best size; return the most parsimonious size within the 1-SE band (direction-aware).
- Keep the existing `get_best_features()` (peak) for backward compatibility; the notebook
  calls the new 1-SE method.
- **Documented caveat (in docstring + README):** BART's RFE has no cross-validation — the
  replicates are sampler-seed reruns on a *fixed* validation slice, so this SE captures
  **sampler noise (~0.003), not data/generalization variance**. The band is therefore tight
  (near-peak). This is an accepted, pragmatic proxy; adding CV to BART is out of scope
  (it would multiply already-long runtimes).
- **Unit tests**: 1-SE selection on a synthetic history with known `replicate_scores`
  (tight band → peak; flat tail within band → smaller set; both directions).

### 3. Notebook changes — configurable `SELECTION_METRIC`

Each notebook gets a single constant near the top, default `'brier'`:

- **`rfe.ipynb`**: `SELECTION_METRIC = 'brier'`; pass it as `model_score_metric`; select via
  `get_best_num_features_1se()`. Echo the chosen metric and selected count.
- **`grid_search.ipynb`**: derive `scoring` from a map
  `{'roc_auc':'roc_auc', 'brier':'neg_brier_score', 'log_loss':'neg_log_loss'}`;
  `BEST_NUM_FEATS` from the 1-SE pick. Continue printing hold-out AUROC + accuracy + Brier +
  log loss (so every run is comparable on all axes regardless of selection metric).
- **`bart_rfe.ipynb`**: `bart_fit` computes the chosen metric (map metric → scorer +
  direction); select via `get_best_features_1se()`. The validation-curve plot labels the
  metric and orients correctly (lower-is-better for Brier).

Flipping back to AUROC later is a one-line change (`SELECTION_METRIC = 'roc_auc'`), which
reproduces the existing champions; the 1-SE rule (not the old 0.005 tolerance) still applies.

### 4. Runs, artifacts, comparison

Scope = both pools × both estimators → **4 RFE runs + 2 grid searches** under Brier+1-SE:

| Run | Pool | Estimator | Reads | Writes (new) |
|---|---|---|---|---|
| A | rank-only | XGBoost | `xgb_features_list.csv` (RANK_ONLY=True) | `rfe_features_kfolds_brier.csv` |
| B | full | XGBoost | `xgb_features_list.csv` (RANK_ONLY=False) | `rfe_features_kfolds_full_brier.csv` |
| C | rank-only | BART | trace from A | `bart_rfe_features_brier.csv` |
| D | full | BART | trace from B | `bart_rfe_features_full_brier.csv` |

- `_brier` suffix preserves the AUROC-selected champion artifacts.
- BART start pools: reuse the established sizes (rank-only ~91 lineage; full set 148, the
  generous pool validated in Run 15/16). Each BART run is ~40+ min on AC; plan for a
  multi-hour batch and keep the Mac on power + `caffeinate`.
- **Comparison**: report each run's hold-out AUROC, accuracy, Brier, log loss. Judge
  primarily on **Brier** (does selecting for calibration beat the 0.2185 bar?) and note the
  AUROC trade (a Brier-selected model may give up a little ranking for better calibration —
  that is the point of the change). Record as new README run rows + a narrative bullet.

## Execution / environment notes (carried from Run 15)

- Headless: env jupyter `/opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/jupyter`,
  prepend `PYTHONPATH=<repo root>`, `--ExecutePreprocessor.kernel_name=python3 --timeout=-1`,
  run in background, watch the sidecar logs.
- Per the verify-notebook-executions rule: confirm clean exit + the echoed `SELECTION_METRIC`
  + selected count before trusting any result.
- Source changes to `data_science_utilities` must be importable headlessly (already covered
  by `PYTHONPATH`).

## Success criteria

- Both RFE classes accept Brier and expose a working, direction-aware 1-SE selector, with
  passing unit tests.
- A single `SELECTION_METRIC` per notebook drives metric + grid scoring + 1-SE direction;
  setting it to `'roc_auc'` reproduces prior behavior (modulo the tolerance→1-SE change).
- 4 runs complete cleanly; README records the Brier+1-SE results vs the AUROC-selected
  champions, with an explicit Brier-first verdict and the BART-SE caveat noted.
