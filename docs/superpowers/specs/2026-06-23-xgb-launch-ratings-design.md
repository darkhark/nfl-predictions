# XGBoost with Launch Ratings — Design (Phase 1)

- **Date:** 2026-06-23
- **Branch:** `xgb-launch-ratings` (off `origin/master`, which now contains Phase 0 / PR #42 —
  2025 launch ratings flow through `src/data/madden/`)
- **Status:** Design — pending user review
- **Program context:** Phase 1 of the four-phase Madden launch-ratings program (Phase 0 = data
  layer, done). Phases 2 (BART) and 3 (Bayesian posterior) are out of scope here.

## 1. Goal

**The first time Madden features enter feature selection.** Run recursive feature elimination
(RFE) over the rank-only candidate pool **plus the 188 `madden_*` launch-rating columns**, then a
randomized XGBoost hyperparameter search on the selected set, and evaluate the 2024+2025 hold-out.

Two questions, answered honestly either way:
1. **Do launch-rating features SURVIVE RFE?** — i.e. does the brier-1se-selected set contain any
   `madden_*` columns, and which?
2. **Does the selected model BEAT the champion** on the hold-out? Anchors: best XGBoost **Run 11 =
   0.707 / Brier 0.2206**; BART **Run 6 = 0.705 / 0.2194**, **Run 10 = 0.708 / 0.2185**.

This is **Run 28** in the README run-log (the numbered slot Phase 0 reserved).

## 2. Background / constraints (verified)

- **Precondition — rebuild the model-ready parquet.** `data/predict_games/input_data/schedule_and_weekly.parquet`
  is git-ignored / generated. The Madden columns reach the candidate pool only when the parquet is
  rebuilt by `scripts/data_assembly/predict_game_winner/schedule_and_weekly.py`, which calls
  `get_schedule_and_weekly_data(range(2003,2026), include_play_by_play=True)`. To populate 2025
  launch ratings the rebuild must run with **`MADDEN_TOOLS_CDN_BASE=https://cdn.madden.tools`** in
  the environment (else `launch.load_madden_launch` degrades 2025 to empty). The rebuild also
  regenerates `model_features_in/xgb_features_list.csv` (the candidate pool, which now includes the
  188 `madden_*` columns — for the first time in an RFE pool).
- **RFE config (from `rfe.ipynb`, matched verbatim):** `ClassifierCrossValidationRecursiveFeatureSelection`
  with `RFE_XGB_PARAMS = {n_estimators:5000, learning_rate:0.15, max_depth:5, eval_metric:'auc'}`
  (a deliberately over-fit ranker), `RANK_ONLY = True`, `SELECTION_METRIC = 'brier'`, selection via
  `get_best_num_features_1se()` (the brier 1-standard-error rule), `RANDOM_SEED = 32`.
- **RANK_ONLY keeps Madden:** `partition.is_rank_only_kept` drops only columns carrying
  `cumulative`/`ewma`/`rolling` markers (unless `_rank`/`_rank_change`). `madden_*` columns carry
  none, so all 188 survive the rank-only filter into the RFE pool.
- **Grid-search protocol (from `grid_search.ipynb`, the champion protocol):** seed-32 shuffle
  (`df.sample(frac=1, random_state=32)` then `np.random.seed(32)`), split **train `season<2022`**,
  **valid `2022–2023`** (early stopping), **hold-out `season>=2024`** (~1,088 rows);
  `OptimalXGBHyperparameterSearch` (`random_search.py`), `SELECTION_METRIC='brier'` (`neg_brier_score`).
- **The hold-out eval suite already exists and is tested:** `data_science_utilities/models/bayes_logistic/evaluate.py`
  → `evaluate(y_true, preds, p_std=None, weeks=None)` returns auroc, accuracy, brier, logloss,
  per-week-auroc mean, and a 10-bin reliability curve. Reuse it (XGB has no posterior →
  `p_std=None`, so width-stratified Brier is skipped) so Run 28 is metric-comparable to Runs 27/10/6.
- **Never feed leakage/meta cols:** `game_id, season, season_type, opp_team, opp_score, target_team,
  target_score, h_win`. `season` is used only for the split, then dropped before fit. The one-week
  shift and the within-season `_ovr` z-score are already baked into the parquet — do not re-apply.

## 3. Decisions (confirmed with the user)

1. **Reproducible script**, not notebooks: a `scripts/experiments/xgb_launch_ratings.py` that
   reuses the existing RFE classifier, `random_search.py`, and `bayes_logistic.evaluate`, exporting
   `results.json` + the selected-feature CSV (the Run-27 `scripts/experiments/bayes_logistic.py`
   precedent). Unit tests cover the Phase-1 data-prep/selection glue.
2. **Single madden-inclusive run vs the recorded champion** (no with/without-madden A/B): one
   RFE→grid→eval pass with the 188 madden columns in the pool, compared to the recorded champions
   (XGB 0.707, BART 0.705/0.708). Cheaper; the "survive RFE" question is answered directly by the
   selected set, and the hold-out is compared to the recorded anchors.
3. **Selection = brier 1-SE** (`SELECTION_METRIC='brier'`, `get_best_num_features_1se`), matching the
   champion — dictated by the comparability contract, not re-litigated.

## 4. Architecture

Three units, each independently runnable (RFE and grid are compute-heavy → run in the background).

### 4.1 Parquet rebuild (precondition — existing script, just run it)
`MADDEN_TOOLS_CDN_BASE=https://cdn.madden.tools python -m scripts.data_assembly.predict_game_winner.schedule_and_weekly`
→ regenerates `schedule_and_weekly.parquet` (2025 madden populated) + `xgb_features_list.csv`.
A guard at the start of the experiment script asserts the parquet exists and that 2025 `madden_*`
coverage is non-trivial (>30% non-null on `_ovr` cols), failing loudly if the rebuild was run
without the CDN env.

### 4.2 `scripts/experiments/xgb_launch_ratings.py` — RFE stage (`run_rfe`)
- Load parquet + `xgb_features_list.csv`; apply the seed-32 shuffle.
- Build the RFE pool: candidate features minus the leakage/meta cols, filtered by
  `partition.is_rank_only_kept` (RANK_ONLY), which retains all 188 `madden_*`.
- Feature selection runs on **pre-hold-out data (`season < 2024`)** via
  `ClassifierCrossValidationRecursiveFeatureSelection(..., RFE_XGB_PARAMS, model_score_metric='brier')`,
  `get_optimal_features_no_grouped_records(...)`, `get_best_num_features_1se()`.
- Write the selected set to `data/predict_games/model_features_in/rfe_features_kfolds_brier_madden.csv`
  (a NEW file — does not overwrite the champion's `rfe_features_kfolds_brier.csv`).

### 4.3 `scripts/experiments/xgb_launch_ratings.py` — grid + eval stage (`run_grid_and_eval`)
- Read the selected CSV; seed-32 shuffle; split train `<2022` / valid `2022–2023` / hold-out `>=2024`.
- `OptimalXGBHyperparameterSearch` over the selected set (brier scoring); fit the best model with
  early stopping on the valid fold.
- Hold-out preds → `bayes_logistic.evaluate(y_true, preds, p_std=None, weeks=holdout_weeks)`.
- **Madden-survival analysis:** count + list the `madden_*` columns in the selected set, and their
  XGBoost importance ranks in the final model.
- Export `data/predict_games/xgb_launch_ratings/results.json` (config, selected-feature count,
  madden columns selected + ranks, full metric suite, champion deltas) and save the tuned model to
  `models/best_random_xgb_model_launch_ratings.json`.

### 4.4 `main()`
Runs `run_rfe` then `run_grid_and_eval` end-to-end; supports running the stages separately (RFE is
the long pole) so the heavy RFE can finish before the grid search.

## 5. Data flow
```
rebuild parquet (CDN env) → schedule_and_weekly.parquet (+ xgb_features_list.csv, incl. 188 madden_*)
   │  seed-32 shuffle
   ▼
RFE pool = candidates − leakage/meta, rank-only-filtered (188 madden_* retained)
   │  CV RFE on season<2024, RFE_XGB_PARAMS, brier-1se
   ▼
rfe_features_kfolds_brier_madden.csv  (selected set; madden survival = which madden_* present)
   │  seed-32 split (train<2022 / valid22-23 / holdout>=2024)
   ▼
OptimalXGBHyperparameterSearch (brier) → best model → holdout preds
   │  bayes_logistic.evaluate(p_std=None, weeks)
   ▼
results.json (+ model JSON) → Run 28 README entry + verdict
```

## 6. Testing
- **Reuse** the already-tested `bayes_logistic.evaluate` suite (no re-test).
- **New unit tests** (pure, no training) for the Phase-1 glue:
  - seed-32 shuffle + split produces the documented partition (train all `<2022`, valid `2022–2023`,
    hold-out `>=2024`) and drops `season` + the leakage/meta cols before fit.
  - the RFE-pool builder applies `is_rank_only_kept` and **retains all `madden_*`** columns
    (assert on a synthetic column list).
  - the madden-survival analyzer correctly counts/lists `madden_*` columns in a selected set.
  - the parquet/coverage guard raises a clear error when 2025 `madden_*` coverage is ~0 (rebuild
    without CDN env).

## 7. Deliverables
- `scripts/experiments/xgb_launch_ratings.py` (reproducible) + unit tests.
- `rfe_features_kfolds_brier_madden.csv`, `data/predict_games/xgb_launch_ratings/results.json`,
  `models/best_random_xgb_model_launch_ratings.json`.
- **README Run 28** entry (table row + narrative) with the honest verdict: which/how many madden
  features survived RFE, the hold-out AUROC/Brier, and the delta vs the champion. A result where
  Madden does NOT survive selection or does NOT beat the champion is a valid, reportable outcome.
- A PR off `xgb-launch-ratings`.

## 8. Out of scope / deferred
- The with/without-madden A/B (user chose the single run).
- BART (Phase 2) and the Bayesian posterior (Phase 3).
- The theedgepredictor `raw→dataset/parquet` upgrade (still deferred from Phase 0).

## 9. Risks
- **Runtime:** RFE over the rank-only pool (k-fold CV XGBoost × ~10 elimination rounds) + a
  RandomizedSearchCV are heavy (tens of minutes each). Run as background steps; record wall-clock.
- **RFE stochasticity:** selection depends on seed 32 (fixed) and the CV folds; report the selected
  count and curve so the choice is auditable. Re-running should reproduce under the fixed seed.
- **Honest-null risk:** Madden may not survive selection or may not move the hold-out. The forced-block
  ablation (Runs 25–26) showed a lift, but selection is a different, harder test — report truthfully.
