# BART with Launch Ratings — Design (Phase 2)

- **Date:** 2026-06-24
- **Branch:** `bart-launch-ratings` (off `origin/master`, which now has Phase 0 + Phase 1)
- **Status:** Design — pending user review
- **Program context:** Phase 2 of the four-phase Madden launch-ratings program. Phase 3
  (Bayesian posterior on the winner's feature set) is out of scope here.

## 1. Goal

**The first time Madden features go through BART.** BART is the project's champion estimator
(Run 10 = **0.708**, the AUROC champion; Run 6 = 0.705) and handles smooth talent→win-probability
relationships differently than XGBoost's axis-aligned splits — the most likely place Madden's
QB-talent signal could actually move the hold-out, after Phase 1's XGBoost null (Runs 28–29, ~0.700).

Run BART backward-elimination feature selection over the **full 188-column Madden family** plus the
BART champion's non-Madden start features, then a final 4-chain BART, and report the 2024+2025
hold-out vs the BART champion. Two questions, answered honestly:
1. **Do Madden features survive BART backward-elimination** (and which)?
2. **Does the selected BART model beat the champion** (Run 10 = 0.708 / 0.2185, Run 6 = 0.705 / 0.2194)?

This is **Run 30** in the README run-log (Run 28 = XGB full pool, Run 29 = XGB lean pool).

## 2. Background / constraints (verified)

- **BART can't start from the full 6,348 rank-only pool.** Each BART fit is ~minutes; backward
  elimination re-fits every iteration. The existing `bart_rfe.ipynb` starts from the **XGBoost-RFE
  top-90** (`rfe_features_kfolds_brier.csv`, `START_POOL_SIZE=90`) — the exact lineage that produced
  champion Run 10. Phase 2 starts from **that same 90 non-Madden set + all 188 Madden columns = 278
  candidates** (verified: the 90 are non-Madden and all present in the current parquet).
- **Parquet:** reuse the Phase-1 Madden-built `schedule_and_weekly.parquet` (verified on disk:
  11,898×9,492, 188 madden cols, 2025 `_ovr` coverage 0.765). A coverage guard (reused from Phase 1)
  asserts 2025 madden coverage > 0.30 before running; rebuild with `MADDEN_TOOLS_CDN_BASE` if absent.
- **BART selection-grade fit (from `bart_rfe.ipynb`, matched):** probit BART `pmb.BART('mu', X, y,
  m=50)`, `pm.sample(draws=500, tune=1000, chains=2, cores=2)`; score the 2022–2023 validation slice
  with **Brier**; return chain-averaged `variable_inclusion` as a `pd.Series` indexed by feature name
  (BartBackwardElimination raises if any candidate is missing from it).
- **Backward elimination (`BartBackwardElimination`):** `drop_rate=0.2, min_features=10,
  replicates=2, max_workers=2, base_seed=32`; select via `get_best_features_1se(higher_is_better=False)`
  (Brier is lower-is-better; the 1-SE band uses the replicate-seed SE).
- **Final BART (from `bart.ipynb`, the champion protocol):** seed-32 shuffle, split train `season<2022`
  / valid `2022–2023` / hold-out `season>=2024` (1,088 rows); probit BART `m=50`,
  `pm.sample(draws=1000, tune=1000, chains=4, cores=4)`; hold-out preds = posterior-mean P(win), and
  the posterior **std** is retained for the width-stratified Brier.
- **NaN handling = the `-100` sentinel** (`BART_NAN_SENTINEL = -100.0`, the established BART
  convention — `bart.ipynb`/`bart_rfe.ipynb`). BART cannot ingest NaN. Madden columns are ~30% NaN;
  the sentinel gives BART a clean "missing" branch. (The median-impute + `*_was_missing` preprocessor
  caveat in the brief is for the Phase-3 Bayesian model only, NOT BART.)
- **Eval suite:** reuse the tested `bayes_logistic.evaluate(y_true, preds, p_std, weeks)` — auroc,
  accuracy, brier, logloss, per-week-auroc, reliability curve, and (since BART has a posterior)
  **width-stratified Brier** via `p_std`. Directly comparable to Runs 6/10/27.
- **Never feed leakage/meta cols:** `game_id, season, season_type, opp_team, opp_score, target_team,
  target_score, h_win`. `season` is used only for splits, then dropped. One-week shift + within-season
  `_ovr` z-score already baked into the parquet.

## 3. Decisions (confirmed with the user)

1. **Start pool = champion-start + full Madden family** (90 non-Madden from the BART champion's
   XGBoost-RFE top-90 + all 188 Madden columns, levels *and* diffs = 278). The cleanest apples-to-apples
   "BART champion's start features + the full Madden family."
2. **Reproducible script** (`scripts/experiments/bart_launch_ratings.py`) reusing
   `BartBackwardElimination`, the `bart_fit` pattern, and `bayes_logistic.evaluate`; unit-test the
   data-prep/start-pool/results glue. (Not notebooks — the Run-27/Phase-1 precedent.)
3. **Brier-1SE selection**, `-100` NaN sentinel, seed-32 protocol — matched to the champion.

## 4. Architecture — `scripts/experiments/bart_launch_ratings.py`

Pure, unit-tested helpers + heavy run functions (the BART fits are the live experiment, run by the
controller in the background — like Phase 1's Task 5).

### 4.1 Pure helpers (unit-tested)
- `build_start_pool(champion_start: list[str], madden_cols: list[str]) -> list[str]` — the 90
  non-Madden champion-start features + all Madden columns, de-duplicated, preserving order. Excludes
  the target/leakage cols defensively.
- `madden_columns(features)` — reused from the Phase-1 pattern (cols containing `madden`).
- `to_bart_matrix(df, features) -> np.ndarray` — `df[features].fillna(-100.0).to_numpy(float)`, with
  an assert that no NaN remains (the stale-output guard from `bart.ipynb`).
- `assert_madden_2025_coverage(df, min_cov=0.30)` — reused guard.
- `build_results(metrics, selected_features, history, best_num_feats)` — madden-survival
  (count + which madden cols selected), the validation curve summary, and champion deltas vs
  `{bart_run6: 0.705/0.2194, bart_run10: 0.708/0.2185}`.

### 4.2 BART fit + selection (heavy)
- `make_bart_fit(train_df, valid_df, y_train, y_valid)` → `bart_fit(features, seed)` closure:
  selection-grade probit BART (`m=50, draws=500, tune=1000, chains=2`), score valid Brier, return
  `{'validation_score': brier, 'variable_inclusion': pd.Series(incl, index=features)}`.
- `run_bart_rfe(df, start_features, out_csv)` → `BartBackwardElimination(bart_fit, drop_rate=0.2,
  min_features=10, replicates=2, max_workers=2, base_seed=32, on_iteration=<sidecar log>)`,
  `rfe.run(start_features)`, write the history; return `get_best_features_1se(False)` (the selected set).

### 4.3 Final BART + eval (heavy)
- `run_final_bart(df, selected_features)` → seed-32 split; final probit BART (`m=50, draws=1000,
  tune=1000, chains=4`); hold-out posterior-mean preds + posterior std; returns `(preds, p_std,
  holdout_df)`.
- Eval: `bayes_logistic.evaluate(y_holdout, preds, p_std=p_std, weeks=holdout_df['week'])`.
- Export `data/predict_games/bart_launch_ratings/results.json` + the selected-feature CSV
  (`bart_rfe_features_brier_madden.csv`); save the trace summary is NOT required (Phase 3 owns the
  posterior export).

### 4.4 `main()` + CLI
`--stage rfe|final|all`; `rfe` writes the selected CSV, `final` reads it and evaluates. The heavy
stages run in the background; supports running them separately (RFE is the long pole).

## 5. Data flow
```
parquet (madden-built) + champion-start CSV (90 non-madden)
   │  build_start_pool → 278 candidates (90 non-madden + 188 madden)
   ▼  seed-32 shuffle; train<2022 / valid22-23 / holdout>=2024; fillna(-100)
BART backward-elimination (drop 20%/iter, brier on valid, variable_inclusion ranks) → brier-1SE
   ▼
bart_rfe_features_brier_madden.csv (selected set; madden survival)
   ▼  final 4-chain BART on selected → holdout posterior mean + std
bayes_logistic.evaluate(p_std=std, weeks) → results.json → Run 30 + verdict vs BART champion
```

## 6. Testing
- **Reuse** the tested `bayes_logistic.evaluate` (no re-test).
- **New unit tests** (pure, no BART training): `build_start_pool` (full madden + champion-start,
  no leakage cols, dedup); `to_bart_matrix` (NaN→-100, no NaN remains, dtype float); the 2025
  coverage guard; `build_results` (madden-survival count + champion deltas). The `bart_fit` contract
  (returns validation_score + name-indexed variable_inclusion Series covering all candidates) is
  verified with a tiny monkeypatched/stub fit (no real PyMC) to prove the BartBackwardElimination
  wiring, OR a single tiny real fit if fast enough.

## 7. Deliverables
- `scripts/experiments/bart_launch_ratings.py` + unit tests.
- `bart_rfe_features_brier_madden.csv`, `data/predict_games/bart_launch_ratings/results.json`.
- **README Run 30** entry (table row + narrative) with the honest verdict: which/how many Madden
  features survived BART selection, the hold-out metric suite, and the delta vs the BART champion.
  A null (Madden survives but doesn't beat 0.708) is a valid, reportable outcome.
- A PR off `bart-launch-ratings`.

## 8. Out of scope / deferred
- The family-pairing ablation (the brief's Phase-2 second half) was **already done in Phase 1**
  (the Madden-inclusive cross-family + internal ablations); not repeated here. Reference it.
- Phase 3 (Bayesian posterior export). The Phase-3 winner is whichever of XGBoost (Phase 1, 0.700)
  or BART (this phase) scores better on the hold-out.
- The theedgepredictor `raw→dataset/parquet` upgrade (still deferred).

## 9. Risks
- **Runtime:** BART RFE over ~278 candidates (drop 20%/iter to 10, ×2 replicates, ~minutes/fit) ≈
  ~3 hr; final 4-chain BART ≈ 20–30 min. Run in the background; record wall-clock. The history is
  persisted per iteration (the util checkpoints), so a mid-run failure leaves completed iterations.
- **Sampler noise:** BART's 1-SE band reflects sampler-seed noise (~0.003), not CV variance — report
  the validation curve so the selected count is auditable. Watch for divergences / poor mixing.
- **Honest-null risk:** Phase 1 showed Madden is sub-additive (substitutes for team-quality signal).
  BART may use it slightly better, but a result that does NOT beat 0.708 is expected-plausible and
  will be reported plainly. The `-100` sentinel on ~30% NaN madden cols is a known modeling choice
  (BART splits the missing branch); flag it in the write-up.
