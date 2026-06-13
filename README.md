# nfl-predictions

An end-to-end machine learning pipeline that predicts NFL game outcomes from real
NFL data ([nflverse](https://github.com/nflverse) via
[`nfl_data_py`](https://github.com/nflverse/nfl_data_py)). The project covers the full
modeling lifecycle — data collection, leakage-aware feature engineering, feature
selection, hyperparameter tuning, probability calibration, and model evaluation — using
XGBoost, plus a Bayesian BART comparison model (`pymc-bart`).

It is a personal project built to apply production data-science methodology (the same
approach I use professionally — custom recursive feature elimination, cross-validation,
calibration, and drift-aware design) to football analytics.

## What it does

- **Target:** predict the winner of each regular-season NFL game (binary classification).
- **Data:** 23 seasons of real NFL data (2003–2025) assembled from schedules, weekly
  team/player stats, and Next Gen Stats.
- **Approach:** engineer prior-week team form features, select a compact feature set with
  a custom RFE routine, tune an XGBoost classifier, and calibrate the output probabilities.

> **Status:** the data pipeline, feature engineering, feature selection, tuning, and
> calibration utilities are complete and tested. Predictive performance on game outcomes
> is still being improved — NFL game prediction is a hard problem and the current model
> does not yet consistently beat a strong baseline. The pipeline and methodology are the
> finished, reusable part; model accuracy is an ongoing line of work (see [Roadmap](#roadmap)).

## Data sources

Data is pulled from [nflverse](https://github.com/nflverse). Schedules come through
`nfl_data_py`; weekly player stats are read directly from nflverse's current
`stats_player` release, because the legacy `nfl_data_py` weekly endpoint was frozen at the
2024 season (and dropped the `dakota` metric — see [run 5](#results--experiment-log)):

| Source | Module | Notes |
| --- | --- | --- |
| Schedules | `src/data/schedule/collect.py` | Game results, rest days, roof type, optional Vegas lines (`nfl_data_py`) |
| Weekly stats | `src/data/weekly/collect.py` | Team-aggregated offensive/defensive box-score stats (nflverse `stats_player` release) |
| Next Gen Stats | `src/data/next_gen_stats/collect.py` | Passing / rushing / receiving advanced metrics |
| Play-by-play | `src/data/play_by_play/collect.py` | Per-play EPA/success/situational metrics aggregated to team-week, split by win-probability context, cached per season |

## Feature engineering

The feature design is deliberately **leakage-aware** — a game is only ever scored on
information that was available before kickoff:

- **Prior-week shift:** each team's stats are shifted forward one week so week *N* is
  predicted using cumulative form through week *N−1*.
- **Target/opponent reframing:** every game is duplicated into two perspectives (each team
  as the "target"), doubling the training signal and making the model symmetric to
  home/away.
- **Form features:** per-team cumulative average and average-change features for offense
  and defense (points, passing/rushing yards, EPA-adjacent box-score stats).
- **Rest & context:** days since previous game (with explicit bye-week and season-opener
  handling), division-game flag, and indoor/dome flag.
- **Franchise relocation mapping:** historical abbreviations are normalized
  (`STL→LA`, `SD→LAC`, `OAK→LV`) so team history joins cleanly across eras.
- **Optional market signal:** Vegas money lines / spreads can be retained as candidate
  features for an ensemble baseline.

Assembled datasets are written to Parquet (`fastparquet`) for fast, schema-consistent reads.

## Modeling utilities (`data_science_utilities/`)

Reusable, model-agnostic-ish components built around XGBoost:

- **Recursive Feature Elimination** (`feature_selection/recursive/classifier.py`):
  iteratively drops the least-important features by a decay rate and selects the *smallest*
  feature set within a tolerance of the best AUROC, to control overfitting. Reduces the
  candidate pool (~554 columns as of the latest run, ~329 after the rank-only filter) to a
  compact selected set (~54).
- **Cross-validated RFE** (`classifier_cross_validation.py`): the same idea under
  `StratifiedKFold` for more stable feature selection.
- **Randomized hyperparameter search** (`hyperparameter_search/random_search.py`):
  `RandomizedSearchCV` over the XGBoost parameter space.
- **Probability calibration** (`utils/calibration/factors.py`): odds-ratio segment
  adjustment factors (Platt-style) to align predicted probabilities with observed rates,
  including grouped and known-factor variants.
- **Anomaly scoring** (`models/xgb/anomaly_score/score.py`): per-column XGBRegressor
  residual scoring to flag unusual feature values during data validation.

## Project structure

```
src/data/                      # nfl_data_py collection + merge/feature logic
  schedule/ weekly/ next_gen_stats/ play_by_play/
  collect_all.py               # orchestration (schedule + weekly assembled)
data_science_utilities/        # reusable XGBoost RFE, search, calibration, anomaly tools
scripts/data_assembly/         # builds and saves the model-ready dataset
notebooks/model_training/      # RFE, cross-validation, grid-search, and BART experiments
models/                        # serialized best XGBoost models (JSON)
data/predict_games/            # input parquet + selected-feature lists
tests/data/                    # unit tests for the data/feature pipeline
```

## Setup

```bash
conda env create -f environment.yml
conda activate nfl-predictions
```

## Usage

Assemble the model-ready dataset (schedule + weekly form features) and save it to Parquet:

```bash
python -m scripts.data_assembly.predict_game_winner.schedule_and_weekly
```

Then run the training/feature-selection notebooks under
`notebooks/model_training/predict_games/schedule_and_weekly/`.

## Testing

```bash
python -m unittest discover -s tests
```

The suite validates feature-engineering correctness against known real games — rest-day
calculations (e.g. Thursday→Sunday turnaround), cumulative scoring averages, the
defense-matches-opposing-offense invariant, and bye-week edge cases.

## Results / experiment log

Runs 1–4 evaluate on a single **2023 hold-out season**; run 5 moves to a **two-season
2024+2025 hold-out** (every prior season is used for train/validation). Cross-validated RFE
selects the feature set, then a randomized hyperparameter search tunes the XGBoost
classifier; run 6 swaps the estimator for BART on the same inputs; run 7 returns to
XGBoost on a candidate pool expanded with play-by-play features. Two **out-of-sample**
metrics are tracked:

- **Hold-out ROC-AUC** — ROC-AUC over all hold-out rows pooled (both target/opp
  perspectives). Reported as the per-week mean, which matches the single pooled curve to
  within ~0.002.
- **Hold-out accuracy** — `best_model.score` on the hold-out rows at a 0.5 threshold.
  Treat as descriptive only: with 544 hold-out games the 95% binomial interval is
  roughly ±4 points, so run-to-run differences of a point or two are mostly noise, and a
  monotone recalibration can move it without changing ROC-AUC at all.
- **Hold-out Brier** — mean squared error of the predicted probabilities over the pooled
  hold-out rows (a proper scoring rule: uniquely minimized by reporting true
  probabilities, so it rewards calibration and sharpness together; lower is better).
  Reference points on the 2024+2025 hold-out: 0.25 = coin flip, **0.2495** = always
  predicting the training-era home-win rate (56.2%). Tracked from run 5 (the earlier
  models were not retained, and runs 1–4 used a different hold-out anyway).

> **In-sample vs out-of-sample.** For runs 1–4 the grid-search *cross-validated* ROC-AUC
> (~0.677) was measured on the training seasons and ran ~0.03 above the 2023 hold-out
> (~0.64) — a normal generalization drop. On the larger run-5 hold-out the relationship
> flips: pooled 2024+2025 ROC-AUC (**0.697**) sits slightly *above* the CV score (0.681),
> i.e. those two seasons were a touch more separable than the training-era cross-validation.
> An earlier "per-game AUROC" column was removed as misleading — it kept only the
> higher-probability row per game, which conditions on the model's own output and reads
> ~0.53–0.55 (near chance) for reasons unrelated to model quality. (For the right way to ask
> "when the model is confident, how good is it?", see the calibration / confidence-bucket /
> home-vs-away-gap cells in `cross_validation/grid_search.ipynb`.)

| # | Date | Feature selection | Features | Hold-out ROC-AUC | Hold-out acc | Hold-out Brier |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | 2024-07-23 | RFE (single split) | 28 | 0.629 | 0.597 | — |
| 2 | 2024-08-06 | Cross-validated RFE (`StratifiedKFold`) | 34 | 0.643 | 0.597 | — |
| 3 | 2026-06-11 | Cross-validated RFE + rank features | 40 | 0.643 | 0.599 | — |
| 4 | 2026-06-11 | Cross-validated RFE, **rank-only** (cumulative averages removed) | 32 | 0.642 | 0.618 | — |
| 5 | 2026-06-11 | Rank-only, **`dakota` dropped + 2024+2025 two-season hold-out** | 54 | **0.697** | **0.647** | 0.2218 |
| 6 | 2026-06-12 | **BART** (`pymc-bart`), same 54 features / split as run 5 | 54 | **0.705** | **0.660** | 0.2194 |
| 7 | 2026-06-12 | Rank-only + **play-by-play features** (EPA/success, situational, PROE × wp context), XGBoost | 45 | 0.696 | 0.647 | 0.2206 |
| 8 | 2026-06-12 | **BART** re-trained on the run-7 play-by-play feature set (corrected — see run 8 note) | 45 | 0.702 | 0.656 | 0.2199 |
| 9 | 2026-06-12 | Rank-only + **Phase 2 directional run/pass features**, XGBoost | 57 | 0.694 | 0.645 | 0.2291 |
| 10 | 2026-06-12 | **BART** on the run-9 directional feature set | 57 | **0.708** | 0.654 | **0.2185** |
| 11 | 2026-06-12 | Rank-only + **Phase 3 features** (trenches, luck, tendencies, penalties, pace), XGBoost | 51 | **0.707** | 0.649 | 0.2206 |

**What changed between runs**

- **Run 1 → 2:** moved feature selection and tuning to cross-validated RFE
  (`StratifiedKFold`) for more stable selection; selected set grew 28 → 34. Hold-out ROC-AUC
  improved (0.629 → 0.643); accuracy was flat.
- **Run 2 → 3:** added **league-wide rank and rank-change features** — for every cumulative
  form stat, each team's rank within its `(season, week)` (offense best-to-worst, defense
  worst-to-best) and the week-over-week change in that rank, plus ranks for points scored /
  allowed. This expanded the candidate pool from ~352 to ~576 columns. RFE selected **40
  features, only 7 of them ranks**, and out-of-sample performance was essentially unchanged
  (ROC-AUC 0.643, accuracy 0.599) — the ranks were mostly crowded out by their
  highly-correlated cumulative-average source columns.
- **Run 3 → 4:** dropped every non-rank cumulative feature (averages, their changes, and
  sums) before RFE, keeping only the rank / rank-change features plus raw per-game stats and
  context. The crowding-out hypothesis held: RFE now selected **22 of 32 features as ranks**
  (vs 7 of 40 in Run 3), and **hold-out accuracy rose to 0.618 — the best of any run** (+2 pts
  over the long-standing 0.597), with ROC-AUC holding at 0.642. Reproduce by setting
  `RANK_ONLY = True` (the default for this experiment) in `cross_validation/rfe.ipynb`; set it
  to `False` to restore the full-feature pipeline.
  > Single-season caveat: this is one 2023 hold-out, so the +2-pt accuracy gain is suggestive,
  > not conclusive — worth confirming across multiple hold-out seasons.
- **Run 4 → 5:** extended the data through 2025 and moved to a **two-season 2024+2025
  hold-out** (train `<2022`, early-stopping slice `2022–2023`). This required switching the
  weekly source to nflverse's current `stats_player` release, which **dropped the `dakota`
  metric** — so it was removed from the feature set and RFE was re-run on the dakota-free,
  rank-only candidate pool, selecting **54 features (37 ranks)**. On the larger hold-out,
  pooled ROC-AUC rose to **0.697** and accuracy to **0.647** (544 games / 1,088 rows). The
  sharper, retrained model also makes far more confident calls: confidence-bucketed accuracy
  is now monotonic (the 0.7–0.8 bucket holds **83 games at ~82%** vs only 5 games before),
  and the home-vs-away probability gap correlates with correctness much more strongly
  (r 0.07 → **0.21**; widest-gap quintile ~81% accurate). The two-season hold-out resolves
  the single-season caveat from run 4.
- **Run 5 → 6:** swapped the estimator — **Bayesian Additive Regression Trees** (probit-link
  BART via `pymc-bart`, m=50 trees, 4 chains) on the *identical* inputs as run 5: same
  parquet, same seed-32 shuffle, same 54 RFE-selected features, same train `<2022` /
  2024+2025 hold-out split (the 2022–2023 slice, which XGBoost used for early stopping, is
  reported as a pure validation check — BART needs no early stopping). Pooled hold-out
  ROC-AUC edged up to **0.705** and accuracy to **0.660**, with better residual metrics
  (Brier 0.219, log loss 0.629) and *under*confident rather than overconfident buckets.
  The Bayesian payoff: posterior uncertainty is informative — picks in the
  narrowest-posterior quartile hit **~79%** vs ~59% in the widest. Note the PGBART sampler
  reports high r-hat / low ESS on some latent `mu` dimensions (common for BART latents);
  posterior-mean *predictions* are stable across reruns (hold-out AUROC 0.702 with 2 chains
  vs 0.705 with 4). See `cross_validation/bart.ipynb`; unlike XGBoost there is no compact
  serialized model artifact, but the seeded notebook re-trains in ~75 s.
- **Run 6 → 7:** added **Phase 1 play-by-play features** (spec:
  `docs/superpowers/specs/2026-06-12-play-by-play-features-design.md`): per-play EPA and
  success rates (overall/pass/rush), early-down success, third-down conversion, red-zone TD
  rate per scrimmage red-zone trip, and pass rate over expected — each split three ways by
  win-probability context (competitive / garbage-leading / garbage-trailing) and computed as
  season-to-date `cumsum(numerator)/cumsum(denominator)` ratios with league-wide ranks and
  rank changes. This grew the rank-only candidate pool from ~350 to **569** columns.
  Cross-validated RFE selected **45 features — 21 of them play-by-play ranks (12 from
  garbage-time contexts)**, displacing many of Run 5's box-score survivors; CV ROC-AUC at
  the selection point is 0.678 (peak 0.680 at 51 features) vs ~0.677 for the Run 5 pool.
  After re-tuning (`grid_search.ipynb`, `BEST_NUM_FEATS = 45`), the 2024+2025 hold-out came
  back **flat vs Run 5: pooled ROC-AUC 0.696 (was 0.697), accuracy 0.647 (was 0.647)**.
  Interpretation: the pbp efficiency metrics carry real signal — RFE prefers them
  head-to-head against box-score ranks — but at the team-week-rank level that signal
  substantially **overlaps** what yards/EPA box totals already encoded, so aggregate skill
  didn't move. The model is no worse and now leans on cleaner inputs (competitive-context
  rates are insulated from garbage-time stat-padding). Phase 2 (directional run/pass
  splits) targets information the box score genuinely lacks; per the phase plan, whether to
  proceed is a judgment call given the flat Phase 1 result. PBP components are cached per
  season in `data/play_by_play/aggregated/` (refresh with
  `get_play_by_play_data(years, refresh=True)` for in-progress seasons).
- **Run 7 → 8:** re-trained **BART** (same `bart.ipynb` setup as run 6: probit link, m=50,
  4 chains, seed 32) on the run-7 45-feature play-by-play set. Corrected result (see the
  run 8 correction note below for why the originally published numbers were invalid):
  hold-out ROC-AUC **0.702**, accuracy **0.656**, Brier 0.2199 — slightly below run 6,
  telling the same story as XGBoost: the Phase 1 efficiency/situational metrics
  substantially overlap the box-score ranks they replaced. The experiment cadence going
  forward is XGBoost first, then BART, for each play-by-play phase.
- **Run 8 → 9:** added **Phase 2 directional features** — average yards and explosive-play
  rate (rush ≥ 10, pass ≥ 20) for 7 run buckets (`run_location` × `run_gap`) and 6 pass
  buckets (`pass_location` × `pass_length`), per wp context, off and def (936 candidate
  columns; rank-only pool grew 569 → 1,193). Era note: nflfastR has no pass charting
  before 2006, so directional pass features are NaN for 2003–2005 (run direction works in
  all eras). RFE selected **57 features — 35 play-by-play (22 directional, 13 of those
  about what defenses allow by direction)** and CV ROC-AUC peaked at **0.685** (highest of
  any pool; 0.681 at the 57-feature selection point). The hold-out stayed flat once more:
  pooled ROC-AUC **0.694**, accuracy **0.645** (vs 0.697/0.647 run 5, 0.696/0.647 run 7).
  Pattern across runs 5/7/9: XGBoost hold-out skill is insensitive to which of these
  correlated rank families it consumes — selection composition changes, aggregate skill
  doesn't. Run 9's Brier (0.2291, backfilled) is also the worst of the two-season-hold-out
  runs: this XGBoost model ranks games as well as its predecessors but its probabilities
  are noticeably less calibrated — every BART run beats every XGBoost run on Brier.
- **Run 8 correction:** the originally published run-8 numbers (0.705/0.660, identical to
  run 6) were **invalid** — `pymc-bart` raises on NaN inputs (sparse wp-context ranks have
  NaN early in seasons), the headless notebook execution failed, and the stale run-6
  outputs left in the notebook were mistakenly recorded as fresh results. `bart.ipynb`
  now fills NaN with an out-of-range sentinel (`BART_NAN_SENTINEL = -100`, letting the
  trees isolate the "no data yet" region the way XGBoost routes missing values) and
  asserts no NaN reaches the sampler. The corrected run 8: hold-out ROC-AUC **0.702**,
  accuracy **0.656**, Brier 0.2199 — slightly *below* run 6, consistent with the flat
  XGBoost result on the same features.
- **Run 9 → 10:** BART on the 57-feature directional set, with the NaN sentinel.
  **Hold-out ROC-AUC 0.708 — the best of any run** (0.705 run 6), with the best
  probability quality too (Brier **0.2185**, log loss **0.6267**) and validation ROC-AUC
  up 0.655 → 0.668. Accuracy at the 0.5 threshold dipped to 0.654 (0.660 run 6) — a
  threshold-sensitive metric at odds with the improved Brier/log-loss, suggesting a
  calibration pass could recover it. Posterior uncertainty remains informative:
  narrowest-posterior-quartile picks hit **77%**, and the width-stratified Brier
  (new cell in `bart.ipynb`) is strictly monotone — **0.171 / 0.220 / 0.234 / 0.246**
  from narrowest to widest posterior quartile, i.e. the narrow-posterior games carry
  most of the model's skill while the widest quartile is essentially the no-skill
  baseline (0.2495). Net: the directional features are the first addition to move the
  champion estimator — modestly, but in ranking AND probability quality simultaneously —
  where XGBoost stayed flat (run 9).
  > Sampler-variability note: PGBART's multiprocess sampling is not bit-reproducible
  > even with a fixed seed. A verification re-run of the identical notebook landed at
  > ROC-AUC 0.712 / accuracy 0.661 / Brier 0.2177 (vs the recorded 0.708 / 0.654 /
  > 0.2185) — treat BART numbers as carrying roughly ±0.004 run-to-run wobble. Both
  > executions beat run 6 on ROC-AUC and Brier, so the directional improvement is
  > robust to sampler noise; the table keeps the first recorded execution.
- **Run 10 → 11:** added **Phase 3 features** — sack/QB-hit/stuff rates, fumble-recovery
  luck, CPOE, YAC over expected, accepted-penalty rates (committed and drawn), and
  tendency/pace metrics (scramble/shotgun/no-huddle rates, seconds per play), all per wp
  context and side (504 candidate columns; rank-only pool 1,193 → 1,528). RFE selected
  **51 features — 24 play-by-play (10 Phase 3, 14 directional)**, with trench features
  prominent (sack rates on both sides, garbage-time stuff rate) plus fumble-recovery
  luck and pace tendencies. **XGBoost finally moved: pooled hold-out ROC-AUC 0.707**
  (vs 0.694–0.697 in runs 5/7/9 — first escape from that band in five feature
  generations), accuracy 0.649, Brier 0.2206 (recovered from run 9's 0.2291). The
  trench/luck family appears to carry signal the box score and the earlier pbp families
  did not.

## Roadmap

- Improve predictive performance toward / past a Vegas-implied baseline.
- Play-by-play Phase 3 (trenches, turnover luck, tendencies — see the spec in
  `docs/superpowers/specs/` and phase plans in `docs/superpowers/plans/`) and Next Gen
  Stats integration at the weekly grain; calibration pass on the run-10 BART champion
  (best AUROC/Brier but 0.5-threshold accuracy lags — calibration may recover it).
- Extend the Bayesian comparison (BART baseline done — run 6): more chains/draws, prior
  sensitivity, and using posterior uncertainty for bet-sizing-style decision rules.
- Address known `TODO`s: prevent season-average leakage in the NGS diff features, and move
  notebook-style execution out of `next_gen_stats/collect.py` import path.
