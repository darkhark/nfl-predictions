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
- **Feature-group ablation** (`feature_groups/`): partitions the candidate pool into
  mutually-exclusive content families and runs the full 2^G subset sweep over
  season-blocked rolling-origin CV, decomposing skill into per-group **Shapley main
  effects** and **pairwise Shapley interaction indices** (estimator-agnostic, with
  XGBoost/BART fold scorers). Answers "which families work together?" *directly* instead
  of inferring it from add-one-family runs — see the
  [feature-group ablation result](#feature-group-ablation-are-the-feature-families-complementary-or-redundant).

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
| 12 | 2026-06-12 | **BART** on the run-11 Phase 3 feature set | 51 | 0.699 | 0.651 | 0.2207 |
| 13 | 2026-06-12 | Extended RFE (30 iter, curve collapses at ~15), tolerance rule picks 17, XGBoost | 17 | 0.688 | 0.632 | 0.2231 |
| 14 | 2026-06-13 | **BART backward-elimination** on BART's own `variable_inclusion` (peak at 32), BART | 32 | 0.705 | 0.658 | 0.2190 |
| 15 | 2026-06-16 | **Full set (rank + aggregated)** — `RANK_ONLY=False`, ~2,349-col pool, XGBoost | 31 | 0.698 | 0.639 | 0.2204 |
| 16 | 2026-06-16 | **BART** on the full-set pool (148-feature pre-filter; validation peak 119) | 119 | 0.699 | 0.645 | 0.2213 |
| 17 | 2026-06-17 | **Brier + 1-SE selection** (vs AUROC + 0.005 tol), rank-only, XGBoost | 25 | 0.700 | 0.645 | 0.2200 |
| 18 | 2026-06-17 | **Brier + 1-SE selection**, full set, XGBoost | 55 | 0.698 | 0.649 | 0.2205 |
| 19 | 2026-06-17 | **Brier + 1-SE selection**, rank-only, BART | 26 | 0.703 | 0.646 | **0.2192** |
| 20 | 2026-06-17 | **Brier + 1-SE selection**, full set, BART | 62 | 0.697 | 0.642 | 0.2207 |
| 21 | 2026-06-17 | **Situational play-call** (down×distance tendency + play-type-split execution + snap-share), rank-only, XGBoost | 24 | 0.697 | 0.638 | 0.2209 |
| 22 | 2026-06-17 | **BART** on the situational rank-only pool | 61 | 0.696 | 0.649 | 0.2209 |
| 23 | 2026-06-17 | **Recency** (EWMA halflife 3 + rolling 4, every cumulative-average stat), rank-only, XGBoost | 36 | 0.693 | 0.627 | 0.2321 |
| 24 | 2026-06-17 | **BART** on the recency rank-only pool | 58 | 0.691 | 0.633 | 0.2223 |
| 25 | 2026-06-18 | **Madden ratings** (player overall ratings injected as team-week features), rank-only, base-vs-madden ablation (all groups vs all − madden) | 6160→6348 | +0.0075 delta | — | −0.0016 delta |
| 26 | 2026-06-19 | **Madden ratings, `_ovr` z-scored within season** (removes cross-era ratings drift; diffs kept raw-point), same base-vs-madden ablation | 6160→6348 | **+0.0129 delta** | — | **−0.0034 delta** |
| 27 | 2026-06-20 | **Bayesian logistic regression** (PyMC, weakly-informative `Normal(0,1)` prior), same 54 features / split as run 6 | 54 | 0.695 | 0.657 | 0.2208 |
| 28 | 2026-06-24 | **Madden launch ratings through RFE** — rank-only pool + 188 `madden_*` (incl. 2025 launch data from Phase 0) as RFE candidates for the first time; brier-1SE, XGBoost | 51 | 0.699 | 0.637 | 0.2211 |
| 29 | 2026-06-24 | **Madden launch ratings, lean ablation-informed pool** — Madden LEVELS only + `schedule_points` + `pbp_phase2_directional` + context (1,954-col pool); brier-1SE, XGBoost | 25 | 0.700 | 0.642 | 0.2209 |

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
- **Run 11 → 12:** BART on the same 51-feature Phase 3 set came back **0.699 / 0.651 /
  Brier 0.2207 — below its run-10 result** (0.708 / 0.654 / 0.2185) by more than the
  ~±0.004 sampler wobble. An instructive reversal: the feature set that finally moved
  XGBoost *hurt* BART, plausibly because RFE selects with XGBoost importances — the
  estimators disagree about which correlated rank families they can exploit.
  **The champion remains run 10's BART on the 57-feature directional set**; run 11's
  XGBoost (0.707/0.2206) is now a close second. Width-stratified Brier stayed strictly
  monotone (0.174 → 0.250), so run 12's posterior uncertainty remains trustworthy even
  at its lower skill. Open question for a future run: estimator-specific feature
  selection (RFE with BART importances, or BART on the run-9 57-set ∪ Phase 3 trench
  picks).
- **Run 12 → 13:** extended RFE from 20 to 30 and then 40 iterations so the CV curve
  actually collapses instead of stopping while flat. The full descent (down to 2
  features): CV AUROC peaks at **0.6835 with 32 features**, holds above 0.67 through
  13, breaks at 11 (0.659), and slides to 0.569 at 2 — so the usable range is roughly
  13–60 features with a peak at 32, and single-digit sets are ruled out by CV alone. The established tolerance
  rule (smallest set within .005 of best) therefore picked **17 features** — but the
  hold-out disagreed sharply: 0.688 / 0.632 / Brier 0.2231, well below run 11's
  51-feature model (0.707/0.649/0.2206). **Methodology lesson:** when the RFE curve is
  flat across a wide range, CV cannot distinguish set sizes and "smallest within
  tolerance" over-shrinks — small sets carry hold-out variance that CV doesn't price.
  Future selections should prefer the CV-peak count (or a 1-SE-style rule) over
  aggressive minimalism. 9 of the 17 survivors were play-by-play features (4 Phase 3,
  3 directional, 2 Phase 1), consistent with the pbp families carrying real signal.
- **Run 13 → 14:** estimator-specific selection at last — `BartBackwardElimination`
  (`data_science_utilities/models/bart/`) drives elimination by BART's OWN
  `variable_inclusion`, re-measured every iteration, with six parallel seed-replicate
  fits to tame PGBART noise (the replicate spread was ~0.007 per iteration; averaging
  six cuts it to ~0.003). Starting from a generous 91-feature pre-filter and selecting
  by the validation-curve peak (the run-13 lesson), **BART's curve peaks at 32
  features** — the *exact* count XGBoost's CV curve peaked at, reached by a completely
  different mechanism (gain-RFE vs inclusion backward-elimination). The two 32-sets
  overlap only 17/32, though: same dimensionality, different composition — and BART's
  leans far more on box-score/aggregate ranks (23 of 32) than XGBoost's pbp-heavy set.
  Hold-out: **ROC-AUC 0.705, accuracy 0.658, Brier 0.2190.** This decisively beats
  run 12 (BART on the XGBoost-selected 51-set: 0.699 / 0.2207), confirming that run-12's
  regression was largely a selection-mismatch artifact — giving BART features chosen by
  BART recovers it. It also essentially ties the run-10 champion (0.708 / 0.2185) within
  PGBART's ~±0.004 wobble, but with **25 fewer features** — the more parsimonious model
  for the same skill. Width-stratified Brier is monotone except a minor q3/widest
  inversion at the noisy tail (0.167 / 0.218 / 0.248 / 0.243). Takeaway: ~32 features is
  a real signal core both estimators find independently; estimator-matched selection
  matters for BART; and we are firmly on a ~0.705–0.708 / ~0.219 Brier plateau that five
  feature generations and two estimators have not broken — the ceiling now looks like a
  data/regime limit, not a feature-engineering one.
- **Run 14 → 15/16 (full ranked + aggregated set):** tested whether adding the aggregated
  cumulative-average / sum / delta columns *back* alongside the ranks helps — `RANK_ONLY =
  False`, a **~2,349-column pool vs rank-only's 1,529** (the extra ~820 are the continuous
  aggregated *values* the rank-only experiment had dropped as collinear with their ranks).
  XGBoost RFE on the full pool selected **31 features (CV peak), hold-out 0.698**; BART
  backward-elimination on a generous 148-feature pre-filter peaked at **119 features,
  hold-out 0.699** (a cheaper 49-feature pre-filter gave 0.698 at 40 features — consistent).
  **All three cluster at 0.698–0.699, ~0.006–0.008 below the rank-only champions** (XGBoost
  run 11 0.707, BART run 14 0.705) — no lift. What *did* change is composition: with the
  aggregated columns available the two estimators disagree sharply (BART's 40-feature set
  overlaps XGBoost's 31 by only 25, **Jaccard 0.54**), and **10 of BART's 15 unique picks
  are uncorrelated (<0.5) to anything XGBoost chose** — genuinely different signal, not
  correlated substitutes — yet hold-out is identical. This is the strongest form of the
  long-running crowding-out result: the aggregated values carry **no incremental hold-out
  signal beyond their ranks**, for either estimator, so rank-only stays the default.
  > Process notes: the null isn't a pre-filter artifact — BART's validation peak sat at 119
  > of its 148-feature pool (it keeps most of what it's given) and the 49→148 pool change
  > moved hold-out only +0.001. Also, an 86-feature BART attempt looked "intractable" only
  > because the laptop slept on battery; on AC each BART fit is ~3 min regardless of 49 vs
  > 148 features (BART cost here is dominated by MCMC sampling, not the split search).
  > Reproduce: `RANK_ONLY = False` in `cross_validation/rfe.ipynb` (writes
  > `rfe_features_kfolds_full.csv`); `cross_validation/grid_search.ipynb` with
  > `SELECTED_RFE_CSV = 'rfe_features_kfolds_full.csv'`, `BEST_NUM_FEATS = 31`;
  > `cross_validation/bart_rfe.ipynb` with `START_POOL_SIZE = 148`.
- **Runs 17–20 (Brier + 1-SE selection):** switched the selection objective from ROC-AUC
  (a pure *ranking* metric, blind to calibration) to **Brier** (a proper scoring rule that
  rewards calibration + sharpness), and replaced XGBoost's smallest-within-0.005-tolerance
  rule and BART's pure-peak with a shared, direction-aware **1-SE rule** (most parsimonious
  set within one standard error of the best). The metric is now a single configurable
  `SELECTION_METRIC` knob in all three notebooks (`'brier'` default; `'roc_auc'` /
  `'log_loss'` switchable), and the XGBoost RFE class gained a `brier` metric + per-fold
  storage + `get_best_num_features_1se()`, BART a `get_best_features_1se()` (all unit-tested).
  Result: **the plateau is metric-robust.** All four runs land at ~0.697–0.703 ROC-AUC /
  0.2192–0.2207 Brier — statistically indistinguishable from the AUROC-selected models, just
  often more parsimonious (BART rank-only: 26 features vs run-14's 32). The best Brier of the
  four, **run 19 (BART rank-only, 0.2192)**, ties the AUROC-selected run-14 BART (0.2190) and
  does not reach run-10's 0.2185 — i.e. selecting *for* calibration did not buy measurably
  better calibration. The full-set runs (18, 20) again trail their rank-only counterparts on
  Brier (0.2205 / 0.2207), consistent with the run-15/16 null on aggregated features.
  > BART-SE caveat: BART's RFE has no cross-validation, so its 1-SE band uses the spread of
  > the 6 sampler-seed replicates — that captures sampler noise (~0.003), not generalization
  > variance, making the band tight (near-peak). The validation Brier curve is also extremely
  > flat (run 19: 0.2297–0.2319 across 91→10 features), so the *count* is only loosely
  > determined; what RFE mainly fixes here is the feature *composition* (via the inclusion
  > ranking), not a sharp elbow. Reproduce: set `SELECTION_METRIC = 'brier'` in the three
  > `cross_validation/` notebooks; artifacts are suffixed `_brier`.
- **Runs 21–22 (situational play-call features):** added a down×distance play-call family —
  pass/run **tendency** by down (1st; 2nd/3rd × short ≤2 / medium 3–6 / long 7+) and
  goal-to-go, **execution split by play type** (`success`/3rd-down `conversion` *on passes*
  vs *on runs*), plus a **wp-context snap-share** family (how often a team's games are
  competitive vs blowouts) — all × 3 contexts × 4 perspectives. Candidate pool grew
  2,349 → 3,249 (+900); new raw columns `ydstogo`/`goal_to_go`; era-safe; unit-tested
  (`src/data/play_by_play/collect.py`). **Both estimators select these features heavily** —
  XGBoost kept **6 of 24** at the 1-SE point, BART **18 of 61** (30%; it especially liked the
  snap-share garbage-time shares and the 3rd-and-short conversion-by-play-type splits) — a far
  higher survival rate than the run-15 aggregated-values null, so they carry real CV signal.
  **But neither moved the hold-out:** XGBoost 0.6966 / 0.2209 (vs run-17 rank-only-no-situational
  0.7005 / 0.2200) and BART 0.6965 / 0.2209 (vs run-19 0.7030 / 0.2192) — both marginally *below*
  their no-situational baselines, and below the 0.705–0.708 champions. Interpretation: the
  situational signal is real in-sample but redundant-for-prediction at the team-week-rank grain
  (selected as substitutes, not additive), and/or the 544-game two-season hold-out is too small
  to resolve it. **Six feature generations and two estimators have now held the
  ~0.705 / ~0.219 plateau** — strong evidence the ceiling is a data/regime limit, not a
  feature-engineering one; the next lever is market signal, not more box/PBP families.
  > Repro note: feature code is on `extend-rfe-and-bart-rfe`; regenerate caches with
  > `get_play_by_play_data(range(2003,2026), refresh=True)` then rebuild via
  > `scripts/data_assembly/predict_game_winner/schedule_and_weekly.py`. The BART run can hit an
  > intermittent PyTensor compile-cache race on its first parallel iteration (`AssertionError`
  > in `cmodule.py`); a re-run on the now-warm cache clears it.
- **Runs 23–24 (recency: EWMA + rolling):** the one remaining non-market lever — a new
  *temporal* axis. Added EWMA (halflife 3) and rolling (last 4) versions of **every**
  cumulative-average stat (weekly box, schedule points, PBP rates) *alongside* the season
  averages, each ranked, so RFE picks form-vs-season-average per stat (pool 3,249 → 9,297;
  rank-only 2,129 → 6,161; the 215 MB regenerated parquet is now gitignored). **Recency is the
  most CV-attractive family of the whole project — both estimators select it at ~50%** (XGBoost
  18/36, BART 29/58) — yet it is the only family that **actively degrades the hold-out**:
  XGBoost **0.693 / 0.2321** (vs no-recency run-21 0.6966 / 0.2209), BART **0.691 / 0.2223**
  (vs run-22 0.6965 / 0.2209) — the worst Brier of any rank-only run, on *both* estimators.
  Interpretation: recency-weighting trades bias for **variance** (small effective sample), so
  recent form looks predictive in CV but chases noise that doesn't generalize on the 544-game
  hold-out. **The non-market feature levers are now exhausted** — aggregated values (runs
  15/16, null), situational play-call (runs 21/22, null), recency (runs 23/24, negative), and
  matchup-diffs (dropped as redundant with ranks). Across **seven feature generations and two
  estimators** the ~0.705 / ~0.219 ceiling has not moved; it is a data/regime limit. The only
  untried lever is **market signal** (de-vigged moneyline-implied probability as a feature and
  baseline), deferred by choice.
  > Repro: `SELECTION_METRIC = 'brier'` in the `cross_validation/` notebooks on the
  > `recency-ewma-features` branch; rebuild the dataset via the assembly script (recency is
  > derived post-cache — no PBP re-download).
- **Run 25 (Madden ratings, base-vs-madden targeted ablation, 2026-06-18):** injected
  Madden player overall ratings as team-week features: per-slot individual ratings
  (QB, RB, TE, WR1/2/3, LT/LG/C/RG/RT, edge left/right), group means (backfield,
  receivers, interior/exterior OL, edge, interior DL, linebackers, cornerbacks, safeties),
  within-season diffs (prev game, 4-game), season-over-season diffs, and target-vs-opp
  matchup deltas (pass-pro edge, interior, skill-cover, pass-rush) — 188 madden columns,
  classified as the `madden_ratings` content family. Coverage: 69–78% non-null on _ovr
  columns in 2009–2024; 0% in 2025 (new nflverse depth-chart schema breaks starter lookup).
  **Targeted ablation (all 7 groups vs all 7 − madden_ratings, rank-only XGBoost, 7 folds
  2019–2025):** mean Brier 0.2283 (with) vs 0.2299 (without), **delta −0.0016 (madden
  helps)**; mean ROC-AUC 0.6693 (with) vs 0.6617 (without), **delta +0.0075**.
  6/7 folds positive on Brier, 6/7 positive on ROC-AUC. **Madden ratings add a consistent
  but modest signal** on this untuned fixed-model scorer — they are the first non-market
  data source in this project to produce a reliably positive Brier delta across folds.
  Note: these are *untuned* XGBoost scores on the ablation metric, not the tuned hold-out
  table scores; the actual hold-out improvement after RFE may be larger, smaller, or null
  (RFE may not select the same madden columns that are carrying the signal).
  > Repro: `conda run -n nfl-predictions python scripts/experiments/madden_ablation.py`
  > on the `madden-ratings-features` branch; artifacts in
  > `data/predict_games/group_ablation/madden_ablation.json`.

- **Run 26 (Madden ratings, `_ovr` z-scored within season, 2026-06-19):** the raw Madden
  overalls carry a real **cross-era scale drift** (league-mean overall ≈ 84 in 2004–08 vs
  ≈ 77 recently), so the absolute level means different things across eras while the model
  trains across all of them. Fix: z-score each `_ovr` *level* within season (mean 0, std 1);
  the `_diff_*` columns are left as raw overall points (differences are already era-invariant).
  **Same targeted ablation:** ROC-AUC delta **+0.0075 → +0.0129** (Madden lift +72%), Brier
  delta **−0.0016 → −0.0034** (≈2×); still 6/7 folds positive, and **2021 flipped from
  hurting (−0.0087) to helping (+0.0087)**. The only non-helping fold is 2025 (no Madden
  starter data — nflverse depth-chart schema change). Removing the era drift roughly doubled
  the signal — strong evidence the level non-stationarity, not the talent signal, was the
  limiter. Same untuned-scorer caveat as Run 25 applies.
  > Both results on disk: `madden_ablation_raw.json` (Run 25) and
  > `madden_ablation_zscore.json` (Run 26) under `data/predict_games/group_ablation/`.
- **Run 27 (classic parametric Bayesian baseline, 2026-06-20):** added a **Bayesian logistic
  regression** (PyMC 5, NUTS) as a linear counterpoint to BART, trained and evaluated on the
  *identical* protocol as run 6 — same parquet, seed-32 shuffle, frozen **54-feature** run-6
  set, train `<2022` / 2024+2025 hold-out, per-row metrics. Features are median-imputed
  (leakage-safe, fit on the train fold; the 54-set is NaN-free so imputation is inert here)
  and z-scored; the posterior is exported as a per-coefficient JSON summary (mean/sd +
  standardizer moments) to seed a later weekly-Madden transfer-learning prior. Two priors:
  - **Weakly-informative `Normal(0, 1)` (reported):** hold-out **ROC-AUC 0.695**, accuracy
    **0.657**, **Brier 0.2208**, log loss 0.631; per-week mean AUROC 0.701. Sampler
    diagnostics are clean — **R-hat 1.000, 0 divergences, min ESS ≈ 9,000** (4 chains ×
    2,000 draws). The Bayesian payoff holds: the posterior-width-stratified Brier is
    **strictly monotone** — **0.174 / 0.220 / 0.233 / 0.256** from narrowest to widest
    posterior quartile — so the per-row predictive uncertainty is trustworthy, and the
    coefficients are interpretable and correctly signed (home field **+0.30** on the logit
    is the single largest effect).
  - **Regularized horseshoe (excluded):** failed to converge — **2,062 divergences, R-hat
    1.53, ESS ≈ 7** at `target_accept=0.95`. Its 0.649 hold-out number is **not trustworthy**
    and is reported only as a diagnostics failure; a reparameterization (or much higher
    `target_accept` / longer tuning) would be needed. The trace is saved for inspection.
  - **Verdict — competitive-adjacent, but the trees still win.** On identical inputs the
    linear Bayesian model lands **~1 ROC-AUC point below** BART run 6 (0.705) and run 10
    (0.708) and the best XGBoost (run 11, 0.707). The gap is consistent with the
    linear-additivity assumption — a GLM cannot represent the feature interactions the trees
    split on. Its distinguishing value is **well-calibrated, monotone uncertainty** and
    **transparent coefficients**, and its saved `Normal(0,1)` posterior is the foundation for
    the planned weekly-Madden transfer-learning update. Reproduce with
    `python scripts/experiments/bayes_logistic.py` (~5 min, seed 32); artifacts land in
    `data/predict_games/bayes_logistic/` (the multi-hundred-MB `.nc` traces are git-ignored;
    the compact `coef_summary_*.json` files are the committed transfer-learning surface).

- **Phase 0 — 2025 launch-ratings data layer (data-quality note, 2026-06-23,
  `madden-launch-ratings`):** not a numbered modeling run — a data-plumbing phase that makes
  2025+ Madden *launch* ratings flow through the existing `src/data/madden/` pipeline. The
  documented "2025 = 0% coverage" gap was never a ratings problem: nflverse changed the
  2025+ depth-chart schema (dated snapshots, granular `pos_abb`, no
  `game_type`/`club_code`/`depth_team`), which broke weekly-starter detection. Phase 0 (a)
  sources 2025 launch ratings from the user's `madden-tools` CDN `players.json` (the
  `label=="Launch"` iteration), bridged to gsis via **nflverse seasonal rosters**; (b)
  normalizes the new depth schema (latest **pre-game-day** snapshot — leakage-safe — with
  positions coarsened to the pre-2025 vocabulary so 2025 matches the all-old-schema training
  distribution); (c) routes by season (≤2024 keeps the existing theedgepredictor source + processed/ gsis bridge; the shared `available_starters` core is reused unchanged). **Measured
  result: 2025 `_ovr` coverage 0% → 76.5%**, inside the 66–78% historical band (2024 = 0.771
  unchanged); 2025 gsis match 0.833 (a file-composition effect — the full 3,067-player Madden
  file includes camp bodies with no nflverse id; starters match fine). Reproduce with
  `MADDEN_TOOLS_CDN_BASE=https://cdn.madden.tools python -m scripts.experiments.madden_coverage_report`;
  numbers in `data/predict_games/madden_coverage/coverage_report.json`. **Run 28 is reserved
  for Phase 1** (the first XGBoost-RFE run that puts launch features through feature selection).
- **Run 28 (Madden launch ratings through RFE — Phase 1, 2026-06-24, `xgb-launch-ratings`):**
  the **first time Madden features entered feature selection** (prior Madden evidence, Runs
  25–26, was forced-block ablation only). RFE ran on the rank-only pool **+ the 188 `madden_*`
  columns** (6,348 candidates incl. the now-populated 2025 launch ratings), brier-1SE →
  **51 features**, then the standard XGBoost random search on the seed-32 / train`<2022` /
  valid`22–23` / 2024+2025 hold-out (1,088 rows). Eval reuses the Run-27
  `bayes_logistic.evaluate` suite for direct comparability.
  - **Madden SURVIVES selection (the positive result):** **13 of the 188 `madden_*` made the
    51-feature set** — and they are the sensible ones, led by **both starting-QB ratings ranked
    #2 and #3 by total-gain importance** (`target_madden_qb_ovr`, `opp_madden_qb_ovr`), plus the
    trench block (`target_madden_edge_ovr`, `opp/target_interior_ol`, `target_exterior_ol`,
    `madden_matchup_pass_pro`), receivers, corners, and the backfield. This is the first
    evidence the launch ratings hold up as *selected* features, not just as a forced block.
  - **But the selected model does NOT beat the champion (the honest null):** hold-out
    **ROC-AUC 0.699 / accuracy 0.637 / Brier 0.2211** — ~0.6–0.9 AUROC points below XGBoost
    run 11 (0.707), BART run 6 (0.705) and run 10 (0.708); Brier ≈ run-11 XGBoost (0.2206) and
    a touch worse than the BART champions (0.2185–0.2194). The reliability curve is monotone and
    reasonably calibrated.
  - **Interpretation:** Madden talent is real signal the selector keeps and the model leans on
    (the QBs are top-3), yet it does not lift XGBoost's *selected* 2024+2025 hold-out past the
    champion — the same ceiling Runs 21–24 documented (added families are selected but don't move
    the regime-limited hold-out). Note the comparison is a single madden-inclusive run vs the
    *recorded* champions (not a same-snapshot with/without-madden A/B), so the 0.008 AUROC gap
    can't be cleanly attributed to Madden; it is consistent with "survives selection, doesn't
    break the ceiling." Contrast Run 26's forced-block ablation, where Madden helped on 7-fold
    rolling CV (+0.0129 AUROC) — CV folds, not the specific 2024+2025 hold-out. Reproduce:
    `MADDEN_TOOLS_CDN_BASE=https://cdn.madden.tools python -m scripts.data_assembly.predict_game_winner.schedule_and_weekly`
    then `python -m scripts.experiments.xgb_launch_ratings --stage rfe` and
    `... --stage grid --best-num-feats 51`; artifacts in `data/predict_games/xgb_launch_ratings/`
    and `models/best_random_xgb_model_launch_ratings.json`. (This phase also fixed a latent
    crash in the shared CV-RFE library when importance pruning stalls — see the PR.)
- **Madden sub-structure ablations (Phase 1, 2026-06-24).** Two follow-up ablations dissected
  *where* the Madden signal lives, decomposing the 188-column family two ways
  (`scripts/experiments/madden_internal_ablation.py`; results under
  `data/predict_games/group_ablation/madden_internal_*.json`):
  - **By unit** (QB / offense-skill / O-line / D-front / coverage / matchup): **the QB unit
    dominates** — Shapley **+0.0107** and unique leave-one-out **+0.0088**, ~7× any other unit;
    every other unit is small and several have negative leave-one-out (redundant).
  - **By measure** (talent `_ovr` levels vs the `_ovr_diff_*` momentum features): **levels carry
    the signal** (Shapley +0.0139, LOO +0.0109); the 138 momentum/diff columns are **redundant**
    (LOO −0.0012, level×diff interaction −0.006). So ~180 of the 188 Madden columns are dilution
    around a QB-talent-level core.
  - **Cross-family pairing** (`run_group_ablation.py` with `MADDEN_LEVELS_ONLY=1`, 8 families incl.
    a 50-column Madden-levels block; `group_ablation_brier_madden_levels.json`): **Madden levels
    are the single most valuable family** (Shapley +0.0056; the only family with meaningful unique
    value, LOO +0.0038 — every PBP family is negative/redundant). But **all Madden×family
    interactions are sub-additive** (no synergy): Madden *substitutes for* the other families'
    team-quality signal rather than complementing it.
- **Run 29 (Madden, lean ablation-informed pool — Phase 1, 2026-06-24):** acting on those
  ablations, RFE ran on a much smaller pool — Madden **levels only** + the two families with
  unique value (`schedule_points`, and `pbp_phase2_directional` chosen over `box_score` as the
  PBP representative) + context (1,954 cols vs Run 28's 6,348), brier-1SE → **25 features**.
  - **Madden now dominates the selected model: 16 of 25 features (64%)** are Madden levels, with
    **`opp_madden_qb_ovr` the single most important feature** and `target_madden_qb_ovr` #5 —
    up from 13/51 (25%) in Run 28's noisy full pool.
  - **But the hold-out is unchanged: ROC-AUC 0.700 / accuracy 0.642 / Brier 0.2209**, statistically
    identical to Run 28 (0.699/0.2211) and still ~0.7 AUROC points below the champions (XGB 11
    0.707, BART 10 0.708; Brier ≈ XGB 0.2206). A more parsimonious model (25 vs 51 feats) at the
    same Brier, not a better one.
  - **Verdict (Phase 1, honest):** Madden launch ratings carry **real, selectable signal,
    concentrated in QB talent levels** — a clean Madden-heavy model predicts nearly as well as the
    champion from half the features. But Madden does **not break the documented hold-out ceiling**
    (Runs 21–24): being *sub-additive*, it substitutes for the existing team-quality signal instead
    of adding to it, so neither the full-pool (Run 28) nor the de-noised lean-pool (Run 29) model
    beats the champion. Reproduce Run 29 with `--stage rfe --tag _run29 --madden-levels-only
    --families madden_ratings,schedule_points,pbp_phase2_directional,context_rest` then
    `--stage grid --best-num-feats 25 --tag _run29`.

## Feature-group ablation: are the feature families complementary or redundant?

Runs 5–24 add one feature *family* at a time and let RFE pick columns, which can only
*infer* redundancy run-by-run (a family is selected heavily yet the hold-out doesn't move).
A dedicated harness (`data_science_utilities/feature_groups/`) measures it directly: it
partitions the ~9,297-column pool into mutually-exclusive **content families**, scores the
**full 2^G subset sweep** with a *fixed, untuned* regularised XGBoost per subset (no
per-subset RFE/tuning, so the *group* effect is isolated from the *selection* effect),
evaluates on **season-blocked rolling-origin CV** (test = each of 2019–2025; the two prior
seasons are the early-stopping window), and decomposes skill into per-group **Shapley main
effects** and **pairwise Shapley interaction indices**.

The 2026-06-17 run swept the **7 families present at that time** — `box_score`,
`schedule_points`, `pbp_phase1/2/3`, `situational_playcall`, `snap_share` — over
`context_rest` as an always-on base (128 subsets × 7 folds = **896 fits**, Brier). These
numbers are rolling-origin CV on an *untuned* model and are **NOT comparable to the tuned
hold-out table above — read the deltas between subsets, not the absolute level.**

The 2026-06-18 run (Run 25 / `madden-ratings-features` branch) introduced an **8th family —
`madden_ratings`** (188 columns, player Madden overall ratings as team-week features).
Rather than re-run the full 2^8 = 256-subset sweep (~60 min), a targeted base-vs-madden
comparison was run directly: full-8-group score vs full-7-group (madden excluded) over the
same 7 folds. **Result: madden_ratings reduces mean Brier by 0.0016 (positive) and increases
mean AUROC by 0.0075 (positive); 6/7 folds agree on both metrics.** See Run 25 note above for
detail. Unlike the 7 original families which are all sub-additive with each other, madden
ratings add a *complementary* signal not already captured by box/pbp/schedule families —
suggesting player quality is a partially orthogonal dimension to team-game performance.

- **Every one of the 21 pairwise interactions is negative (sub-additive); zero are
  positive.** No pair of families complements another. Most sub-additive:
  `box_score + pbp_phase1` (−0.0040); least: `pbp_phase2_directional + pbp_phase3` (−0.0024).
- **Standalone vs leave-one-out is the redundancy fingerprint.** Each family lowers Brier by
  ~0.016–0.022 *on its own*, but each family's *marginal* value once the other six are
  present is ≈ 0 (several slightly negative) — a redundancy of ~0.018–0.021 for **all
  seven**. They are mutually interchangeable carriers of the same signal.
- **Parsimony beats the kitchen sink.** Base-only (8 context cols) Brier 0.2507 → best single
  family ~0.231–0.234 → the best subset is just **3 families**
  (`box_score + pbp_phase1 + schedule_points`, 0.2273); the best overall is a 4-family set
  (0.2271). The **full 7-family set (0.2299) ranks only 72/128** — adding families past the
  first ~3 is mildly *worse*, not better. ~80% of the achievable improvement comes from the
  first family added.
- Shapley main effects are all small and similar (+0.0019 to +0.0046), with `pbp_phase1` the
  largest individually.

This is the crowding-out the experiment log inferred for seven feature generations, now
measured directly across all families at once: the box/pbp/schedule families are **tapped
out**. **Caveats:** on a metric sitting at the documented ~0.219-Brier plateau, a negative
interaction is consistent with *both* genuine informational redundancy *and* metric-ceiling
saturation — the index cannot separate them (corroborate with feature correlations or
re-run in log-loss space); and the cross-fold SE is a *lower bound*, because the
expanding-window folds share nested training data and overlapping validation windows.
**Implication:** the only sources likely to carry *orthogonal* signal are the two **not yet
in the pool** — **market** (de-vigged odds; the `schedule` collector supports
`keep_odds=True`) and **Next Gen Stats** (player-tracking metrics; collected but not
integrated, and blocked on the season-average `_diff` leakage fix plus team-week
aggregation). Reproduce: `cross_validation/group_ablation.ipynb` or
`cross_validation/run_group_ablation.py`; artifacts in `data/predict_games/group_ablation/`.

**Madden ratings (Run 25, `madden_ratings` family):** unlike the original 7 families,
Madden adds a *genuinely complementary* signal (Brier −0.0016, ROC-AUC +0.0075, 6/7 folds
positive). The targeted ablation indicates player Madden ratings capture something the
box/pbp/schedule metrics miss — plausibly pre-season talent assessments that are orthogonal
to in-season form. The full Shapley decomposition with all 8 families is pending (256-subset
× 7-fold sweep, ~60 min on a laptop).

## Roadmap

- Improve predictive performance toward / past a Vegas-implied baseline. Five feature
  generations (runs 5–14) across two estimators have plateaued at ~0.705–0.708 ROC-AUC /
  ~0.219 Brier, and the [feature-group ablation](#feature-group-ablation-are-the-feature-families-complementary-or-redundant)
  now confirms quantitatively that all seven in-dataset families are mutually redundant.
  So the priority is the two **untapped data sources** that might carry orthogonal signal:
  **market** (de-vigged moneyline-implied probabilities — ≈ a `keep_odds=True` reassembly,
  the `schedule` collector already supports it) and **Next Gen Stats** (needs the `_diff`
  leakage fix and team-week aggregation below before it can join as a family), rather than
  more box/pbp features.
- Calibration pass on the run-10 / run-14 BART models (best AUROC/Brier but 0.5-threshold
  accuracy lags — calibration may recover it) and Next Gen Stats integration at the
  weekly grain.
- Faster estimator-matched selection (run 14 follow-up): a cheap 1–2-replicate BART
  backward pass for the ranking + curve shape, then a fully-parallel multi-replicate
  size-sweep in the ~24–45 promising region, instead of replicating every step.
- Extend the Bayesian comparison: the BART baseline (run 6) and a **parametric Bayesian
  logistic regression** (run 27) are both done. Next: fix the regularized-horseshoe
  convergence (reparameterize / raise `target_accept`), and build the weekly-Madden
  **transfer-learning** model that seeds its prior from run 27's saved coefficient posterior.
  Posterior uncertainty (now shown to be monotone/trustworthy) feeds bet-sizing-style
  decision rules.
- Address known `TODO`s: prevent season-average leakage in the NGS diff features, and move
  notebook-style execution out of `next_gen_stats/collect.py` import path.
