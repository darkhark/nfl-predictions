# nfl-predictions

An end-to-end machine learning pipeline that predicts NFL game outcomes from real
NFL data ([nflverse](https://github.com/nflverse) via
[`nfl_data_py`](https://github.com/nflverse/nfl_data_py)). The project covers the full
modeling lifecycle — data collection, leakage-aware feature engineering, feature
selection, hyperparameter tuning, probability calibration, and model evaluation — using
XGBoost, with Bayesian approaches planned.

It is a personal project built to apply production data-science methodology (the same
approach I use professionally — custom recursive feature elimination, cross-validation,
calibration, and drift-aware design) to football analytics.

## What it does

- **Target:** predict the winner of each regular-season NFL game (binary classification).
- **Data:** 21 seasons of real NFL data (2003–2023) assembled from schedules, weekly
  team/player stats, and Next Gen Stats.
- **Approach:** engineer prior-week team form features, select a compact feature set with
  a custom RFE routine, tune an XGBoost classifier, and calibrate the output probabilities.

> **Status:** the data pipeline, feature engineering, feature selection, tuning, and
> calibration utilities are complete and tested. Predictive performance on game outcomes
> is still being improved — NFL game prediction is a hard problem and the current model
> does not yet consistently beat a strong baseline. The pipeline and methodology are the
> finished, reusable part; model accuracy is an ongoing line of work (see [Roadmap](#roadmap)).

## Data sources

All data is pulled from `nfl_data_py`:

| Source | Module | Notes |
| --- | --- | --- |
| Schedules | `src/data/schedule/collect.py` | Game results, rest days, roof type, optional Vegas lines |
| Weekly stats | `src/data/weekly/collect.py` | Team-aggregated offensive/defensive box-score stats |
| Next Gen Stats | `src/data/next_gen_stats/collect.py` | Passing / rushing / receiving advanced metrics |
| Play-by-play | `src/data/play_by_play/collect.py` | Scaffolded for future week-level aggregation |

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
  candidate pool (~571 columns as of the latest run) to a compact selected set (~40).
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
notebooks/model_training/      # RFE, cross-validation, and grid-search experiments
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

All runs evaluate on a **2023 hold-out season** (every prior season is used for
train/validation). Cross-validated RFE selects the feature set, then a randomized
hyperparameter search tunes the XGBoost classifier. Two **out-of-sample** metrics are tracked:

- **Hold-out ROC-AUC (2023)** — ROC-AUC over all 2023 rows pooled (both target/opp
  perspectives). Reported as the per-week mean, which matches the single pooled curve to
  within ~0.001.
- **Hold-out accuracy** — `best_model.score` on the 2023 rows at a 0.5 threshold.

> **In-sample vs out-of-sample.** The RFE and grid-search *cross-validated* ROC-AUC
> (~0.677 for runs 3–4) is measured on the training seasons and is naturally higher than the
> 2023 hold-out ROC-AUC (~0.64); the ~0.03 gap is the normal generalization drop, not a
> regression introduced by tuning. An earlier "per-game AUROC" column was removed as
> misleading — it kept only the higher-probability row per game, which conditions on the
> model's own output and reads ~0.53–0.55 (near chance) for reasons unrelated to model quality.

| # | Date | Feature selection | Features | Hold-out ROC-AUC | Hold-out acc |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | 2024-07-23 | RFE (single split) | 28 | 0.629 | 0.597 |
| 2 | 2024-08-06 | Cross-validated RFE (`StratifiedKFold`) | 34 | 0.643 | 0.597 |
| 3 | 2026-06-11 | Cross-validated RFE + rank features | 40 | 0.643 | 0.599 |
| 4 | 2026-06-11 | Cross-validated RFE, **rank-only** (cumulative averages removed) | 32 | 0.642 | **0.618** |

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

## Roadmap

- Improve predictive performance toward / past a Vegas-implied baseline.
- Finish play-by-play and Next Gen Stats integration at the weekly grain.
- Add Bayesian models for comparison and uncertainty quantification.
- Address known `TODO`s: prevent season-average leakage in the NGS diff features, and move
  notebook-style execution out of `next_gen_stats/collect.py` import path.
