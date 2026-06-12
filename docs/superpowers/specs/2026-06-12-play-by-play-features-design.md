# Play-by-Play Features for Game Winner Prediction — Design

**Date:** 2026-06-12
**Status:** Approved
**Baseline to beat:** pooled 2024–2025 hold-out AUROC 0.697 / accuracy 0.647 (Run 5, rank-only RFE)

## Overview

Extend the prediction pipeline with team-week features derived from nflfastR
play-by-play data (`nfl_data_py.import_pbp_data`, ~50k plays x 396 columns per
season, available 1999+, used here for 2003+ to match the existing training
span). PBP enables per-play efficiency, situational, directional, and
context-split metrics that cannot be derived from weekly box-score totals.

Features are delivered in three phases, each ending with a full re-run of
cross-validated RFE so the lift of each feature family is measured
independently against the hold-out baseline.

## Goals

- Add PBP-derived team-week features that plug into the existing
  cumulative -> rank -> rank-change machinery and the existing
  target/opponent game-frame assembly with no changes to leakage handling.
- Split every metric by win-probability context: competitive,
  garbage-time-leading, garbage-time-trailing.
- Measure per-family lift on the pooled 2024–2025 hold-out.

## Non-Goals

- Player-level features (aggregation is team-level only).
- FTN/PFF charting data (play action, personnel, coverage) — not in
  nflfastR PBP; out of scope.
- Replacing the existing weekly box-score features.

## Architecture

`src/data/play_by_play/collect.py` grows from the current stub into three
responsibilities:

### 1. Collect

- `nfl.import_pbp_data(years=[season], downcast=False)` per season,
  immediately selecting only the ~40 required columns.
- The **aggregated** team-week frame for each season is cached to
  `data/play_by_play/aggregated/{season}.parquet`. Raw PBP is not cached.
- A `refresh` flag forces a re-pull (needed for the in-progress season).

### 2. Aggregate

- Each play is bucketed by win-probability context using the offense's `wp`:
  - `competitive`: 0.05 <= wp <= 0.95
  - `garbage_leading`: wp > 0.95
  - `garbage_trailing`: wp < 0.05
- Rationale for the three-way split: leading-garbage (clock-killing runs vs
  soft coverage) and trailing-garbage (pass-spam vs prevent defense) are
  opposite regimes; pooling them would mix their signals.
- Plays aggregate to one row per `[team, season, week, season_type,
  opponent_team]`. Offense comes from grouping on `posteam`, defense from the
  mirror groupby on `defteam`; the two merge on team-week.
- Output columns follow the existing naming convention: `off_*` and
  `def_opp_*`, with a context suffix, e.g.
  `off_epa_per_play_competitive`, `def_opp_success_rate_garbage_trailing`.
- **Component sums, not rates, are stored per team-week** (yards, plays,
  successes, EPA totals, dropbacks, explosive plays, etc.). Every cumulative
  rate is computed as `cumsum(numerator) / cumsum(denominator)` within
  `[team, season]` — never an average of weekly rates. This makes sparse
  cells (e.g., 2 garbage-time left-end runs in a week) contribute correctly
  to season-to-date ratios.
- Cells with a zero denominator are **NaN, not 0** (a team with no
  trailing-garbage snaps has no rate, not a rate of zero). XGBoost handles
  NaN natively; ranks are computed over teams that have values.

### 3. Transform

- Cumulative rates, league-wide ranks within `[season, week]` (offense
  descending where higher is better, defense ascending where lower-allowed is
  better), and rank changes within `[team, season]` — same semantics as the
  weekly module.
- The rank/rank-change helpers currently in `src/data/weekly/collect.py`
  (`add_rank_columns`, `_rank_change_frame`) are **extracted into a shared
  module** (e.g., `src/data/transformations.py`) and reused by both weekly
  and PBP collectors. Behavior-preserving refactor; existing weekly and
  schedule tests guard it.

### Team abbreviation alignment

nflfastR PBP uses era-specific abbreviations (`SD`, `OAK`, `STL`); the
pipeline normalizes to current ones (`LAC`, `LV`, `LA`) in
`src/data/schedule/collect.py`. That mapping moves into the shared helpers
module and is applied to `posteam`/`defteam` before aggregation, so all three
collectors use one copy. A test asserts no unmatched team keys after the
merge.

## Merge into the game frame

The PBP frame never knows about home/away or target/opp; it rides the
existing rails in `src/data/collect_all.py`:

1. PBP team-week frame joins the weekly team-week frame on
   `[team, season, week]` **before** any home/away logic.
2. The schedule merges the combined frame twice (on home team and away team);
   the prefix renames (`off_*` -> `off_home_*`, etc.) carry PBP columns along.
3. The target/opponent duplication (two rows per game, `target_*`/`opp_*`,
   `is_home_target`) is untouched.
4. The shift-by-one-week leakage guard matches on `target_*`/`opp_*`
   prefixes and therefore shifts the new columns automatically.

## Feature families by phase

All metrics are computed per context split (competitive / garbage_leading /
garbage_trailing), for offense and defense.

### Phase 1 — efficiency and situational (~240 candidate columns)

- EPA per play; pass EPA per dropback; rush EPA per carry
- Success rate (overall, pass, rush) — `success` column (EPA > 0)
- Early-down (1st/2nd) success rate
- Third-down conversion rate
- Red-zone TD rate, per red-zone drive (drives reaching `yardline_100 <= 20`)
- PROE: mean of `qb_dropback - xpass`

### Phase 2 — directional (original motivating idea, ~600 candidates)

- Run buckets: `run_location` x `run_gap` -> left/right end, tackle, guard,
  plus middle (7 buckets). Per bucket: avg yards, explosive rate
  (rush >= 10 yds).
- Pass buckets: `pass_location` x `pass_length` (6 buckets). Per bucket:
  avg yards, explosive rate (pass >= 20 yds).

### Phase 3 — trenches, luck, tendencies

- Sack rate and QB-hit rate per dropback (created by defense / allowed by
  offense)
- Rush stuff rate (carries <= 0 yards)
- Fumble recovery rate (`fumble_lost / fumble`) — turnover-luck regression
- CPOE; YAC over expected (actual YAC vs `xyac_mean_yardage`)
- Penalty rate and penalty yards per play
- Scramble rate, shotgun rate, no-huddle rate, seconds per play

## Era caveats

- `cpoe` and `xpass` are NaN before 2006 (no pass-location charting). Left
  as NaN; PROE and CPOE features are simply absent-as-NaN for 2003–2005.
- `wp` exists for all seasons in scope.

## Testing

Unit tests per family, modeled on `tests/data/weekly/test_collect.py`, using
synthetic PBP frames:

- Garbage-time bucketing at the 0.05/0.95 boundaries (three-way split).
- Cumulative-ratio math, including the sparse-bucket case (zero-denominator
  weeks -> NaN; later weeks recover the correct season-to-date ratio).
- Offense/defense mirroring: team A's offensive EPA in a game equals team
  B's defensive EPA allowed.
- Rank direction (1 = best) and range (1..teams with values).
- Historical team abbreviations map correctly; no unmatched keys post-merge.
- New columns follow the `off_*`/`def_opp_*` convention and are shifted by
  the leakage guard (integration test through `collect_all`).

## Evaluation per phase

1. `python -m scripts.data_assembly.predict_game_winner.schedule_and_weekly`
   regenerates the model-ready parquet and feature list.
2. Re-run cross-validated RFE
   (`notebooks/.../cross_validation/rfe.ipynb`) on the expanded candidate
   set (rank-only toggle as in Run 5).
3. Compare pooled 2024–2025 hold-out AUROC/accuracy to the 0.697 / 0.647
   baseline. Record results in the README results table.

## Risks

- **Sparsity**: garbage-time and fine-grained directional cells are thin,
  especially early in seasons. Mitigated by cumulative component-sum ratios,
  NaN semantics, and rank features; RFE discards what stays noise.
- **Feature explosion**: ~700 new candidates across all phases. Mitigated by
  the staged rollout (RFE per phase) and the rank-only candidate mode.
- **Memory**: raw PBP is large. Mitigated by per-season collection with
  column selection and caching only the aggregated frames.
