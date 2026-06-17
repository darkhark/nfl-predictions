# Recency features: EWMA + rolling-window versions of every cumulative-average stat

**Date:** 2026-06-17
**Status:** Approved design — ready for implementation plan
**Motivation:** Every "form" feature in the model is a flat season-to-date expanding mean —
Week 1 weighs the same as last week. Recency-weighting adds a genuinely new temporal axis
("current form"), the most promising remaining non-market lever after six feature generations
plateaued at ~0.705 AUROC / ~0.219 Brier. This adds **EWMA (halflife 3)** and **rolling
(last 4)** versions of every stat that currently has a `cumulative_average`, *alongside* the
cumulative averages, so RFE/BART pick form-vs-season-average per stat.

## Decisions (locked during brainstorming)

- **Methods:** both EWMA *and* rolling-window (let selection compare them).
- **Scope:** all cumulative-average stats — weekly box stats, schedule points, PBP rates.
- **Augment, not replace:** keep the cumulative averages; add recency beside them.
- **Horizons:** EWMA `halflife=3` games; rolling `window=4` games (`min_periods=1`).

## Background — current expanding-mean mechanics (the pattern recency extends)

- **Weekly box stats** (`weekly/collect.py::create_cumulative_columns`): per stat,
  `cumulative_sum = groupby([team,season])[stat].cumsum()`,
  `cumulative_average = cumulative_sum / game_count`, plus `_cumulative_average_change`
  (`.diff()`). Offense groups by `(team, season)`; defense by `(opp_team, season)`.
- **PBP rates** (`play_by_play/collect.py::_add_cumulative_rate_columns`): per
  `(metric, numerator, denominator)`, `cumulative_average = cumsum(num)/cumsum(den)` within
  `(team, season)` (offense) / `(opp_team, season)` (defense), per wp-context, with the
  zero-denominator → NaN rule.
- **Schedule points** (`schedule/collect.py`): cumulative avg points scored / allowed.
- **Ranks** (`transformations.add_rank_and_rank_change_columns`): every column matching
  `off_*_cumulative_average` / `def_opp_*_cumulative_average` gets a within-(season,week)
  `_rank` and `_rank_change` (offense ranked descending, defense ascending).
- **Leakage:** the whole feature frame is shifted one week forward in
  `collect_all._shift_data`, so a week-N row carries values through week N-1.

## Design

### Naming
For base feature `{prefix}_{stat}` (prefix ∈ `off`, `def_opp`; PBP also has the
`_{context}` suffix), add, beside the existing `_cumulative_average` family:
- `{...}_ewma_average` (+ `_rank`, `_rank_change`)
- `{...}_rolling_average` (+ `_rank`, `_rank_change`)

No `_change` variants for the recency families (the cumulative `_change` exists for
historical continuity; recency change is YAGNI).

### Weekly + schedule (simple means)
In the per-stat loop, compute per group (the same `groupby_columns` already used for
cumulative), on the game-ordered series:
- `ewma_average = grp[stat].transform(lambda s: s.ewm(halflife=3, adjust=True).mean())`
- `rolling_average = grp[stat].transform(lambda s: s.rolling(4, min_periods=1).mean())`
The frame is already `sort_values(groupby_columns + [game_count_col])`, so each group's
series is in game order. Grouping by `(team, season)` / `(opp_team, season)` resets recency
each season automatically.

### PBP rates (ratios of recency-weighted components)
In `_add_cumulative_rate_columns`, alongside the existing `off_cumulative` / `def_cumulative`
cumsum frames, build recency-weighted component frames per group and form the ratio:
- EWMA: `off_ewm = grp[component_cols].transform(lambda s: s.ewm(halflife=3, adjust=True).mean())`;
  `off_{metric}_{context}_ewma_average = off_ewm[f'{num}_{context}'] / off_ewm[f'{den}_{context}'].where(!=0)`.
- Rolling: `off_roll = grp[component_cols].transform(lambda s: s.rolling(4, min_periods=1).sum())`;
  rate `= off_roll[num]/off_roll[den].where(!=0)`. (Ratio of windowed *sums* = the last-4-game
  rate; EWMA uses `.mean()` ratio, where the shared weights cancel, giving the recency-weighted
  rate.)
Defense uses the `(opp_team, season)` grouping and the `DEFENSE_CONTEXT_SWAP`, exactly as the
cumulative path does. Snap-share (the cross-context ratio added in the situational work) also
gets ewma/rolling versions via the same cross-context-total ratio.

### Rank discovery (auto-rank the recency families)
Extend the column discovery in `weekly.add_rank_columns`, `get_play_by_play_features`, and the
schedule equivalent from `endswith('_cumulative_average')` to
`endswith(('_cumulative_average', '_ewma_average', '_rolling_average'))` for both the
`off_*` and `def_opp_*` lists. `add_rank_and_rank_change_columns` then produces `_rank` /
`_rank_change` for the recency families unchanged.

### Rank-only filter (rfe.ipynb) — generalize
The current filter keeps `f` if `'cumulative' not in f or f.endswith('_rank'|'_rank_change')`.
Recency raw values (`_ewma_average`, `_rolling_average`) contain no `'cumulative'`, so they
would leak through as *un-ranked raw columns*, defeating rank-only mode. Replace with a rule
that treats all three average families uniformly:
```python
RAW_AVERAGE_SUFFIXES = ('_cumulative_average', '_ewma_average', '_rolling_average',
                        '_cumulative_average_change')
keep = (not f.endswith(RAW_AVERAGE_SUFFIXES)) or f.endswith(('_rank', '_rank_change'))
```
i.e. drop every raw average / change column, keep ranks + rank-changes + non-average columns.
Verify the rank-only candidate count is unchanged for the *existing* families (regression
check) and now also excludes recency raw values.

## Feature count

Each stat's average family roughly triples (cumulative + ewma + rolling, each with
`_rank` + `_rank_change`). Candidate pool ~3,249 → ~8,000. Rank-only mode + RFE prune, as in
every prior generation.

## Regeneration (lighter than the situational work)

Recency is derived **post-cache**: PBP recency is computed in `get_play_by_play_features`
from the existing component caches (no raw re-download), and weekly/schedule recency at
build time. So regeneration = rebuild the merged dataset via
`scripts/data_assembly/predict_game_winner/schedule_and_weekly.py` (refetches the small
weekly per-season source files, reuses PBP component caches) + the feature list. No
`refresh=True` PBP download.

## Testing (unittest)

- **Weekly/schedule** (`tests/data/weekly/test_collect.py`, schedule test): on a synthetic
  per-team game-ordered series, `ewma_average` equals pandas `ewm(halflife=3).mean()` and
  `rolling_average` equals `rolling(4, min_periods=1).mean()`; values reset across a season
  boundary (a new season's game 1 is not influenced by the prior season).
- **PBP** (`tests/data/play_by_play/test_collect.py`): recency rate equals
  `ewm(num)/ewm(den)` and `rolling_sum(num)/rolling_sum(den)` on a synthetic component series;
  zero-denominator → NaN; defense context-swap holds.
- **Rank discovery:** recency columns receive `_rank` / `_rank_change`.
- **Rank-only filter:** keeps recency ranks, drops recency raw values, and leaves the
  existing-family rank-only count unchanged.

## Success criteria

- Recency (EWMA + rolling) columns exist for every cumulative-average stat across weekly,
  schedule, and PBP, with `_rank` / `_rank_change`, passing unit tests and no regression in
  existing features/tests.
- Rebuilt dataset + feature list carry the recency families (pool ~3,249 → ~8,000).
- (Separate follow-up) evaluate via rank-only RFE + grid/BART and record as the next run.
