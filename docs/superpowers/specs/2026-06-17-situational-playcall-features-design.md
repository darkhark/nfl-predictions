# Situational play-call play-by-play features

**Date:** 2026-06-17
**Status:** Approved design — ready for implementation plan
**Motivation:** The current PBP features capture per-play and directional efficiency and an
aggregate pass-rate-over-expected (`proe`), but **no down-and-distance-conditioned play-call
tendency or execution**. This adds: (a) pass/run **tendency** by down × distance, (b)
**execution split by play type** (success / 3rd-down conversion *on passes* vs *on runs*),
and (c) a **wp-context snap-share** family capturing how often each team's games are
competitive vs blowouts. All flow through the existing aggregation → cumulative-average →
rank → rank-change pipeline, across 3 wp-contexts × 4 perspectives.

## Scope

In scope: new per-play components + rate definitions in
`src/data/play_by_play/collect.py`, two new raw columns, a cache-version bump, and unit
tests. Out of scope: the downstream RFE/grid evaluation run (a separate follow-up), and
"predictability" features (deviation from league-average tendency — future, needs per-bucket
league baselines).

## Background — existing pipeline (must follow this pattern)

`src/data/play_by_play/collect.py` builds features as:
1. **Per-play component columns** — integer/float `(numerator, denominator)` pieces, e.g.
   `third_down_count`, `third_down_conversion_sum`, `dropback_count` (lines ~234–243).
2. Plays assigned to a **wp-context** bucket: `competitive` (0.05 ≤ wp ≤ 0.95),
   `garbage_leading` (wp > 0.95), `garbage_trailing` (wp < 0.05).
3. Aggregated (summed) by `(team, season, week, season_type, opponent, context)`, pivoted
   wide by context.
4. **Rates** computed from a registered list of `(rate_name, numerator_col, denominator_col)`
   tuples (e.g. `('third_down_conversion_rate', 'third_down_conversion_sum',
   'third_down_count')`, lines ~158–169).
5. Cumulative season averages by `(team, season)` → offense (`off_target`/`off_opp`) and by
   `(opponent_team, season)` → defense (`def_target`/`def_opp`); then league rank + rank_change.

New work plugs into steps 1 and 4; steps 2/3/5 are reused unchanged. (The snap-share family
needs one small addition at step 4 — see below.)

## New raw columns

Add to `REQUIRED_PBP_COLUMNS`: **`ydstogo`**, **`goal_to_go`** (both nflverse-native, all
eras — no pre-2006 charting gap). There is no cache-version constant — cached seasons in
`data/play_by_play/aggregated/` are invalidated by re-running with `refresh=True`
(re-downloads + re-aggregates); the new columns appear only after that regeneration.

## Bucket definitions

Distance bins: **short** = `ydstogo ≤ 2`, **medium** = `3 ≤ ydstogo ≤ 6`, **long** =
`ydstogo ≥ 7`. Goal-to-go uses the `goal_to_go` flag (overrides distance). Buckets:

| Bucket key | Plays in bucket |
|---|---|
| `down1` | `down == 1` and not goal_to_go (≈ all 1st-and-10; distance not split — 1st-and-short/med are rare post-penalty cases) |
| `down2_short` / `down2_med` / `down2_long` | `down == 2`, not goal_to_go, by distance bin |
| `down3_short` / `down3_med` / `down3_long` | `down == 3`, not goal_to_go, by distance bin |
| `goalToGo` | `goal_to_go == 1` (any down) |

8 buckets. Only pass/rush plays are counted (the pipeline already filters to pass/rush plays
with EPA, line ~228), so `play_count = pass_count + run_count` within a bucket.

## Metrics per bucket (24 base rates)

For every bucket B:
- **`{B}_pass_rate`** = `{B}_pass_count / {B}_play_count` — tendency (the play mix).
- **`{B}_success_rate_pass`** = `{B}_pass_success_sum / {B}_pass_count` — execution passing.
- **`{B}_success_rate_run`** = `{B}_run_success_sum / {B}_run_count` — execution running.

Exception — the three **down3** buckets use **conversion** instead of success (canonical
3rd-down metric):
- **`{B}_conversion_rate_pass`** = `{B}_pass_conversion_sum / {B}_pass_count`
- **`{B}_conversion_rate_run`** = `{B}_run_conversion_sum / {B}_run_count`

→ 8 buckets × 3 metrics = **24 base rates**: 8 `pass_rate` (tendency) + 16 efficiency
(10 success-based for down1/down2_*/goalToGo, 6 conversion-based for the three down3_*).

### Per-play component columns (step 1)

For each bucket B, with `in_B` the boolean bucket membership:
- `{B}_play_count = in_B`
- `{B}_pass_count = in_B & pass`
- `{B}_run_count  = in_B & rush`
- success buckets (down1, down2_*, goalToGo): `{B}_pass_success_sum = in_B & pass & success`,
  `{B}_run_success_sum = in_B & rush & success`
- conversion buckets (down3_*): `{B}_pass_conversion_sum = in_B & pass & third_down_converted`,
  `{B}_run_conversion_sum = in_B & rush & third_down_converted` (`third_down_converted` is
  already pulled; fill NaN → 0 as the existing `third_down_conversion_sum` does, line ~243)

These are pure per-play indicator products, summed by the existing aggregation. Register the
24 rates in the rate-definition list (step 4); cumulative/rank/context all follow.

## wp-context snap-share family (12 base)

Captures **how often** a team is in each game state (the frequency signal the situational
*rates* cannot, since an empty bucket is NaN, not a count). Unlike the in-context rates,
snap-share is a **cross-context ratio**, so it needs a small dedicated calc at step 4 (after
the pivot, where each context's `play_count` is its own column):

- `snap_share_competitive`     = `play_count_competitive     / play_count_total`
- `snap_share_garbage_leading` = `play_count_garbage_leading / play_count_total`
- `snap_share_garbage_trailing`= `play_count_garbage_trailing/ play_count_total`

where `play_count_total` is the row sum across the three contexts. These 3 shares × 4
perspectives = **12 base features** → cumulative average + rank + rank_change like the rest.
(The 3 shares sum to 1 → one is linearly redundant; left in for symmetry, RFE prunes.)
Note: snap-share is itself context-level, so it is NOT additionally split by context (it
*describes* the context mix); it does get the 4 perspectives.

## Perspectives & contexts (reused machinery)

Every situational rate gets the standard **3 wp-contexts × 4 perspectives**
(`off_target`, `def_target`, `off_opp`, `def_opp`) — so a matchup row carries both teams'
offensive *and* defensive situational profiles. Garbage-context situational cells will be
thin/often-NaN; this is intentional (kept per the user's request), handled by the existing
NaN treatment (XGBoost native; BART `BART_NAN_SENTINEL = -100`), and pruned by rank-only/RFE.

## Feature-count estimate

- Situational: 24 base × 3 contexts × 4 perspectives = **288** cumulative-average features,
  + rank + rank_change ≈ **~860 columns**.
- Snap-share: 12 base × (cumulative + rank + rank_change) ≈ **~36 columns**.
- Net: candidate pool grows ~2,349 → ~3,250. Consistent with how runs 7/9/11 each grew the
  pool by a family; rank-only mode + RFE do the pruning.

## Testing (unittest, `tests/data/play_by_play/test_collect.py`)

- **Bucket membership:** synthetic plays at `(down, ydstogo, goal_to_go)` land in the right
  bucket; goal-to-go overrides distance; distance bins boundary-correct (2/3 and 6/7 edges).
- **Component products:** `{B}_pass_count`, `{B}_run_count`, success/conversion sums equal
  the hand-counted values on a small synthetic frame; `play_count == pass_count + run_count`.
- **Rate computation:** `pass_rate`, `success_rate_pass/run`, `conversion_rate_pass/run`
  equal num/den; empty bucket → NaN (not 0/0 error).
- **Snap-share:** shares sum to 1 across contexts on a synthetic team-week; correct ratios;
  zero-total row → NaN, no divide error.
- **Era safety:** rows from a pre-2006 season still populate (down/distance exist all eras),
  unlike charting-dependent features.
- **Schema determinism:** new component/rate columns appear in the season-aggregation output
  for every context (follow the existing deterministic-schema test pattern).

## Success criteria

- `ydstogo`/`goal_to_go` pulled; cache version bumped; a re-aggregated season exposes the new
  situational rate columns across all 3 contexts × 4 perspectives plus the 12 snap-share
  columns, with passing unit tests.
- No regression in existing PBP feature columns or tests.
- (Follow-up, separate run) evaluate via the rank-only RFE + grid/BART pipeline and record as
  the next run in the README — not part of this spec.
