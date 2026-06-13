# Play-by-Play Features Phase 3 Implementation Plan (Trenches, Luck, Tendencies)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the final Phase 3 play-by-play features — sack/QB-hit/stuff rates, fumble-recovery luck, CPOE, YAC over expected, penalty rates (committed and drawn), and tendency/pace metrics — riding the Phase 1–2 architecture.

**Architecture:** Three small extensions to `src/data/play_by_play/collect.py`: (1) twelve new in-universe play components feeding nine rate metrics; (2) a new penalty aggregation over penalty rows (penalties live partly on no-play rows outside the pass/rush universe) feeding four metrics; (3) the red-zone drive aggregation generalizes to `_aggregate_drive_components`, emitting pace components (drive seconds / scrimmage snaps) for every drive alongside the existing red-zone counts, feeding one metric. Everything downstream (pivot, cache, cumulative ratios, defense context swap, ranks, collect_all rails) is untouched. Caches must be regenerated (schema grows 168 → 222 component columns).

**Tech Stack:** Python 3.11 (conda env `nfl-predictions`), pandas, nfl_data_py, unittest.

**Spec:** `docs/superpowers/specs/2026-06-12-play-by-play-features-design.md` (Phase 3 section). Baselines: XGBoost run 9 (0.694 / 0.645 / Brier 0.2291), BART run 10 (**0.708** / 0.654 / **0.2185**) on the 2024+2025 pooled hold-out.

---

## Conventions and verified facts

- All commands via `conda run -n nfl-predictions python ...` from the repo root; test module form `python -m unittest tests.data.play_by_play.test_collect -v` (never `discover`).
- **Verified against real downloads (2026-06-12):** `sack`, `qb_hit`, `qb_scramble`, `shotgun`, `no_huddle`, `fumble`, `fumble_lost`, `game_seconds_remaining` are 100% non-null on pass/rush plays in ALL eras (2003/2006/2024 checked). `cpoe` is 0% in 2003–2005, ~80% from 2006+ (NaN-era handling identical to PROE). `yards_after_catch`/`xyac_mean_yardage` are 0% in 2003, ~100%/~91–96% on completions from 2006+. `penalty` is ~97% non-null over all rows (compare with `== 1`, which is False for NaN); `penalty_team` is 100% non-null on penalty rows. `penalty_yards` may be NaN — `fillna(0)`.
- Naming traps: no new fragment contains `off`, `opp`, `home`, `away`, `target`, or starts with `team_`. All names below checked: `sack_rate`, `qb_hit_rate`, `stuff_rate`, `fumble_lost_rate`, `scramble_rate`, `shotgun_rate`, `no_huddle_rate`, `cpoe`, `yac_over_expected`, `pen_committed_*`, `pen_drawn_*`, `seconds_per_play`.
- Feature math: **14 new metrics** (9 play + 4 penalty + 1 pace) × 3 contexts × 2 sides × 3 kinds = **252 new feature columns**; `get_play_by_play_features` grows 648 → **900** (903 with keys). Components grow 56 → **74** (cache 222 component columns). Dataset grows 1,852 → **2,356** columns; features list 1,845 → **2,349**.
- Defense-swap semantics come free: `def_opp_sack_rate` = sacks that defense creates per opposing dropback; `def_opp_pen_drawn_rate` = penalties that defense commits (drawn by offenses against it).
- The feature-count assertion in `TestGetPlayByPlayFeatures.test_feature_count_and_naming` is bumped **in each task that wires metrics** (Task 2 → 810, Task 3 → 882, Task 4 → 900) so the suite is green at every commit.
- Cache invalidation: Phase 2 caches lack the new components and are read back verbatim — Task 5 deletes and regenerates them. Do not run the assembly script or integration tests between Tasks 2–4 and Task 5.

## File structure

| File | Action | Responsibility |
|---|---|---|
| `src/data/play_by_play/collect.py` | Modify | New constants; play components; `_aggregate_penalty_components`; `_aggregate_red_zone_components` → `_aggregate_drive_components` with pace |
| `tests/data/play_by_play/test_collect.py` | Modify | Extended `make_play`; new test classes; renamed/adjusted drive tests; count bumps |
| `data/play_by_play/aggregated/*.parquet`, `data/predict_games/*` | Regenerate | New 222-component schema, 2,356-column dataset |
| `notebooks/.../rfe.ipynb`, `grid_search.ipynb`, `bart.ipynb` | Re-run | Runs 11 (XGBoost) and 12 (BART) |
| `README.md` | Modify | Runs 11–12 |

---

### Task 1: Constants and extended fixture

**Files:**
- Modify: `src/data/play_by_play/collect.py` (constants region)
- Modify: `tests/data/play_by_play/test_collect.py` (`make_play`)

- [ ] **Step 1: Extend `make_play`** — replace the function with this version (15 new params, defaults inert for all existing call sites):

```python
def make_play(posteam='AAA', defteam='BBB', season=2023, week=1, season_type='REG',
              play_id=1, game_id='2023_01_AAA_BBB', is_pass=0, is_rush=0,
              down=1, yardline_100=75.0, third_down_converted=0.0, success=0.0,
              epa=0.0, wp=0.5, xpass=None, fixed_drive=1, fixed_drive_result='Punt',
              yards_gained=0.0, run_location=None, run_gap=None,
              pass_location=None, pass_length=None,
              sack=0.0, qb_hit=0.0, qb_scramble=0.0, shotgun=0.0, no_huddle=0.0,
              fumble=0.0, fumble_lost=0.0, cpoe=None, complete_pass=0.0,
              yards_after_catch=0.0, xyac_mean_yardage=None,
              penalty=0.0, penalty_team=None, penalty_yards=0.0,
              game_seconds_remaining=3600.0):
    """One synthetic nflfastR play row. 'pass'/'rush' are reserved words as kwargs,
    hence is_pass/is_rush."""
    return {
        'posteam': posteam, 'defteam': defteam, 'season': season, 'week': week,
        'season_type': season_type, 'play_id': play_id, 'game_id': game_id,
        'pass': is_pass, 'rush': is_rush, 'down': down, 'yardline_100': yardline_100,
        'third_down_converted': third_down_converted, 'success': success, 'epa': epa,
        # float('nan') rather than None so the xpass column is float64 like real
        # nflfastR data, not object dtype
        'wp': wp, 'xpass': float('nan') if xpass is None else xpass,
        'fixed_drive': fixed_drive, 'fixed_drive_result': fixed_drive_result,
        'yards_gained': yards_gained, 'run_location': run_location,
        'run_gap': run_gap, 'pass_location': pass_location,
        'pass_length': pass_length,
        'sack': sack, 'qb_hit': qb_hit, 'qb_scramble': qb_scramble,
        'shotgun': shotgun, 'no_huddle': no_huddle,
        'fumble': fumble, 'fumble_lost': fumble_lost,
        'cpoe': float('nan') if cpoe is None else cpoe,
        'complete_pass': complete_pass, 'yards_after_catch': yards_after_catch,
        'xyac_mean_yardage': float('nan') if xyac_mean_yardage is None else xyac_mean_yardage,
        'penalty': penalty, 'penalty_team': penalty_team,
        'penalty_yards': penalty_yards,
        'game_seconds_remaining': game_seconds_remaining,
    }
```

- [ ] **Step 2: Write the failing tests** — append:

```python
class TestPhase3Constants(unittest.TestCase):

    def test_phase3_generated_lists(self):
        self.assertEqual(len(collect.PHASE3_PLAY_COMPONENT_COLUMNS), 12)
        self.assertEqual(len(collect.PENALTY_COMPONENT_COLUMNS), 4)
        self.assertEqual(len(collect.PHASE3_RATE_METRICS), 9)
        self.assertEqual(len(collect.PENALTY_RATE_METRICS), 4)
        self.assertEqual(len(collect.PACE_RATE_METRICS), 1)
        self.assertIn(('sack_rate', 'sack_count', 'dropback_count'),
                      collect.PHASE3_RATE_METRICS)
        self.assertIn(('seconds_per_play', 'pace_seconds_sum', 'pace_play_count'),
                      collect.PACE_RATE_METRICS)

    def test_phase3_required_columns(self):
        for column in ('sack', 'qb_hit', 'qb_scramble', 'shotgun', 'no_huddle',
                       'fumble', 'fumble_lost', 'cpoe', 'complete_pass',
                       'yards_after_catch', 'xyac_mean_yardage',
                       'penalty', 'penalty_team', 'penalty_yards',
                       'game_seconds_remaining'):
            self.assertIn(column, collect.REQUIRED_PBP_COLUMNS)
```

- [ ] **Step 3: Run to verify failure**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect.TestPhase3Constants -v`
Expected: FAIL with `AttributeError: ... no attribute 'PHASE3_PLAY_COMPONENT_COLUMNS'`

- [ ] **Step 4: Add the constants.** In `src/data/play_by_play/collect.py`:

(a) Extend `REQUIRED_PBP_COLUMNS` (append the 15 names and extend the comment):

```python
# Verified present for every season 2003-2025. epa/wp/success/fixed_drive are fully
# populated on pass/rush plays back to 2003; xpass is fully null before 2006, which
# makes PROE NaN there by the zero-denominator rule. pass_location/pass_length are
# fully null before 2006 too (directional pass features go NaN there the same way);
# run_location is ~95% populated on rushes in all eras, run_gap ~70% (middle runs
# have no gap by definition). Phase 3: sack/qb_hit/qb_scramble/shotgun/no_huddle/
# fumble/fumble_lost/game_seconds_remaining are 100% populated in all eras; cpoe and
# yards_after_catch/xyac_mean_yardage are null before 2006 (CPOE and YAC-over-expected
# go NaN there); penalty is ~97% non-null (compare with == 1), penalty_team is always
# set on penalty rows, penalty_yards may be NaN.
REQUIRED_PBP_COLUMNS = [
    'posteam', 'defteam', 'season', 'week', 'season_type', 'play_id', 'game_id',
    'pass', 'rush', 'down', 'yardline_100', 'third_down_converted', 'success',
    'epa', 'wp', 'xpass', 'fixed_drive', 'fixed_drive_result',
    'yards_gained', 'run_location', 'run_gap', 'pass_location', 'pass_length',
    'sack', 'qb_hit', 'qb_scramble', 'shotgun', 'no_huddle',
    'fumble', 'fumble_lost', 'cpoe', 'complete_pass',
    'yards_after_catch', 'xyac_mean_yardage',
    'penalty', 'penalty_team', 'penalty_yards', 'game_seconds_remaining',
]
```

(b) Insert directly after the `DIRECTIONAL_RATE_METRICS` block and BEFORE
`PLAY_COMPONENT_COLUMNS` (wired in Tasks 2–4, not here):

```python
# Phase 3: trenches, turnover luck, and tendency components. Sacks/hits/scrambles are
# per dropback, stuffs per carry, fumble-recovery luck per fumble; cpoe and
# YAC-over-expected average only over plays where nflfastR charts them (2006+).
PHASE3_PLAY_COMPONENT_COLUMNS = [
    'sack_count', 'qb_hit_count', 'stuff_count',
    'fumble_sum', 'fumble_lost_sum',
    'scramble_count', 'shotgun_count', 'no_huddle_count',
    'cpoe_sum', 'cpoe_play_count',
    'yac_minus_xyac_sum', 'xyac_play_count',
]

# Penalties live partly on no-play rows outside the pass/rush universe, so they get
# their own aggregation pass. Committed = by the offense (penalty_team == posteam);
# drawn = by the defense against it (penalty_team == defteam). Rates are per
# scrimmage play (play_count denominator).
PENALTY_COMPONENT_COLUMNS = [
    'pen_committed_count', 'pen_committed_yards_sum',
    'pen_drawn_count', 'pen_drawn_yards_sum',
]

PHASE3_RATE_METRICS = [
    ('sack_rate', 'sack_count', 'dropback_count'),
    ('qb_hit_rate', 'qb_hit_count', 'dropback_count'),
    ('scramble_rate', 'scramble_count', 'dropback_count'),
    ('stuff_rate', 'stuff_count', 'rush_count'),
    ('fumble_lost_rate', 'fumble_lost_sum', 'fumble_sum'),
    ('shotgun_rate', 'shotgun_count', 'play_count'),
    ('no_huddle_rate', 'no_huddle_count', 'play_count'),
    ('cpoe', 'cpoe_sum', 'cpoe_play_count'),
    ('yac_over_expected', 'yac_minus_xyac_sum', 'xyac_play_count'),
]
PENALTY_RATE_METRICS = [
    ('pen_committed_rate', 'pen_committed_count', 'play_count'),
    ('pen_committed_yards_per_play', 'pen_committed_yards_sum', 'play_count'),
    ('pen_drawn_rate', 'pen_drawn_count', 'play_count'),
    ('pen_drawn_yards_per_play', 'pen_drawn_yards_sum', 'play_count'),
]
PACE_RATE_METRICS = [
    ('seconds_per_play', 'pace_seconds_sum', 'pace_play_count'),
]
```

- [ ] **Step 5: Run full module** — `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect -v`
Expected: ALL 41 tests PASS (39 existing + 2 new; nothing consumes the constants yet).

- [ ] **Step 6: Commit**

```bash
git add src/data/play_by_play/collect.py tests/data/play_by_play/test_collect.py
git commit -m "Register Phase 3 component and rate-metric constants"
```

---

### Task 2: In-universe play components (trenches, luck, tendencies, CPOE, YAC)

**Files:**
- Modify: `src/data/play_by_play/collect.py` (`PLAY_COMPONENT_COLUMNS`, `RATE_METRICS`, `_aggregate_play_components`)
- Test: `tests/data/play_by_play/test_collect.py`

- [ ] **Step 1: Write the failing tests** — append:

```python
class TestPhase3PlayComponents(unittest.TestCase):

    def _aggregate(self, plays):
        return collect._aggregate_play_components(pd.DataFrame(plays))

    def test_trench_components(self):
        result = self._aggregate([
            # a sack: pass play, negative yards, qb hit
            make_play(play_id=1, is_pass=1, sack=1.0, qb_hit=1.0, yards_gained=-7.0,
                      epa=-1.5),
            # a clean completed pass
            make_play(play_id=2, is_pass=1, complete_pass=1.0, yards_gained=12.0,
                      epa=0.8),
            # a stuffed run (0 yards counts as stuffed)
            make_play(play_id=3, is_rush=1, yards_gained=0.0, epa=-0.4),
            # a healthy run
            make_play(play_id=4, is_rush=1, yards_gained=6.0, epa=0.3),
        ])
        row = result.iloc[0]
        self.assertEqual(row['sack_count'], 1)
        self.assertEqual(row['qb_hit_count'], 1)
        self.assertEqual(row['stuff_count'], 1)
        self.assertEqual(row['dropback_count'], 2)
        self.assertEqual(row['rush_count'], 2)

    def test_fumble_luck_components(self):
        result = self._aggregate([
            make_play(play_id=1, is_rush=1, fumble=1.0, fumble_lost=1.0, epa=-2.0),
            make_play(play_id=2, is_rush=1, fumble=1.0, fumble_lost=0.0, epa=-0.5),
            make_play(play_id=3, is_rush=1, epa=0.1),
        ])
        row = result.iloc[0]
        self.assertEqual(row['fumble_sum'], 2)
        self.assertEqual(row['fumble_lost_sum'], 1)

    def test_tendency_components(self):
        result = self._aggregate([
            make_play(play_id=1, is_pass=1, shotgun=1.0, no_huddle=1.0,
                      qb_scramble=1.0, epa=0.2),
            make_play(play_id=2, is_rush=1, shotgun=1.0, epa=0.1),
        ])
        row = result.iloc[0]
        self.assertEqual(row['scramble_count'], 1)
        self.assertEqual(row['shotgun_count'], 2)
        self.assertEqual(row['no_huddle_count'], 1)

    def test_cpoe_and_yac_skip_uncharted_plays(self):
        result = self._aggregate([
            # charted completion: cpoe 5.0, yac 8 vs expected 5.5 -> +2.5
            make_play(play_id=1, is_pass=1, cpoe=5.0, complete_pass=1.0,
                      yards_after_catch=8.0, xyac_mean_yardage=5.5, epa=0.6),
            # pre-2006-style pass: no cpoe, no xyac -> contributes to neither metric
            make_play(play_id=2, is_pass=1, complete_pass=1.0,
                      yards_after_catch=4.0, epa=0.2),
        ])
        row = result.iloc[0]
        self.assertEqual(row['cpoe_play_count'], 1)
        self.assertAlmostEqual(row['cpoe_sum'], 5.0)
        self.assertEqual(row['xyac_play_count'], 1)
        self.assertAlmostEqual(row['yac_minus_xyac_sum'], 2.5)

    def test_stuff_requires_known_yardage(self):
        # NaN yards_gained must not count as a stuff
        result = self._aggregate([
            make_play(play_id=1, is_rush=1, yards_gained=float('nan'), epa=0.0),
        ])
        self.assertEqual(result.iloc[0]['stuff_count'], 0)
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect.TestPhase3PlayComponents -v`
Expected: FAIL with `KeyError` (components not yet in the groupby selection) or missing columns.

- [ ] **Step 3: Wire and implement.**

(a) `PLAY_COMPONENT_COLUMNS` gains the Phase 3 tail:

```python
PLAY_COMPONENT_COLUMNS = [
    'play_count', 'epa_sum', 'success_sum',
    'dropback_count', 'dropback_epa_sum', 'dropback_success_sum',
    'rush_count', 'rush_epa_sum', 'rush_success_sum',
    'early_down_count', 'early_down_success_sum',
    'third_down_count', 'third_down_conversion_sum',
    'xpass_play_count', 'pass_minus_xpass_sum',
] + DIRECTIONAL_COMPONENT_COLUMNS + PHASE3_PLAY_COMPONENT_COLUMNS
```

(b) `RATE_METRICS` gains `+ PHASE3_RATE_METRICS` after `+ DIRECTIONAL_RATE_METRICS`:

```python
] + DIRECTIONAL_RATE_METRICS + PHASE3_RATE_METRICS
```

(c) In `_aggregate_play_components`, after the directional bucket loop and before the
final `return`, add:

```python
    plays['sack_count'] = plays['sack']
    plays['qb_hit_count'] = plays['qb_hit']
    # NaN yards_gained must not count as a stuff: the raw column comparison is False
    # for NaN, unlike the zero-filled `yards` used for explosives above.
    plays['stuff_count'] = ((plays['rush'] == 1) & (plays['yards_gained'] <= 0)).astype(int)
    plays['fumble_sum'] = plays['fumble']
    plays['fumble_lost_sum'] = plays['fumble_lost']
    plays['scramble_count'] = plays['qb_scramble']
    plays['shotgun_count'] = plays['shotgun']
    plays['no_huddle_count'] = plays['no_huddle']
    has_cpoe = plays['cpoe'].notna()
    plays['cpoe_play_count'] = has_cpoe.astype(int)
    plays['cpoe_sum'] = plays['cpoe'].where(has_cpoe, 0)
    has_xyac = (
        (plays['complete_pass'] == 1)
        & plays['xyac_mean_yardage'].notna()
        & plays['yards_after_catch'].notna()
    )
    plays['xyac_play_count'] = has_xyac.astype(int)
    plays['yac_minus_xyac_sum'] = (
        plays['yards_after_catch'] - plays['xyac_mean_yardage']
    ).where(has_xyac, 0)
```

(d) Bump the feature-count assertion in `TestGetPlayByPlayFeatures.test_feature_count_and_naming`:

```python
        # 45 metrics (10 aggregate + 26 directional + 9 phase-3 play) x 3 contexts
        # x 2 sides x 3 column kinds (rate, rank, rank_change)
        self.assertEqual(len(feature_cols), 810)
```

- [ ] **Step 4: Run full module** — expected ALL 46 tests PASS (41 + 5 new).

- [ ] **Step 5: Commit**

```bash
git add src/data/play_by_play/collect.py tests/data/play_by_play/test_collect.py
git commit -m "Add trench, luck, tendency, CPOE and YAC play components"
```

---

### Task 3: Penalty aggregation

**Files:**
- Modify: `src/data/play_by_play/collect.py` (`COMPONENT_COLUMNS` wiring, `RATE_METRICS`, new `_aggregate_penalty_components`, `_aggregate_season` merge)
- Test: `tests/data/play_by_play/test_collect.py`

- [ ] **Step 1: Write the failing tests** — append:

```python
class TestPenaltyComponents(unittest.TestCase):

    def test_committed_and_drawn_split_by_penalty_team(self):
        plays = pd.DataFrame([
            # offense (AAA) commits a 10-yard penalty on a scrimmage play
            make_play(play_id=1, is_pass=1, epa=-0.5, penalty=1.0,
                      penalty_team='AAA', penalty_yards=10.0),
            # defense (BBB) commits a 5-yard penalty on a no-play row
            # (pass == rush == 0: outside the scrimmage universe, must still count)
            make_play(play_id=2, penalty=1.0, penalty_team='BBB', penalty_yards=5.0),
            # clean play
            make_play(play_id=3, is_rush=1, epa=0.1),
        ])
        result = collect._aggregate_penalty_components(plays)
        row = result.iloc[0]
        self.assertEqual(row['pen_committed_count'], 1)
        self.assertEqual(row['pen_committed_yards_sum'], 10.0)
        self.assertEqual(row['pen_drawn_count'], 1)
        self.assertEqual(row['pen_drawn_yards_sum'], 5.0)

    def test_nan_penalty_rows_are_not_penalties(self):
        plays = pd.DataFrame([
            make_play(play_id=1, is_pass=1, epa=0.1, penalty=float('nan')),
        ])
        result = collect._aggregate_penalty_components(plays)
        self.assertTrue(result.empty)

    def test_penalty_team_is_normalized_for_relocated_franchises(self):
        # Consistency guard: real pbp already uses modern abbreviations everywhere, but
        # if any input ever carried era codes, posteam/defteam are normalized and
        # penalty_team must be too, or the committed/drawn comparison desynchronizes.
        plays = pd.DataFrame([
            make_play(posteam='SD', defteam='OAK', play_id=1, is_pass=1, epa=0.1),
            make_play(posteam='SD', defteam='OAK', play_id=2, penalty=1.0,
                      penalty_team='SD', penalty_yards=10.0),
        ])
        season = collect._aggregate_season(plays)
        row = season[season['team'] == 'LAC'].iloc[0]
        self.assertEqual(row['pen_committed_count_competitive'], 1)
        self.assertEqual(row['pen_committed_yards_sum_competitive'], 10.0)

    def test_penalty_components_reach_the_season_frame(self):
        plays = pd.DataFrame([
            make_play(play_id=1, is_pass=1, epa=0.2),
            make_play(play_id=2, penalty=1.0, penalty_team='AAA', penalty_yards=15.0),
        ])
        season = collect._aggregate_season(plays)
        row = season[season['team'] == 'AAA'].iloc[0]
        self.assertEqual(row['pen_committed_count_competitive'], 1)
        self.assertEqual(row['pen_committed_yards_sum_competitive'], 15.0)
        self.assertEqual(row['play_count_competitive'], 1)
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect.TestPenaltyComponents -v`
Expected: FAIL with `AttributeError: ... no attribute '_aggregate_penalty_components'`

- [ ] **Step 3: Implement.**

(a) Add after `_aggregate_red_zone_components` (current name; renamed in Task 4):

```python
def _aggregate_penalty_components(pbp_df):
    """
    Count penalties per team-week-context over ALL rows with teams attached: accepted
    penalties frequently live on no-play rows outside the pass/rush universe, so this is
    a separate aggregation pass (like drives). Committed = flagged on the offense
    (penalty_team == posteam); drawn = flagged on the defense (penalty_team == defteam).
    The matching rates use scrimmage play_count as the denominator, so they read as
    "penalties per offensive snap". NaN penalty values compare False and are ignored.
    """
    penalties = pbp_df[
        (pbp_df['penalty'] == 1)
        & pbp_df['posteam'].notna()
        & pbp_df['defteam'].notna()
    ].copy()
    penalties[CONTEXT_COL] = _assign_wp_context(penalties['wp'])

    committed = penalties['penalty_team'] == penalties['posteam']
    drawn = penalties['penalty_team'] == penalties['defteam']
    yards = penalties['penalty_yards'].fillna(0)
    penalties['pen_committed_count'] = committed.astype(int)
    penalties['pen_committed_yards_sum'] = yards * committed
    penalties['pen_drawn_count'] = drawn.astype(int)
    penalties['pen_drawn_yards_sum'] = yards * drawn

    return penalties.groupby(
        AGGREGATION_KEY_COLUMNS + [CONTEXT_COL]
    )[PENALTY_COMPONENT_COLUMNS].sum().reset_index()
```

(b) Wire the components into the aggregate list — `COMPONENT_COLUMNS` becomes:

```python
COMPONENT_COLUMNS = PLAY_COMPONENT_COLUMNS + DRIVE_COMPONENT_COLUMNS + PENALTY_COMPONENT_COLUMNS
```

(c) `RATE_METRICS` gains `+ PENALTY_RATE_METRICS`:

```python
] + DIRECTIONAL_RATE_METRICS + PHASE3_RATE_METRICS + PENALTY_RATE_METRICS
```

(d) In `_aggregate_season`, FIRST extend the team normalization to `penalty_team`.
(Verified 2026-06-12: nflfastR pbp already uses MODERN abbreviations for all seasons —
2005 has LAC/LV/LA, never SD/OAK/STL — so all three replaces are no-ops on real data;
the era mapping exists for schedule data. This line is a consistency guard so the
committed/drawn comparison can never desynchronize if any input ever carries era
codes: the three team columns must always receive identical treatment.)

```python
    pbp_df['posteam'] = pbp_df['posteam'].replace(TEAM_ABBR_MAPPINGS)
    pbp_df['defteam'] = pbp_df['defteam'].replace(TEAM_ABBR_MAPPINGS)
    pbp_df['penalty_team'] = pbp_df['penalty_team'].replace(TEAM_ABBR_MAPPINGS)
```

Then merge the third component frame (replace the current two-frame merge):

```python
    play_components = _aggregate_play_components(pbp_df)
    drive_components = _aggregate_red_zone_components(pbp_df)
    penalty_components = _aggregate_penalty_components(pbp_df)
    components = play_components.merge(
        drive_components, on=AGGREGATION_KEY_COLUMNS + [CONTEXT_COL], how='outer'
    ).merge(
        penalty_components, on=AGGREGATION_KEY_COLUMNS + [CONTEXT_COL], how='outer'
    )
    # Fill only the component columns: a side missing from the outer merge means zero
    # plays/drives/penalties, and restricting the fill keeps pandas from
    # object-downcasting keys.
    components[COMPONENT_COLUMNS] = components[COMPONENT_COLUMNS].fillna(0)
```

(NOTE: `_aggregate_season` references `_aggregate_red_zone_components` here; Task 4
renames it to `_aggregate_drive_components` and updates this call site.)

(e) Bump the feature-count assertion to **882** with comment
`# 49 metrics (10 aggregate + 26 directional + 9 phase-3 play + 4 penalty) x 3 x 2 x 3`.

- [ ] **Step 4: Run full module** — expected ALL 49 tests PASS (46 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/data/play_by_play/collect.py tests/data/play_by_play/test_collect.py
git commit -m "Aggregate committed and drawn penalties per wp context"
```

---

### Task 4: Pace — generalize the drive aggregation

**Files:**
- Modify: `src/data/play_by_play/collect.py` (`_aggregate_red_zone_components` → `_aggregate_drive_components`, `DRIVE_COMPONENT_COLUMNS`, `RATE_METRICS`, `_aggregate_season` call site)
- Test: `tests/data/play_by_play/test_collect.py` (rename class refs, adjust two assertions, add pace tests)

- [ ] **Step 1: Write the failing tests** — append:

```python
class TestPaceComponents(unittest.TestCase):

    def test_seconds_and_snaps_accumulate_per_drive(self):
        plays = pd.DataFrame([
            # drive 1: three snaps, first at 3600s, last at 3500s -> 100 elapsed, 3 snaps
            make_play(play_id=1, fixed_drive=1, is_pass=1, epa=0.1,
                      game_seconds_remaining=3600.0),
            make_play(play_id=2, fixed_drive=1, is_rush=1, epa=0.1,
                      game_seconds_remaining=3560.0),
            make_play(play_id=3, fixed_drive=1, is_pass=1, epa=0.1,
                      game_seconds_remaining=3500.0),
            # drive 2: one snap -> 0 elapsed, 1 snap
            make_play(play_id=4, fixed_drive=2, is_rush=1, epa=0.1,
                      game_seconds_remaining=3300.0),
        ])
        result = collect._aggregate_drive_components(plays)
        row = result.iloc[0]
        self.assertEqual(row['pace_seconds_sum'], 100.0)
        self.assertEqual(row['pace_play_count'], 4)

    def test_every_drive_contributes_pace_but_only_deep_drives_count_red_zone(self):
        plays = pd.DataFrame([
            make_play(play_id=1, fixed_drive=1, is_pass=1, yardline_100=60.0, epa=0.1,
                      game_seconds_remaining=3600.0),
            make_play(play_id=2, fixed_drive=1, is_rush=1, yardline_100=15.0, epa=0.1,
                      game_seconds_remaining=3550.0, fixed_drive_result='Touchdown'),
        ])
        # make both rows agree on the drive result, as real data does
        plays.loc[0, 'fixed_drive_result'] = 'Touchdown'
        result = collect._aggregate_drive_components(plays)
        row = result.iloc[0]
        self.assertEqual(row['red_zone_drive_count'], 1)
        self.assertEqual(row['red_zone_td_drive_count'], 1)
        self.assertEqual(row['pace_seconds_sum'], 50.0)
        self.assertEqual(row['pace_play_count'], 2)
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect.TestPaceComponents -v`
Expected: FAIL with `AttributeError: ... no attribute '_aggregate_drive_components'`

- [ ] **Step 3: Implement.**

(a) Replace `_aggregate_red_zone_components` entirely with:

```python
def _aggregate_drive_components(pbp_df):
    """
    Drive-level components per team-week-context, over scrimmage plays only (pass or
    rush): PAT and kickoff rows share the drive's fixed_drive number at misleading
    yardlines (a PAT snapped at the 15 would otherwise turn every long touchdown into a
    fake red-zone trip). A drive's context comes from the win probability on its first
    scrimmage play.

    Red zone: a drive counts as a trip when any of its scrimmage plays starts at or
    inside the opponent's 20; drives that enter only via a kick or kneel are
    intentionally excluded on both sides of the red_zone_td_rate ratio.

    Pace: game-clock seconds elapsed between the drive's first and last scrimmage snap,
    over its scrimmage snap count. This undercounts by the final play's duration
    (n snaps bound n-1 intervals) — a consistent bias that cancels in cross-team
    comparison. clip(lower=0) guards overtime clock quirks.

    fixed_drive numbers drives across the whole game, so (game_id, fixed_drive) is
    unique.
    """
    scrimmage = pbp_df[(pbp_df['pass'] == 1) | (pbp_df['rush'] == 1)]
    drive_plays = scrimmage[scrimmage['fixed_drive'].notna() & scrimmage['posteam'].notna()].sort_values('play_id')
    drives = drive_plays.groupby(['game_id', 'fixed_drive'] + AGGREGATION_KEY_COLUMNS).agg(
        min_yardline_100=('yardline_100', 'min'),
        first_play_wp=('wp', 'first'),
        drive_result=('fixed_drive_result', 'first'),
        first_gsr=('game_seconds_remaining', 'first'),
        last_gsr=('game_seconds_remaining', 'last'),
        scrimmage_snaps=('play_id', 'count'),
    ).reset_index()

    drives[CONTEXT_COL] = _assign_wp_context(drives['first_play_wp'])
    reached_red_zone = drives['min_yardline_100'] <= RED_ZONE_YARDLINE
    drives['red_zone_drive_count'] = reached_red_zone.astype(int)
    drives['red_zone_td_drive_count'] = (
        reached_red_zone & (drives['drive_result'] == 'Touchdown')
    ).astype(int)
    drives['pace_seconds_sum'] = (drives['first_gsr'] - drives['last_gsr']).clip(lower=0)
    drives['pace_play_count'] = drives['scrimmage_snaps']

    return drives.groupby(
        AGGREGATION_KEY_COLUMNS + [CONTEXT_COL]
    )[DRIVE_COMPONENT_COLUMNS].sum().reset_index()
```

(b) `DRIVE_COMPONENT_COLUMNS` becomes:

```python
DRIVE_COMPONENT_COLUMNS = [
    'red_zone_drive_count', 'red_zone_td_drive_count',
    'pace_seconds_sum', 'pace_play_count',
]
```

(c) `RATE_METRICS` gains `+ PACE_RATE_METRICS` (final form:
`] + DIRECTIONAL_RATE_METRICS + PHASE3_RATE_METRICS + PENALTY_RATE_METRICS + PACE_RATE_METRICS`).

(d) Update the call site in `_aggregate_season`:
`drive_components = _aggregate_drive_components(pbp_df)`.

(e) Update the existing test class `TestAggregateRedZoneComponents`: every call to
`collect._aggregate_red_zone_components(...)` becomes
`collect._aggregate_drive_components(...)`, and ONE assertion changes — in
`test_non_scrimmage_rows_do_not_create_red_zone_trips`, the long-TD drive now
legitimately appears as a pace row, so replace

```python
        result = collect._aggregate_red_zone_components(plays)
        self.assertTrue(result.empty)
```

with

```python
        result = collect._aggregate_drive_components(plays)
        # the drive appears for pace purposes, but must contribute no red-zone trip
        self.assertEqual(result['red_zone_drive_count'].sum(), 0)
        self.assertEqual(result['pace_play_count'].sum(), 1)  # the scrimmage snap only
```

All other assertions in that class hold unchanged (red-zone counts are unaffected by
non-red-zone drives joining the frame, because grouping sums within team-week-context).

(f) Bump the feature-count assertion to **900** with comment
`# 50 metrics (10 aggregate + 26 directional + 9 phase-3 play + 4 penalty + 1 pace) x 3 x 2 x 3`.

- [ ] **Step 4: Run full module** — expected ALL 51 tests PASS (49 + 2 new).
Also grep for stragglers: `grep -rn "_aggregate_red_zone_components" src tests` → no matches.

- [ ] **Step 5: Commit**

```bash
git add src/data/play_by_play/collect.py tests/data/play_by_play/test_collect.py
git commit -m "Generalize drive aggregation: red zone plus pace components"
```

---

### Task 5: Regenerate caches and dataset

- [ ] **Step 1: Real-data smoke test first (1 season, ~1 min):**

```bash
conda run -n nfl-predictions python -c "
from src.data.play_by_play import collect
feats = collect.get_play_by_play_features([2023], refresh=True)
print('total columns:', len(feats.columns))  # expect 903
kc = feats[feats['team'] == 'KC'].sort_values('week')
for col in ('off_sack_rate_competitive_cumulative_average',
            'off_seconds_per_play_competitive_cumulative_average',
            'off_pen_committed_rate_competitive_cumulative_average'):
    print(col, '->', round(kc[col].iloc[6], 3))
"
```

Expected: 903 columns; KC sack rate ~0.03–0.10, seconds per play ~25–40, penalty rate
~0.05–0.15. If any value is wildly off, STOP and investigate before burning 25 minutes
on the full regen.

- [ ] **Step 2: Full regen (LONG ~25 min):**

```bash
rm data/play_by_play/aggregated/*.parquet
conda run -n nfl-predictions python -m scripts.data_assembly.predict_game_winner.schedule_and_weekly
```

- [ ] **Step 3: Verify** — total columns 2,356; 504 new feature columns matching
`sack_rate|qb_hit_rate|stuff_rate|fumble_lost_rate|scramble_rate|shotgun_rate|no_huddle_rate|cpoe|yac_over_expected|pen_|seconds_per_play`;
cpoe/yac columns 0% non-null pre-2006 and high 2006+; sack/pace columns ~1.0 in all
eras (weeks ≥ 3); features list 2,349.

- [ ] **Step 4: Integration suite:** `conda run -n nfl-predictions python -m unittest tests.data.test_collect_all -v` — 8 PASS.

- [ ] **Step 5: Commit** (caches + parquet + features list), message
"Regenerate caches and dataset with Phase 3 features".

---

### Task 6: Run 11 — RFE + grid search (XGBoost)

Same procedure as run 9 (see the Phase 2 plan, Task 5, for the exact extraction
commands): execute `rfe.ipynb` (PYTHONPATH=repo root, nbconvert --execute --inplace, NO
pipe through tail — check the exit code), read `get_best_num_features(.005)` → N11, set
`BEST_NUM_FEATS = <N11>` in `grid_search.ipynb` cell 1 (JSON edit), execute it, compute
pooled hold-out AUROC/accuracy/Brier with the standard snippet (seed-32 shuffle,
season >= 2024, `models/best_random_xgb_model.json`), then record run 11 in README
(table row + changelog bullet: pool size, selected count, Phase 3 feature count among
selected, hold-out vs runs 5/7/9, Brier). Commit
"Record Run 11: XGBoost with Phase 3 play-by-play features".

### Task 7: Run 12 — BART

Set `BEST_NUM_FEATS = <N11>` in `bart.ipynb` cell 1; execute with the verification
protocol (exit code via `&& echo NBCONVERT_OK`, then confirm the shape echo
`train (9724, <N11>)` and that headline metrics CHANGED from run 10's recorded values
before trusting them — PGBART wobbles ~±0.004 run-to-run, identical-to-4-decimals
means a stale/failed execution). The NaN sentinel handles Phase 3's NaN features
(cpoe/yac pre-2006) automatically. Extract headline + Brier + width-stratified Brier
cells; record run 12 in README with the sampler-wobble caveat. Commit
"Record Run 12: BART on Phase 3 feature set".

### Task 8: PR

Push `add-phase3-pbp-features`, `gh pr create --base master` titled
"Phase 3 play-by-play: trenches, luck, tendencies (runs 11-12)" with summary (14
metrics / 252 features / era caveats), results vs baselines, and test plan; body ends
with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
