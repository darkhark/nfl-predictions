# Play-by-Play Features Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Phase 1 play-by-play features (EPA/success, situational, PROE) split by win-probability context, plugged into the existing cumulative/rank/target-opp pipeline.

**Architecture:** A shared `src/data/transformations.py` module holds the rank machinery and team-abbreviation map (extracted from weekly/schedule). `src/data/play_by_play/collect.py` aggregates plays to team-week **component sums** (cached per season as parquet), computes cumulative rates as `cumsum(numerator)/cumsum(denominator)`, swaps garbage-time context labels for the defense's perspective, then ranks via the shared helper. `collect_all.get_schedule_and_weekly_data` gains an `include_play_by_play` flag that left-joins the feature frame onto the weekly frame before any home/away logic.

**Tech Stack:** Python 3.11 (conda env `nfl-predictions`), pandas, nfl_data_py, unittest (no pytest in env).

**Spec:** `docs/superpowers/specs/2026-06-12-play-by-play-features-design.md`

---

## Conventions for every task

- Run all Python through the conda env: `conda run -n nfl-predictions python ...`
- Run a single test module from the repo root (namespace imports work; do NOT use `unittest discover`, it requires `__init__.py` files this repo doesn't have):
  `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect -v`
- Verified facts (checked against real downloads, 2026-06-12): `nfl.import_pbp_data(years=[y], columns=[...], downcast=False)` returns exactly the requested columns; all columns below exist for 2003–2025; `epa`/`wp`/`success`/`fixed_drive`/`fixed_drive_result` are 100% non-null on pass/rush plays back to 2003; `xpass` is 100% null in 2003–2005 (PROE will be NaN there — expected, per spec).
- Design decisions locked in the spec: cumulative rates only (no raw per-week rate columns, no `_cumulative_average_change` columns — `_rank_change` carries momentum, consistent with the rank-only Run 5 direction); zero-denominator cells are NaN, never 0; defense context labels are swapped (offense `garbage_leading` = defense `garbage_trailing`).
- Metric/context name constraint: new column fragments must NOT contain the substrings `opp`, `off`, `home`, `away`, `target`, or start with `team_` — collect_all renames columns by substring. All names below were checked against this.

## File structure

| File | Action | Responsibility |
|---|---|---|
| `src/data/transformations.py` | Create | Shared rank/rank-change helpers, `TEAM_ABBR_MAPPINGS`, shared key-column constants |
| `src/data/weekly/collect.py` | Modify | Delegate `add_rank_columns` to transformations |
| `src/data/schedule/collect.py` | Modify | Import `TEAM_ABBR_MAPPINGS` from transformations |
| `src/data/play_by_play/collect.py` | Rewrite | PBP download + per-season cache, context split, component aggregation, cumulative rates, ranks |
| `src/data/collect_all.py` | Modify | `include_play_by_play` flag + merge |
| `scripts/data_assembly/predict_game_winner/schedule_and_weekly.py` | Modify | Pass `include_play_by_play=True` |
| `tests/data/test_transformations.py` | Create | Synthetic tests for shared rank helpers |
| `tests/data/play_by_play/test_collect.py` | Create | Synthetic tests for all PBP logic |
| `tests/data/test_collect_all.py` | Modify | Integration tests for the four column families + shift |

---

### Task 1: Shared transformations module

**Files:**
- Create: `src/data/transformations.py`
- Test: `tests/data/test_transformations.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_transformations.py`:

```python
import unittest

import pandas as pd

from src.data import transformations


def make_frame():
    """Two teams, two weeks. Offense column where higher is better, defense column
    where lower is better. AAA is better on both in week 1; they swap in week 2."""
    return pd.DataFrame([
        {'team': 'AAA', 'opp_team': 'BBB', 'season': 2023, 'week': 1,
         'team_game_count': 1, 'opp_game_count': 1,
         'off_metric_cumulative_average': 30.0, 'def_opp_metric_cumulative_average': 10.0},
        {'team': 'BBB', 'opp_team': 'AAA', 'season': 2023, 'week': 1,
         'team_game_count': 1, 'opp_game_count': 1,
         'off_metric_cumulative_average': 20.0, 'def_opp_metric_cumulative_average': 15.0},
        {'team': 'AAA', 'opp_team': 'BBB', 'season': 2023, 'week': 2,
         'team_game_count': 2, 'opp_game_count': 2,
         'off_metric_cumulative_average': 18.0, 'def_opp_metric_cumulative_average': 16.0},
        {'team': 'BBB', 'opp_team': 'AAA', 'season': 2023, 'week': 2,
         'team_game_count': 2, 'opp_game_count': 2,
         'off_metric_cumulative_average': 25.0, 'def_opp_metric_cumulative_average': 11.0},
    ])


class TestAddRankAndRankChangeColumns(unittest.TestCase):

    def setUp(self):
        self.result = transformations.add_rank_and_rank_change_columns(
            make_frame(),
            off_cols=['off_metric_cumulative_average'],
            def_cols=['def_opp_metric_cumulative_average'],
        )

    def _value(self, team, week, col):
        row = self.result[(self.result['team'] == team) & (self.result['week'] == week)]
        return row[col].values[0]

    def test_offense_rank_one_is_highest_value(self):
        self.assertEqual(self._value('AAA', 1, 'off_metric_cumulative_average_rank'), 1)
        self.assertEqual(self._value('BBB', 1, 'off_metric_cumulative_average_rank'), 2)

    def test_defense_rank_one_is_lowest_value(self):
        self.assertEqual(self._value('AAA', 1, 'def_opp_metric_cumulative_average_rank'), 1)
        self.assertEqual(self._value('BBB', 1, 'def_opp_metric_cumulative_average_rank'), 2)

    def test_first_game_rank_change_is_zero(self):
        self.assertEqual(self._value('AAA', 1, 'off_metric_cumulative_average_rank_change'), 0)
        self.assertEqual(self._value('AAA', 1, 'def_opp_metric_cumulative_average_rank_change'), 0)

    def test_offense_rank_change_is_week_over_week_diff(self):
        # AAA falls from rank 1 to rank 2 -> change +1; BBB rises -> change -1
        self.assertEqual(self._value('AAA', 2, 'off_metric_cumulative_average_rank_change'), 1)
        self.assertEqual(self._value('BBB', 2, 'off_metric_cumulative_average_rank_change'), -1)

    def test_defense_rank_change_is_grouped_by_opponent(self):
        # def_opp columns describe the OPPONENT's defense, so changes group by opp_team.
        # On AAA's rows the opponent is BBB: BBB's defense allowed 10.0 then 16.0, so its
        # rank goes 1 -> 2 and the change is +1.
        self.assertEqual(self._value('AAA', 2, 'def_opp_metric_cumulative_average_rank_change'), 1)

    def test_nan_values_get_nan_rank(self):
        frame = make_frame()
        frame.loc[0, 'off_metric_cumulative_average'] = float('nan')
        result = transformations.add_rank_and_rank_change_columns(
            frame, off_cols=['off_metric_cumulative_average'],
            def_cols=['def_opp_metric_cumulative_average'],
        )
        week1 = result[result['week'] == 1]
        self.assertTrue(pd.isna(
            week1[week1['team'] == 'AAA']['off_metric_cumulative_average_rank'].values[0]
        ))
        # The remaining team still ranks 1 among teams with values
        self.assertEqual(
            week1[week1['team'] == 'BBB']['off_metric_cumulative_average_rank'].values[0], 1
        )


class TestTeamAbbrMappings(unittest.TestCase):

    def test_relocated_franchises_map_to_current_abbreviations(self):
        self.assertEqual(transformations.TEAM_ABBR_MAPPINGS['STL'], 'LA')
        self.assertEqual(transformations.TEAM_ABBR_MAPPINGS['SD'], 'LAC')
        self.assertEqual(transformations.TEAM_ABBR_MAPPINGS['OAK'], 'LV')


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n nfl-predictions python -m unittest tests.data.test_transformations -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.transformations'`

- [ ] **Step 3: Write the implementation**

Create `src/data/transformations.py`. The rank logic is moved verbatim from `src/data/weekly/collect.py:128-149` (parameterized on the column lists); the mapping is moved from `src/data/schedule/collect.py:20-24`:

```python
import pandas as pd

# If a team moved cities, the name in current data releases is the most recent name.
# Older sources (schedules, play-by-play) use the era abbreviation; map to the current
# one so frames merge cleanly on team.
TEAM_ABBR_MAPPINGS = {
    'STL': 'LA',
    'SD': 'LAC',
    'OAK': 'LV'
}

TEAM_COL = 'team'
OPPONENT_TEAM_COL = 'opp_team'
SEASON_COL = 'season'
WEEK_COL = 'week'
TEAM_GAME_COUNT_COL = 'team_game_count'
OPP_GAME_COUNT_COL = 'opp_game_count'


def add_rank_and_rank_change_columns(df, off_cols, def_cols):
    """
    Add cross-sectional rank and week-over-week rank-change columns for the given offense
    and defense columns, computed within each (season, week).

    Offense columns are ranked descending, so rank 1 is the highest value (best offense).
    Defense columns are ranked ascending, so rank 1 is the fewest allowed (best defense).
    Ties share the better rank (competition ranking), matching how league standings are
    reported. NaN values receive NaN ranks; remaining teams are ranked among themselves.

    Rank-change is the per-team change in that rank from the previous game: offense ranks
    are diffed within (team, season) and defense ranks within (opp_team, season), because
    def_opp_* columns describe the opponent's defense. The first game of each group has a
    rank-change of 0.

    :param df: a team-week frame with team/opp_team/season/week/game-count columns
    :param off_cols: offense columns to rank (higher is better)
    :param def_cols: defense columns to rank (lower is better)
    :return: the frame with *_rank and *_rank_change columns appended
    """
    off_ranks = df.groupby([SEASON_COL, WEEK_COL])[off_cols].rank(ascending=False, method='min').add_suffix('_rank')
    def_ranks = df.groupby([SEASON_COL, WEEK_COL])[def_cols].rank(ascending=True, method='min').add_suffix('_rank')
    df = pd.concat([df, off_ranks, def_ranks], axis=1)

    off_rank_changes = rank_change_frame(df, list(off_ranks.columns), [TEAM_COL, SEASON_COL], TEAM_GAME_COUNT_COL)
    def_rank_changes = rank_change_frame(df, list(def_ranks.columns), [OPPONENT_TEAM_COL, SEASON_COL], OPP_GAME_COUNT_COL)
    df = pd.concat([df, off_rank_changes, def_rank_changes], axis=1)

    df.sort_values(by=[TEAM_COL, SEASON_COL, TEAM_GAME_COUNT_COL], inplace=True)
    return df


def rank_change_frame(df, rank_cols, groupby_columns, game_count_col):
    """
    Build a *_rank_change frame for each rank column, computed as the change from the
    entity's previous game. The entity is whatever groupby_columns identifies (the team for
    offense ranks, the opponent for defense ranks). The first game of each group is filled
    with 0. The returned frame is indexed like df so it can be concatenated back on.
    """
    ordered = df.sort_values(by=groupby_columns + [game_count_col])
    changes = ordered.groupby(groupby_columns)[rank_cols].diff().fillna(0)
    return changes.add_suffix('_change')
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n nfl-predictions python -m unittest tests.data.test_transformations -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/transformations.py tests/data/test_transformations.py
git commit -m "Add shared rank/rank-change helpers and team abbreviation map"
```

---

### Task 2: Delegate weekly and schedule modules to the shared helpers

**Files:**
- Modify: `src/data/weekly/collect.py:108-149`
- Modify: `src/data/schedule/collect.py:17-24`

- [ ] **Step 1: Update weekly/collect.py**

Add to the imports at the top (line 1 area):

```python
import pandas as pd

from src.data import transformations
```

Replace the whole bodies of `add_rank_columns` and delete `_rank_change_frame` (lines 108–149). The public name `add_rank_columns` is kept because `get_weekly_data` calls it:

```python
def add_rank_columns(df):
    """
    Add cross-sectional rank and week-over-week rank-change columns for every cumulative
    average stat. See transformations.add_rank_and_rank_change_columns for the ranking
    semantics (offense descending, defense ascending, competition ranking, first game 0).

    :param df: The concatenated weekly dataframe, after the per-year cumulative columns are built
    :return: The dataframe with *_rank and *_rank_change columns appended
    """
    off_cols = [col for col in df.columns if col.startswith('off_') and col.endswith('_cumulative_average')]
    def_cols = [col for col in df.columns if col.startswith('def_opp_') and col.endswith('_cumulative_average')]
    return transformations.add_rank_and_rank_change_columns(df, off_cols, def_cols)
```

- [ ] **Step 2: Update schedule/collect.py**

Replace the local mapping definition (lines 17–24) with an import. Keep the module-level name so any external reference to `schedule_collect.TEAM_ABBR_MAPPINGS` still resolves:

```python
from src.data.transformations import TEAM_ABBR_MAPPINGS
```

(Delete the `TEAM_ABBR_MAPPINGS = {...}` dict literal; keep the explanatory comment above it, moving it to `transformations.py` if it is not already covered by the comment written in Task 1 — it is, so just delete.)

- [ ] **Step 3: Run the existing weekly and schedule tests (network-backed, ~1-2 min)**

Run: `conda run -n nfl-predictions python -m unittest tests.data.weekly.test_collect tests.data.schedule.test_collect -v`
Expected: all PASS (behavior-preserving refactor; these tests cover rank direction, span, change semantics)

- [ ] **Step 4: Commit**

```bash
git add src/data/weekly/collect.py src/data/schedule/collect.py
git commit -m "Delegate weekly ranks and schedule team mapping to shared transformations"
```

---

### Task 3: Win-probability context assignment

**Files:**
- Rewrite: `src/data/play_by_play/collect.py`
- Test: `tests/data/play_by_play/test_collect.py` (new directory, no `__init__.py` — matches repo convention)

- [ ] **Step 1: Write the failing tests**

Create `tests/data/play_by_play/test_collect.py`. The `make_play` helper is used by every later task's tests, so it is written now with all play fields:

```python
import unittest
from unittest import mock

import pandas as pd

from src.data.play_by_play import collect


def make_play(posteam='AAA', defteam='BBB', season=2023, week=1, season_type='REG',
              play_id=1, game_id='2023_01_AAA_BBB', is_pass=0, is_rush=0,
              down=1, yardline_100=75.0, third_down_converted=0.0, success=0.0,
              epa=0.0, wp=0.5, xpass=None, fixed_drive=1, fixed_drive_result='Punt'):
    """One synthetic nflfastR play row. 'pass'/'rush' are reserved words as kwargs,
    hence is_pass/is_rush."""
    return {
        'posteam': posteam, 'defteam': defteam, 'season': season, 'week': week,
        'season_type': season_type, 'play_id': play_id, 'game_id': game_id,
        'pass': is_pass, 'rush': is_rush, 'down': down, 'yardline_100': yardline_100,
        'third_down_converted': third_down_converted, 'success': success, 'epa': epa,
        'wp': wp, 'xpass': xpass, 'fixed_drive': fixed_drive,
        'fixed_drive_result': fixed_drive_result,
    }


class TestAssignWpContext(unittest.TestCase):

    def test_thresholds_and_boundaries(self):
        wp = pd.Series([0.5, 0.05, 0.95, 0.951, 0.049, 0.0, 1.0, float('nan')])
        result = collect._assign_wp_context(wp)
        expected = [
            collect.COMPETITIVE,        # 0.5
            collect.COMPETITIVE,        # 0.05 boundary is inclusive
            collect.COMPETITIVE,        # 0.95 boundary is inclusive
            collect.GARBAGE_LEADING,    # 0.951
            collect.GARBAGE_TRAILING,   # 0.049
            collect.GARBAGE_TRAILING,   # 0.0
            collect.GARBAGE_LEADING,    # 1.0
            collect.COMPETITIVE,        # NaN wp defaults to competitive
        ]
        self.assertEqual(list(result), expected)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_assign_wp_context'`

- [ ] **Step 3: Write the implementation**

Replace the entire contents of `src/data/play_by_play/collect.py` (the current file is a 17-line stub whose TODO this work resolves):

```python
import os

import nfl_data_py as nfl
import pandas as pd

from src.data import transformations
from src.data.transformations import (
    TEAM_ABBR_MAPPINGS, TEAM_COL, OPPONENT_TEAM_COL, SEASON_COL, WEEK_COL,
    TEAM_GAME_COUNT_COL, OPP_GAME_COUNT_COL,
)

CACHE_DIR = os.path.join('data', 'play_by_play', 'aggregated')

SEASON_TYPE_COL = 'season_type'
CONTEXT_COL = 'wp_context'

# Verified present for every season 2003-2025. epa/wp/success/fixed_drive are fully
# populated on pass/rush plays back to 2003; xpass is fully null before 2006, which
# makes PROE NaN there by the zero-denominator rule.
REQUIRED_PBP_COLUMNS = [
    'posteam', 'defteam', 'season', 'week', 'season_type', 'play_id', 'game_id',
    'pass', 'rush', 'down', 'yardline_100', 'third_down_converted', 'success',
    'epa', 'wp', 'xpass', 'fixed_drive', 'fixed_drive_result',
]

COMPETITIVE = 'competitive'
GARBAGE_LEADING = 'garbage_leading'
GARBAGE_TRAILING = 'garbage_trailing'
WP_CONTEXTS = [COMPETITIVE, GARBAGE_LEADING, GARBAGE_TRAILING]
GARBAGE_WP_THRESHOLD = 0.95
# 1 - 0.95 is 0.050000000000000044 in floating point, which would misclassify the
# inclusive 0.05 boundary; round keeps the bounds coupled and exact.
GARBAGE_WP_LOWER_THRESHOLD = round(1 - GARBAGE_WP_THRESHOLD, 10)
RED_ZONE_YARDLINE = 20

# wp is always the offense's win probability. From the defense's perspective the
# offense's garbage_leading is garbage_trailing and vice versa; competitive is symmetric.
DEFENSE_CONTEXT_SWAP = {
    COMPETITIVE: COMPETITIVE,
    GARBAGE_LEADING: GARBAGE_TRAILING,
    GARBAGE_TRAILING: GARBAGE_LEADING,
}

AGGREGATION_KEY_COLUMNS = ['posteam', 'season', 'week', 'season_type', 'defteam']

PLAY_COMPONENT_COLUMNS = [
    'play_count', 'epa_sum', 'success_sum',
    'dropback_count', 'dropback_epa_sum', 'dropback_success_sum',
    'rush_count', 'rush_epa_sum', 'rush_success_sum',
    'early_down_count', 'early_down_success_sum',
    'third_down_count', 'third_down_conversion_sum',
    'xpass_play_count', 'pass_minus_xpass_sum',
]
DRIVE_COMPONENT_COLUMNS = ['red_zone_drive_count', 'red_zone_td_drive_count']
COMPONENT_COLUMNS = PLAY_COMPONENT_COLUMNS + DRIVE_COMPONENT_COLUMNS

# (metric_name, numerator_component, denominator_component). Cumulative rates are always
# cumsum(numerator) / cumsum(denominator) so sparse weeks accumulate correctly.
RATE_METRICS = [
    ('epa_per_play', 'epa_sum', 'play_count'),
    ('pass_epa_per_dropback', 'dropback_epa_sum', 'dropback_count'),
    ('rush_epa_per_carry', 'rush_epa_sum', 'rush_count'),
    ('success_rate', 'success_sum', 'play_count'),
    ('pass_success_rate', 'dropback_success_sum', 'dropback_count'),
    ('rush_success_rate', 'rush_success_sum', 'rush_count'),
    ('early_down_success_rate', 'early_down_success_sum', 'early_down_count'),
    ('third_down_conversion_rate', 'third_down_conversion_sum', 'third_down_count'),
    ('red_zone_td_rate', 'red_zone_td_drive_count', 'red_zone_drive_count'),
    ('proe', 'pass_minus_xpass_sum', 'xpass_play_count'),
]


def _assign_wp_context(wp):
    """
    Bucket each play by the offense's pre-snap win probability: competitive
    (0.05 <= wp <= 0.95, boundaries inclusive), garbage_leading (wp > 0.95) or
    garbage_trailing (wp < 0.05). NaN wp defaults to competitive so the play is
    not silently dropped (wp is fully populated on pass/rush plays back to 2003).
    """
    context = pd.Series(COMPETITIVE, index=wp.index)
    context[wp > GARBAGE_WP_THRESHOLD] = GARBAGE_LEADING
    context[wp < GARBAGE_WP_LOWER_THRESHOLD] = GARBAGE_TRAILING
    return context
```

(NaN comparisons are False in pandas, so NaN wp falls through to the competitive default.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect -v`
Expected: 1 test PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/play_by_play/collect.py tests/data/play_by_play/test_collect.py
git commit -m "Add wp context bucketing for play-by-play features"
```

---

### Task 4: Play-level component aggregation

**Files:**
- Modify: `src/data/play_by_play/collect.py`
- Test: `tests/data/play_by_play/test_collect.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/play_by_play/test_collect.py`:

```python
class TestAggregatePlayComponents(unittest.TestCase):

    def setUp(self):
        plays = pd.DataFrame([
            # competitive: one successful pass, one failed rush (both early downs)
            make_play(play_id=1, is_pass=1, down=1, success=1.0, epa=0.5, wp=0.5, xpass=0.6),
            make_play(play_id=2, is_rush=1, down=2, success=0.0, epa=-0.2, wp=0.5, xpass=0.3),
            # garbage_leading: converted third-down pass, no xpass value
            make_play(play_id=3, is_pass=1, down=3, third_down_converted=1.0,
                      success=1.0, epa=1.0, wp=0.96, xpass=None),
            # not a pass or rush play (e.g. kickoff): must be excluded entirely
            make_play(play_id=4, is_pass=0, is_rush=0, epa=2.0, wp=0.5),
        ])
        self.result = collect._aggregate_play_components(plays)

    def _row(self, context):
        return self.result[self.result[collect.CONTEXT_COL] == context].iloc[0]

    def test_competitive_components(self):
        row = self._row(collect.COMPETITIVE)
        self.assertEqual(row['play_count'], 2)
        self.assertAlmostEqual(row['epa_sum'], 0.3)
        self.assertEqual(row['success_sum'], 1)
        self.assertEqual(row['dropback_count'], 1)
        self.assertAlmostEqual(row['dropback_epa_sum'], 0.5)
        self.assertEqual(row['dropback_success_sum'], 1)
        self.assertEqual(row['rush_count'], 1)
        self.assertAlmostEqual(row['rush_epa_sum'], -0.2)
        self.assertEqual(row['rush_success_sum'], 0)
        self.assertEqual(row['early_down_count'], 2)
        self.assertEqual(row['early_down_success_sum'], 1)
        self.assertEqual(row['third_down_count'], 0)
        self.assertEqual(row['third_down_conversion_sum'], 0)
        self.assertEqual(row['xpass_play_count'], 2)
        # (1 - 0.6) + (0 - 0.3) = 0.1
        self.assertAlmostEqual(row['pass_minus_xpass_sum'], 0.1)

    def test_garbage_leading_components(self):
        row = self._row(collect.GARBAGE_LEADING)
        self.assertEqual(row['play_count'], 1)
        self.assertEqual(row['third_down_count'], 1)
        self.assertEqual(row['third_down_conversion_sum'], 1)
        # xpass was NaN: play contributes to neither PROE component
        self.assertEqual(row['xpass_play_count'], 0)
        self.assertEqual(row['pass_minus_xpass_sum'], 0)

    def test_non_pass_rush_plays_are_excluded(self):
        self.assertEqual(self.result['play_count'].sum(), 3)

    def test_grain_is_team_week_context(self):
        expected_keys = collect.AGGREGATION_KEY_COLUMNS + [collect.CONTEXT_COL]
        for key in expected_keys:
            self.assertIn(key, self.result.columns)
        self.assertFalse(self.result.duplicated(subset=expected_keys).any())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect -v`
Expected: new tests FAIL with `AttributeError: ... no attribute '_aggregate_play_components'`

- [ ] **Step 3: Write the implementation**

Append to `src/data/play_by_play/collect.py`:

```python
def _aggregate_play_components(pbp_df):
    """
    Aggregate the offensive play universe (pass or rush plays with an EPA value) to
    component sums per team-week-context. Components are numerator/denominator building
    blocks; rates are only ever computed from season-to-date component sums.
    """
    plays = pbp_df[((pbp_df['pass'] == 1) | (pbp_df['rush'] == 1)) & pbp_df['epa'].notna()].copy()
    plays[CONTEXT_COL] = _assign_wp_context(plays['wp'])

    plays['play_count'] = 1
    plays['epa_sum'] = plays['epa']
    plays['success_sum'] = plays['success']
    plays['dropback_count'] = plays['pass']
    plays['dropback_epa_sum'] = plays['epa'] * plays['pass']
    plays['dropback_success_sum'] = plays['success'] * plays['pass']
    plays['rush_count'] = plays['rush']
    plays['rush_epa_sum'] = plays['epa'] * plays['rush']
    plays['rush_success_sum'] = plays['success'] * plays['rush']
    plays['early_down_count'] = plays['down'].isin([1, 2]).astype(int)
    plays['early_down_success_sum'] = plays['early_down_count'] * plays['success']
    plays['third_down_count'] = (plays['down'] == 3).astype(int)
    plays['third_down_conversion_sum'] = plays['third_down_converted'].fillna(0)
    has_xpass = plays['xpass'].notna()
    plays['xpass_play_count'] = has_xpass.astype(int)
    plays['pass_minus_xpass_sum'] = (plays['pass'] - plays['xpass']).where(has_xpass, 0)

    return plays.groupby(
        AGGREGATION_KEY_COLUMNS + [CONTEXT_COL]
    )[PLAY_COMPONENT_COLUMNS].sum().reset_index()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/play_by_play/collect.py tests/data/play_by_play/test_collect.py
git commit -m "Aggregate play-by-play plays to team-week component sums per wp context"
```

---

### Task 5: Red-zone drive components

**Files:**
- Modify: `src/data/play_by_play/collect.py`
- Test: `tests/data/play_by_play/test_collect.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/play_by_play/test_collect.py`:

```python
class TestAggregateRedZoneComponents(unittest.TestCase):

    def setUp(self):
        plays = pd.DataFrame([
            # Drive 1: reaches the red zone (min yardline 18), ends in a TD.
            # First play wp 0.5 -> the whole drive counts as competitive even though
            # a later play is in garbage range.
            make_play(play_id=1, fixed_drive=1, yardline_100=45.0, wp=0.50,
                      fixed_drive_result='Touchdown', is_pass=1),
            make_play(play_id=2, fixed_drive=1, yardline_100=18.0, wp=0.96,
                      fixed_drive_result='Touchdown', is_rush=1),
            # Drive 2: never reaches the red zone -> excluded
            make_play(play_id=3, fixed_drive=2, yardline_100=60.0, wp=0.5,
                      fixed_drive_result='Punt', is_pass=1),
            # Drive 3: red zone field goal in garbage_leading (first play wp 0.97)
            make_play(play_id=4, fixed_drive=3, yardline_100=15.0, wp=0.97,
                      fixed_drive_result='Field goal', is_rush=1),
        ])
        self.result = collect._aggregate_red_zone_components(plays)

    def _row(self, context):
        return self.result[self.result[collect.CONTEXT_COL] == context].iloc[0]

    def test_red_zone_drive_counts_by_context(self):
        competitive = self._row(collect.COMPETITIVE)
        self.assertEqual(competitive['red_zone_drive_count'], 1)
        self.assertEqual(competitive['red_zone_td_drive_count'], 1)
        leading = self._row(collect.GARBAGE_LEADING)
        self.assertEqual(leading['red_zone_drive_count'], 1)
        self.assertEqual(leading['red_zone_td_drive_count'], 0)

    def test_non_red_zone_drives_are_excluded(self):
        self.assertEqual(self.result['red_zone_drive_count'].sum(), 2)

    def test_drive_context_uses_first_play_in_play_id_order(self):
        # Same drive 1 rows but shuffled: context must still come from play_id 1 (wp 0.5)
        plays = pd.DataFrame([
            make_play(play_id=2, fixed_drive=1, yardline_100=18.0, wp=0.96,
                      fixed_drive_result='Touchdown', is_rush=1),
            make_play(play_id=1, fixed_drive=1, yardline_100=45.0, wp=0.50,
                      fixed_drive_result='Touchdown', is_pass=1),
        ])
        result = collect._aggregate_red_zone_components(plays)
        self.assertEqual(result[collect.CONTEXT_COL].iloc[0], collect.COMPETITIVE)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect -v`
Expected: new tests FAIL with `AttributeError: ... no attribute '_aggregate_red_zone_components'`

- [ ] **Step 3: Write the implementation**

Append to `src/data/play_by_play/collect.py`:

```python
def _aggregate_red_zone_components(pbp_df):
    """
    Count red-zone trips per team-week-context at the drive level: a drive counts as a
    red-zone trip when any of its plays starts at or inside the opponent's 20. A drive's
    context comes from the win probability on its first play (drives can drift across
    contexts mid-drive; the first play reflects the situation the drive started in).
    fixed_drive numbers drives across the whole game, so (game_id, fixed_drive) is unique.
    """
    drive_plays = pbp_df[pbp_df['fixed_drive'].notna() & pbp_df['posteam'].notna()].sort_values('play_id')
    drives = drive_plays.groupby(['game_id', 'fixed_drive'] + AGGREGATION_KEY_COLUMNS).agg(
        min_yardline_100=('yardline_100', 'min'),
        first_play_wp=('wp', 'first'),
        drive_result=('fixed_drive_result', 'first'),
    ).reset_index()

    red_zone_drives = drives[drives['min_yardline_100'] <= RED_ZONE_YARDLINE].copy()
    red_zone_drives[CONTEXT_COL] = _assign_wp_context(red_zone_drives['first_play_wp'])
    red_zone_drives['red_zone_drive_count'] = 1
    red_zone_drives['red_zone_td_drive_count'] = (red_zone_drives['drive_result'] == 'Touchdown').astype(int)

    return red_zone_drives.groupby(
        AGGREGATION_KEY_COLUMNS + [CONTEXT_COL]
    )[DRIVE_COMPONENT_COLUMNS].sum().reset_index()
```

(`fixed_drive_result` is 'Touchdown' only for offensive touchdowns; defensive scores appear as 'Opp touchdown' and correctly don't count.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/play_by_play/collect.py tests/data/play_by_play/test_collect.py
git commit -m "Count red-zone drive trips and touchdowns per wp context"
```

---

### Task 6: Season aggregation — merge, pivot contexts wide, map team abbreviations

**Files:**
- Modify: `src/data/play_by_play/collect.py`
- Test: `tests/data/play_by_play/test_collect.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/play_by_play/test_collect.py`:

```python
class TestAggregateSeason(unittest.TestCase):

    def setUp(self):
        plays = pd.DataFrame([
            # 'SD' must come out as 'LAC'; opponent 'OAK' as 'LV'
            make_play(posteam='SD', defteam='OAK', play_id=1, is_pass=1, epa=0.5,
                      success=1.0, wp=0.5, yardline_100=15.0, fixed_drive=1,
                      fixed_drive_result='Touchdown'),
            make_play(posteam='OAK', defteam='SD', play_id=2, is_rush=1, epa=-0.1,
                      wp=0.4, fixed_drive=2),
        ])
        self.result = collect._aggregate_season(plays)

    def test_team_abbreviations_are_normalized(self):
        self.assertEqual(set(self.result['team']), {'LAC', 'LV'})
        self.assertEqual(set(self.result['opp_team']), {'LAC', 'LV'})

    def test_one_row_per_team_game(self):
        self.assertEqual(len(self.result), 2)

    def test_all_component_context_columns_exist_and_missing_contexts_are_zero(self):
        for component in collect.COMPONENT_COLUMNS:
            for context in collect.WP_CONTEXTS:
                self.assertIn(f'{component}_{context}', self.result.columns)
        lac = self.result[self.result['team'] == 'LAC'].iloc[0]
        # LAC had no garbage-time plays: those component cells are 0 (no plays), not NaN
        self.assertEqual(lac['play_count_garbage_leading'], 0)
        self.assertEqual(lac['play_count_garbage_trailing'], 0)

    def test_play_and_drive_components_land_on_the_same_row(self):
        lac = self.result[self.result['team'] == 'LAC'].iloc[0]
        self.assertEqual(lac['play_count_competitive'], 1)
        self.assertEqual(lac['red_zone_drive_count_competitive'], 1)
        self.assertEqual(lac['red_zone_td_drive_count_competitive'], 1)

    def test_missing_required_column_raises(self):
        plays = pd.DataFrame([make_play()]).drop(columns=['epa'])
        with self.assertRaises(ValueError):
            collect._aggregate_season(plays)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect -v`
Expected: new tests FAIL with `AttributeError: ... no attribute '_aggregate_season'`

- [ ] **Step 3: Write the implementation**

Append to `src/data/play_by_play/collect.py`:

```python
def _validate_required_columns(pbp_df):
    missing = sorted(set(REQUIRED_PBP_COLUMNS) - set(pbp_df.columns))
    if missing:
        raise ValueError(f'Play-by-play data is missing required columns: {missing}')


def _pivot_context_components(components):
    """
    Pivot the long (team-week-context) component frame wide so each component becomes
    three columns, one per context (e.g. play_count_competitive). A context absent for a
    team-week means zero plays happened in it, so component sums fill with 0; the
    zero-denominator rule later turns the corresponding rates into NaN.
    """
    pivoted = components.pivot_table(
        index=[TEAM_COL, SEASON_COL, WEEK_COL, SEASON_TYPE_COL, OPPONENT_TEAM_COL],
        columns=CONTEXT_COL,
        values=COMPONENT_COLUMNS,
        aggfunc='sum',
        fill_value=0,
    )
    pivoted.columns = [f'{component}_{context}' for component, context in pivoted.columns]
    for component in COMPONENT_COLUMNS:
        for context in WP_CONTEXTS:
            column = f'{component}_{context}'
            if column not in pivoted.columns:
                pivoted[column] = 0
    return pivoted.reset_index()


def _aggregate_season(pbp_df):
    """
    Reduce one season of raw play-by-play to one row per team-game with component sums
    per wp context. This is the frame that gets cached per season.
    """
    _validate_required_columns(pbp_df)
    pbp_df = pbp_df[pbp_df['posteam'].notna() & pbp_df['defteam'].notna()].copy()
    pbp_df['posteam'] = pbp_df['posteam'].replace(TEAM_ABBR_MAPPINGS)
    pbp_df['defteam'] = pbp_df['defteam'].replace(TEAM_ABBR_MAPPINGS)

    play_components = _aggregate_play_components(pbp_df)
    drive_components = _aggregate_red_zone_components(pbp_df)
    components = play_components.merge(
        drive_components, on=AGGREGATION_KEY_COLUMNS + [CONTEXT_COL], how='outer'
    ).fillna(0)
    components = components.rename(columns={'posteam': TEAM_COL, 'defteam': OPPONENT_TEAM_COL})
    return _pivot_context_components(components)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/play_by_play/collect.py tests/data/play_by_play/test_collect.py
git commit -m "Aggregate a season of play-by-play to wide team-week component sums"
```

---

### Task 7: Cached per-season collector

**Files:**
- Modify: `src/data/play_by_play/collect.py`
- Test: `tests/data/play_by_play/test_collect.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/play_by_play/test_collect.py` (add `import tempfile` to the top of the file with the other imports):

```python
class TestGetPlayByPlayDataCaching(unittest.TestCase):

    def _synthetic_pbp(self):
        return pd.DataFrame([
            make_play(posteam='AAA', defteam='BBB', play_id=1, is_pass=1, epa=0.5, wp=0.5),
            make_play(posteam='BBB', defteam='AAA', play_id=2, is_rush=1, epa=0.1, wp=0.5),
        ])

    def test_download_happens_once_then_cache_is_read(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.object(collect, 'CACHE_DIR', tmp_dir), \
                    mock.patch.object(collect.nfl, 'import_pbp_data',
                                      return_value=self._synthetic_pbp()) as import_mock:
                first = collect.get_play_by_play_data([2023])
                second = collect.get_play_by_play_data([2023])
            import_mock.assert_called_once()
            pd.testing.assert_frame_equal(
                first.sort_index(axis=1), second.sort_index(axis=1)
            )

    def test_refresh_forces_redownload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.object(collect, 'CACHE_DIR', tmp_dir), \
                    mock.patch.object(collect.nfl, 'import_pbp_data',
                                      return_value=self._synthetic_pbp()) as import_mock:
                collect.get_play_by_play_data([2023])
                collect.get_play_by_play_data([2023], refresh=True)
            self.assertEqual(import_mock.call_count, 2)

    def test_years_type_validation(self):
        with self.assertRaises(TypeError):
            collect.get_play_by_play_data('2023')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect -v`
Expected: new tests FAIL (`get_play_by_play_data` signature mismatch / TypeError test fails since the stub accepts anything)

- [ ] **Step 3: Write the implementation**

In `src/data/play_by_play/collect.py`, replace the old stub function `get_play_by_play_data` entirely (it and its TODO comment are superseded) with:

```python
def get_play_by_play_data(years, refresh=False):
    """
    Return team-week component sums for the specified season(s), one row per team-game.

    Raw play-by-play is ~50k rows x 396 columns per season, so each season is downloaded
    once (selecting only REQUIRED_PBP_COLUMNS), aggregated, and cached to
    CACHE_DIR/{year}.parquet. Subsequent calls read the small aggregated frame. Pass
    refresh=True to re-download (needed while a season is in progress).

    :param years: list of years to collect data for or a single year
    :param refresh: re-download and re-aggregate even when a cache file exists
    :return: concatenated component frame across the requested seasons
    """
    if isinstance(years, int):
        years = [years]
    elif not isinstance(years, list):
        raise TypeError('years must be a list or an integer')

    season_frames = []
    for year in years:
        cache_path = os.path.join(CACHE_DIR, f'{year}.parquet')
        if os.path.exists(cache_path) and not refresh:
            season_frames.append(pd.read_parquet(cache_path))
            continue
        raw_pbp = nfl.import_pbp_data(years=[year], columns=REQUIRED_PBP_COLUMNS, downcast=False)
        season_components = _aggregate_season(raw_pbp)
        os.makedirs(CACHE_DIR, exist_ok=True)
        season_components.to_parquet(cache_path, index=False)
        season_frames.append(season_components)
    return pd.concat(season_frames, ignore_index=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/play_by_play/collect.py tests/data/play_by_play/test_collect.py
git commit -m "Cache aggregated play-by-play components per season"
```

---

### Task 8: Game counts and cumulative rate columns (offense + mirrored defense)

**Files:**
- Modify: `src/data/play_by_play/collect.py`
- Test: `tests/data/play_by_play/test_collect.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/play_by_play/test_collect.py`. The helper builds component rows with every component zeroed unless overridden:

```python
def make_component_row(team, opp_team, season=2023, week=1, season_type='REG', **overrides):
    row = {'team': team, 'opp_team': opp_team, 'season': season, 'week': week,
           'season_type': season_type}
    for component in collect.COMPONENT_COLUMNS:
        for context in collect.WP_CONTEXTS:
            row[f'{component}_{context}'] = 0
    row.update(overrides)
    return row


class TestCumulativeRateColumns(unittest.TestCase):

    def setUp(self):
        components = pd.DataFrame([
            # AAA week 1 vs BBB: 10 competitive plays / 5 EPA; 2 leading-garbage plays / 1 EPA
            make_component_row('AAA', 'BBB', week=1,
                               play_count_competitive=10, epa_sum_competitive=5.0,
                               play_count_garbage_leading=2, epa_sum_garbage_leading=1.0),
            make_component_row('BBB', 'AAA', week=1,
                               play_count_competitive=8, epa_sum_competitive=-2.0),
            # AAA week 2 vs CCC: efficiency drops; first trailing-garbage snaps appear.
            # Play count differs from week 1 on purpose so ratio-of-cumsums and
            # mean-of-weekly-rates give different answers and the test can tell them apart.
            make_component_row('AAA', 'CCC', week=2,
                               play_count_competitive=20, epa_sum_competitive=1.0,
                               play_count_garbage_trailing=4, epa_sum_garbage_trailing=-1.0),
            make_component_row('CCC', 'AAA', week=2,
                               play_count_competitive=12, epa_sum_competitive=0.0),
        ]).reset_index(drop=True)
        components = collect._add_game_count_columns(components).reset_index(drop=True)
        self.result = collect._add_cumulative_rate_columns(components)

    def _value(self, team, week, col):
        row = self.result[(self.result['team'] == team) & (self.result['week'] == week)]
        return row[col].values[0]

    def test_cumulative_rate_is_ratio_of_cumulative_sums(self):
        # Week 1: 5/10 = 0.5. Week 2: (5+1)/(10+20) = 0.2.
        # A wrong mean-of-weekly-rates implementation would give (0.5 + 0.05)/2 = 0.275.
        self.assertAlmostEqual(
            self._value('AAA', 1, 'off_epa_per_play_competitive_cumulative_average'), 0.5)
        self.assertAlmostEqual(
            self._value('AAA', 2, 'off_epa_per_play_competitive_cumulative_average'), 0.2)

    def test_zero_denominator_is_nan_then_recovers(self):
        # AAA had no trailing-garbage snaps in week 1 -> NaN, not 0
        self.assertTrue(pd.isna(
            self._value('AAA', 1, 'off_epa_per_play_garbage_trailing_cumulative_average')))
        # Week 2: cumulative = (0 + -1.0) / (0 + 4) = -0.25
        self.assertAlmostEqual(
            self._value('AAA', 2, 'off_epa_per_play_garbage_trailing_cumulative_average'), -0.25)

    def test_defense_mirrors_opponent_offense_on_first_game(self):
        # def_opp_* on AAA's week-1 row is BBB's defense; BBB's only game so far is this
        # one, so it equals AAA's own offense value from the same game.
        self.assertAlmostEqual(
            self._value('AAA', 1, 'def_opp_epa_per_play_competitive_cumulative_average'), 0.5)

    def test_defense_context_labels_are_swapped(self):
        # AAA's leading-garbage offense (1.0 EPA / 2 plays) is BBB's trailing-garbage
        # defense: the defense was on the field while ITS team was likely losing.
        self.assertAlmostEqual(
            self._value('AAA', 1, 'def_opp_epa_per_play_garbage_trailing_cumulative_average'), 0.5)
        self.assertTrue(pd.isna(
            self._value('AAA', 1, 'def_opp_epa_per_play_garbage_leading_cumulative_average')))

    def test_defense_accumulates_across_opponent_games(self):
        # CCC's week-2 def_opp row tracks AAA's defense. AAA defended BBB week 1
        # (8 plays, -2 EPA) and CCC week 2 (12 plays, 0 EPA): (-2+0)/(8+12) = -0.1
        self.assertAlmostEqual(
            self._value('CCC', 2, 'def_opp_epa_per_play_competitive_cumulative_average'), -0.1)

    def test_all_rate_metric_columns_exist(self):
        for metric, _, _ in collect.RATE_METRICS:
            for context in collect.WP_CONTEXTS:
                self.assertIn(f'off_{metric}_{context}_cumulative_average', self.result.columns)
                self.assertIn(f'def_opp_{metric}_{context}_cumulative_average', self.result.columns)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect -v`
Expected: new tests FAIL with `AttributeError: ... no attribute '_add_game_count_columns'`

- [ ] **Step 3: Write the implementation**

Append to `src/data/play_by_play/collect.py`:

```python
def _add_game_count_columns(df):
    """
    Add per-team and per-opponent game counters within each season. Week number is not a
    reliable divisor because of byes, so cumulative math orders and groups by these
    counters, mirroring the weekly module.
    """
    df.sort_values(by=[TEAM_COL, SEASON_COL, WEEK_COL], inplace=True)
    df[TEAM_GAME_COUNT_COL] = df.groupby([TEAM_COL, SEASON_COL]).cumcount() + 1
    df.sort_values(by=[OPPONENT_TEAM_COL, SEASON_COL, WEEK_COL], inplace=True)
    df[OPP_GAME_COUNT_COL] = df.groupby([OPPONENT_TEAM_COL, SEASON_COL]).cumcount() + 1
    df.sort_values(by=[TEAM_COL, SEASON_COL, WEEK_COL], inplace=True)
    return df


def _add_cumulative_rate_columns(df):
    """
    Add season-to-date rate columns for every metric/context, offense and defense.

    Every rate is cumsum(numerator) / cumsum(denominator) within the season - never an
    average of weekly rates - so thin contexts accumulate correctly. A zero cumulative
    denominator yields NaN: no snaps means no rate, not a rate of zero.

    Offense accumulates within (team, season). Defense accumulates the same base
    components within (opp_team, season) - exactly the weekly module's def_opp pattern,
    so def_opp_* on a row describes the opponent's defense season-to-date. Defense
    context labels are swapped via DEFENSE_CONTEXT_SWAP because wp belongs to the
    offense: plays where the offense was garbage_leading are the defense's
    garbage_trailing snaps.

    Requires a unique index (reset_index before calling) and game-count columns.
    """
    component_cols = [f'{component}_{context}'
                      for component in COMPONENT_COLUMNS for context in WP_CONTEXTS]

    df = df.sort_values(by=[TEAM_COL, SEASON_COL, TEAM_GAME_COUNT_COL])
    off_cumulative = df.groupby([TEAM_COL, SEASON_COL])[component_cols].cumsum()

    opp_ordered = df.sort_values(by=[OPPONENT_TEAM_COL, SEASON_COL, OPP_GAME_COUNT_COL])
    def_cumulative = opp_ordered.groupby(
        [OPPONENT_TEAM_COL, SEASON_COL]
    )[component_cols].cumsum()

    rate_columns = {}
    for metric, numerator, denominator in RATE_METRICS:
        for context in WP_CONTEXTS:
            offense_numerator = off_cumulative[f'{numerator}_{context}']
            offense_denominator = off_cumulative[f'{denominator}_{context}']
            rate_columns[f'off_{metric}_{context}_cumulative_average'] = (
                offense_numerator / offense_denominator.where(offense_denominator != 0)
            )

            defense_context = DEFENSE_CONTEXT_SWAP[context]
            defense_numerator = def_cumulative[f'{numerator}_{context}']
            defense_denominator = def_cumulative[f'{denominator}_{context}']
            rate_columns[f'def_opp_{metric}_{defense_context}_cumulative_average'] = (
                defense_numerator / defense_denominator.where(defense_denominator != 0)
            )

    return pd.concat([df, pd.DataFrame(rate_columns)], axis=1)
```

(The cumsum frames keep the original row index, so the final `pd.concat(axis=1)` aligns by index regardless of the different sort orders.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/play_by_play/collect.py tests/data/play_by_play/test_collect.py
git commit -m "Compute cumulative component-ratio rates for offense and mirrored defense"
```

---

### Task 9: Ranks and the public feature API

**Files:**
- Modify: `src/data/play_by_play/collect.py`
- Test: `tests/data/play_by_play/test_collect.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/play_by_play/test_collect.py`:

```python
class TestGetPlayByPlayFeatures(unittest.TestCase):

    def _synthetic_pbp(self):
        # Two games in one week: four teams so ranks span 1..4
        return pd.DataFrame([
            make_play(posteam='AAA', defteam='BBB', game_id='g1', play_id=1,
                      is_pass=1, epa=1.0, success=1.0, wp=0.5),
            make_play(posteam='BBB', defteam='AAA', game_id='g1', play_id=2,
                      is_rush=1, epa=0.5, success=1.0, wp=0.5),
            make_play(posteam='CCC', defteam='DDD', game_id='g2', play_id=1,
                      is_pass=1, epa=-0.5, success=0.0, wp=0.5),
            make_play(posteam='DDD', defteam='CCC', game_id='g2', play_id=2,
                      is_rush=1, epa=-1.0, success=0.0, wp=0.5),
        ])

    def _features(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.object(collect, 'CACHE_DIR', tmp_dir), \
                    mock.patch.object(collect.nfl, 'import_pbp_data',
                                      return_value=self._synthetic_pbp()):
                return collect.get_play_by_play_features([2023])

    def test_feature_count_and_naming(self):
        features = self._features()
        feature_cols = [col for col in features.columns
                        if col.startswith('off_') or col.startswith('def_opp_')]
        # 10 metrics x 3 contexts x 2 sides x 3 column kinds (rate, rank, rank_change)
        self.assertEqual(len(feature_cols), 180)
        self.assertEqual(list(features.columns[:3]), ['team', 'season', 'week'])

    def test_offense_rank_one_is_best_epa(self):
        features = self._features()
        rank_col = 'off_epa_per_play_competitive_cumulative_average_rank'
        best = features[features[rank_col] == 1]
        self.assertEqual(best['team'].values[0], 'AAA')
        worst = features[features[rank_col] == 4]
        self.assertEqual(worst['team'].values[0], 'DDD')

    def test_defense_rank_one_allows_least_epa(self):
        features = self._features()
        # def_opp on a row describes that row's opponent. CCC's defense allowed DDD's
        # -1.0 EPA/play (least allowed -> rank 1) and CCC is the opponent on DDD's row.
        rank_col = 'def_opp_epa_per_play_competitive_cumulative_average_rank'
        best = features[features[rank_col] == 1]
        self.assertEqual(best['team'].values[0], 'DDD')

    def test_no_component_columns_leak_into_output(self):
        features = self._features()
        for component in collect.COMPONENT_COLUMNS:
            for context in collect.WP_CONTEXTS:
                self.assertNotIn(f'{component}_{context}', features.columns)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect -v`
Expected: new tests FAIL with `AttributeError: ... no attribute 'get_play_by_play_features'`

- [ ] **Step 3: Write the implementation**

Append to `src/data/play_by_play/collect.py`:

```python
def get_play_by_play_features(years, refresh=False):
    """
    Build the model-facing play-by-play feature frame: one row per team-game keyed by
    (team, season, week), with cumulative rate, rank, and rank-change columns for every
    Phase 1 metric, wp context, and side of the ball. Column naming follows the weekly
    module's off_* / def_opp_* convention so collect_all's home/away renames, target
    duplication, and one-week leakage shift apply to these features unchanged.

    :param years: list of years to collect data for or a single year
    :param refresh: re-download and re-aggregate even when a season cache file exists
    :return: feature frame ready to merge onto the weekly frame
    """
    components = get_play_by_play_data(years, refresh=refresh).reset_index(drop=True)
    components = _add_game_count_columns(components).reset_index(drop=True)
    df = _add_cumulative_rate_columns(components)

    off_cols = [col for col in df.columns
                if col.startswith('off_') and col.endswith('_cumulative_average')]
    def_cols = [col for col in df.columns
                if col.startswith('def_opp_') and col.endswith('_cumulative_average')]
    df = transformations.add_rank_and_rank_change_columns(df, off_cols, def_cols)

    feature_cols = [col for col in df.columns
                    if col.startswith('off_') or col.startswith('def_opp_')]
    return df[[TEAM_COL, SEASON_COL, WEEK_COL] + feature_cols].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect -v`
Expected: all PASS (full module: ~20 tests)

- [ ] **Step 5: Commit**

```bash
git add src/data/play_by_play/collect.py tests/data/play_by_play/test_collect.py
git commit -m "Expose ranked play-by-play features keyed by team-season-week"
```

---

### Task 10: collect_all integration

**Files:**
- Modify: `src/data/collect_all.py:26-54`
- Test: `tests/data/test_collect_all.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_collect_all.py`. NOTE: this is a network-backed integration test like the rest of the file; the first run downloads + caches 2023 play-by-play (~1 minute), later runs read the cache:

```python
class TestPlayByPlayIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.all_data = collect_all.get_schedule_and_weekly_data([2023], include_play_by_play=True)

    def test_play_by_play_columns_reach_all_four_families(self):
        expected_columns = [
            'off_target_epa_per_play_competitive_cumulative_average_rank',
            'off_opp_epa_per_play_competitive_cumulative_average_rank',
            'def_target_epa_per_play_garbage_trailing_cumulative_average_rank',
            'def_opp_epa_per_play_garbage_trailing_cumulative_average_rank',
        ]
        for column in expected_columns:
            self.assertIn(column, self.all_data.columns)

    def test_play_by_play_features_are_populated_after_week_two(self):
        late_weeks = self.all_data[self.all_data['week'] >= 3]
        non_null_fraction = late_weeks[
            'off_target_epa_per_play_competitive_cumulative_average'
        ].notna().mean()
        self.assertGreater(non_null_fraction, 0.95)

    def test_play_by_play_features_are_shifted_off_week_one(self):
        # The leakage shift moves every stat forward one game, so week 1 (a team's first
        # game) must have no play-by-play feature values. Unshifted columns would be
        # populated here.
        week_one = self.all_data[self.all_data['week'] == 1]
        self.assertTrue(
            week_one['off_target_epa_per_play_competitive_cumulative_average'].isna().all()
        )

    def test_excluding_play_by_play_keeps_legacy_columns_only(self):
        legacy = collect_all.get_schedule_and_weekly_data([2023])
        pbp_columns = [col for col in legacy.columns if 'epa_per_play' in col]
        self.assertEqual(pbp_columns, [])
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `conda run -n nfl-predictions python -m unittest tests.data.test_collect_all.TestPlayByPlayIntegration -v`
Expected: FAIL with `TypeError: get_schedule_and_weekly_data() got an unexpected keyword argument 'include_play_by_play'`

- [ ] **Step 3: Write the implementation**

In `src/data/collect_all.py`, change the signature and weekly-frame construction of `get_schedule_and_weekly_data` (lines 26–41). The docstring's source list gains Play by Play, and the merge happens before any home/away logic so the existing rails handle everything downstream:

```python
def get_schedule_and_weekly_data(years, include_play_by_play=False):
    """
    Collects schedule and weekly data for the specified year(s). Merges the data from the following sources:
        - Weekly
        - Schedule
        - Play by Play (optional): wp-context-split efficiency/situational features, merged
          onto the weekly frame by (team, season, week) before the home/away merge so the
          target/opp renames and the one-week leakage shift apply to them unchanged

    :param years: list of years to collect data for or a single year
    :param include_play_by_play: merge play-by-play features onto the weekly frame
    :return:
    """
    if isinstance(years, int):
        years = [years]
    elif not isinstance(years, list):
        raise TypeError('years must be a list or an integer')

    weekly_data = weekly_collect.get_weekly_data(years).reset_index(drop=True)
    if include_play_by_play:
        pbp_features = pbp_collect.get_play_by_play_features(years)
        weekly_data = weekly_data.merge(
            pbp_features,
            on=['team', 'season', 'week'],
            how='left',
            validate='one_to_one',
        )
    schedule_data = schedule_collect.get_schedule_data(years).reset_index(drop=True)
```

The rest of the function body (from `home_team_is_team = ...` on) is unchanged. The import `from src.data.play_by_play import collect as pbp_collect` already exists at the top of the file (line 4).

- [ ] **Step 4: Run the new tests to verify they pass (slow: downloads 2023 pbp on first run)**

Run: `conda run -n nfl-predictions python -m unittest tests.data.test_collect_all.TestPlayByPlayIntegration -v`
Expected: 4 tests PASS

- [ ] **Step 5: Run the full pre-existing collect_all test class to confirm no regression (slow, network)**

Run: `conda run -n nfl-predictions python -m unittest tests.data.test_collect_all -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/data/collect_all.py tests/data/test_collect_all.py
git commit -m "Merge play-by-play features into the schedule-and-weekly pipeline behind a flag"
```

---

### Task 11: Assembly script, cache commit, and dataset regeneration

**Files:**
- Modify: `scripts/data_assembly/predict_game_winner/schedule_and_weekly.py:17`
- Commit: `data/play_by_play/aggregated/*.parquet` (small per-season component frames — committing them makes test and assembly runs reproducible and fast)

- [ ] **Step 1: Update the assembly script**

Change line 17 of `scripts/data_assembly/predict_game_winner/schedule_and_weekly.py` from:

```python
schedule_and_weekly_df = get_schedule_and_weekly_data(YEARS)
```

to:

```python
schedule_and_weekly_df = get_schedule_and_weekly_data(YEARS, include_play_by_play=True)
```

- [ ] **Step 2: Regenerate the model-ready dataset (LONG: first run downloads 23 seasons of pbp, ~20–30 min; subsequent runs read the season caches)**

Run: `conda run -n nfl-predictions python -m scripts.data_assembly.predict_game_winner.schedule_and_weekly`
Expected: per-season download progress lines, then exits 0. `data/play_by_play/aggregated/` contains `2003.parquet` … `2025.parquet`.

- [ ] **Step 3: Verify the regenerated dataset**

Run:

```bash
conda run -n nfl-predictions python -c "
import pandas as pd
df = pd.read_parquet('data/predict_games/input_data/schedule_and_weekly.parquet')
pbp_cols = [c for c in df.columns if any(m in c for m in ('epa_per_play', 'success_rate', 'third_down_conversion_rate', 'red_zone_td_rate', 'proe'))]
print('total columns:', len(df.columns))
print('pbp feature columns:', len(pbp_cols))
sample = 'off_target_epa_per_play_competitive_cumulative_average_rank'
print('sample col present:', sample in df.columns)
late = df[(df['week'] >= 3)]
print('sample col non-null fraction weeks>=3:', round(late[sample].notna().mean(), 3))
features = pd.read_csv('data/predict_games/model_features_in/xgb_features_list.csv')
print('features list length:', len(features))
"
```

Expected: pbp feature columns = 360 (each of the 180 features appears twice, once with a `target` prefix and once with an `opp` prefix), sample col present True, non-null fraction > 0.9, features list grew by ~360 over its previous length.

- [ ] **Step 4: Commit**

```bash
git add scripts/data_assembly/predict_game_winner/schedule_and_weekly.py \
        data/play_by_play/aggregated \
        data/predict_games/input_data/schedule_and_weekly.parquet \
        data/predict_games/model_features_in/xgb_features_list.csv
git commit -m "Regenerate model dataset with Phase 1 play-by-play features"
```

---

### Task 12: Phase 1 evaluation (manual checkpoint)

This is the spec's per-phase evaluation gate, not an automated step. Re-run cross-validated RFE on the expanded candidate set and compare against the Run 5 baseline (pooled 2024–2025 hold-out AUROC 0.697 / accuracy 0.647).

- [ ] **Step 1: Execute the RFE notebook (very long-running; can also be run interactively in Jupyter)**

Run: `conda run -n nfl-predictions jupyter nbconvert --to notebook --execute --inplace notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/rfe.ipynb`
Expected: completes without error; selected-feature CSV updated under `data/predict_games/model_features_in/`.

- [ ] **Step 2: Record results**

Compare hold-out AUROC/accuracy to the baseline and record the outcome (feature count, how many pbp features survived RFE, metric deltas) in the README results table, as Runs 3–5 were recorded. Decision: if Phase 1 lifts hold-out AUROC, proceed to Phase 2 (directional features) on this foundation; if flat, discuss before building Phase 2.

- [ ] **Step 3: Commit results**

```bash
git add data/predict_games/model_features_in README.md notebooks
git commit -m "Record Phase 1 play-by-play RFE results"
```
