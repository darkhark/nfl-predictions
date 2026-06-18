# Madden Ratings — Features & Integration Implementation Plan (Plan 2 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Plan 1's player-season table into leakage-safe team-game Madden features
(per-position, grouped, matchup, with diffs), wire them into the prediction dataset as a
single `madden_ratings` feature group, and run the base-vs-base+madden ablation.

**Architecture:** Add `src/data/madden/starters.py` (weekly leakage-safe starters from
depth charts + injuries), `src/data/madden/features.py` (slot/group overalls, diffs,
matchup deltas), and `src/data/madden/collect.py` (`get_madden_data(years)` →
team-week `madden_*` frame). Integrate in `collect_all.py` via an **explicit target/opp
double-merge after the one-week shift** (Madden week-N info is pre-game-N, so it aligns
to the game week, not the shifted cumulative stats). Register the group in
`partition.py`. Rebuild the parquet and ablate.

**Tech Stack:** Python 3.13, pandas, `nfl_data_py` (`import_depth_charts`,
`import_injuries`), `unittest`, the `feature_groups/` ablation harness.

## Global Constraints

- **Consumes Plan 1 contract** (player-season frame): `season, full_name, team,
  position(Madden side-aware), overall, weight, power_moves, finesse_moves, gsis_id,
  role∈{edge,interior_dl,off_ball_lb,cornerback,safety,qb,backfield,receiver,tight_end,
  interior_ol,exterior_ol,specialist,unknown}, side∈{left,right,none}`.
- **Leakage rule:** only pre-kickoff info. Starters = depth-chart rank-1 minus
  `report_status ∈ {Out, Doubtful}`, next-man-up to highest available. Madden overalls
  are preseason (season-constant). Features align to the **game week** (merged after the
  pipeline's `_shift_data`), never shifted with cumulative stats.
- **Diff windows are games-played** (team's previous game / 4 games back, skipping byes),
  computed per `(team, season)` ordered by `week`. First game → NA prev-diff; games 1–4 →
  NA 4-game diff.
- **Column naming:** team-week columns use the neutral prefix `madden_` and are merged as
  `target_madden_*` / `opp_madden_*`; matchup columns are `madden_matchup_*`. Every model
  column thus contains the substring `madden`, which is how `partition.py` classifies them.
- **nfl_data_py schemas (verified, 2023):**
  - depth charts: `season, club_code, week, game_type, depth_team(str '1','2'…),
    gsis_id, position(coarse: QB/RB/FB/WR/TE/T/G/C/DE/DT/NT/OLB/ILB/MLB/CB/FS/SS/DB/K/P/LS),
    formation, full_name`.
  - injuries: `season, team, week, gsis_id, report_status∈{Out,Questionable,Doubtful}`.
- **Slots / groups (canonical, substring-safe):**
  - granular: `qb, rb, wr1, wr2, wr3, te, lt, lg, c, rg, rt, edge_left, edge_right`
  - groups: `backfield, receivers, interior_ol, exterior_ol, edge, interior_dl,
    linebacker, cornerback, safety`
  - measures per slot/group: `_ovr, _ovr_diff_prev, _ovr_diff_4g, _ovr_diff_prev_season`

---

## File Structure

- `src/data/madden/starters.py` — `get_weekly_starters(years)`; depth chart + injury
  reconciliation, next-man-up. Pure helpers + one I/O entry point.
- `src/data/madden/features.py` — `build_team_week_overalls`, `add_within_season_diffs`,
  `add_prev_season_diff`, `add_madden_matchup_columns`.
- `src/data/madden/collect.py` — `get_madden_data(years)` orchestrator → team-week frame.
- Modify: `src/data/collect_all.py` (double-merge + matchup call).
- Modify: `data_science_utilities/feature_groups/partition.py` (madden_ratings family).
- Modify: `scripts/data_assembly/predict_game_winner/schedule_and_weekly.py` (no code
  change expected — madden columns flow through automatically; verify only).
- Tests: `tests/data/madden/test_starters.py`, `test_features.py`,
  `tests/data/test_collect_all.py` (extend), `tests/data_science_utilities/test_feature_group_partition.py` (extend).

**Plan 2 team-week output contract (`get_madden_data`):** one row per `(team, season,
week)` with key columns `team, season, week` plus `madden_<slot|group>_<measure>` floats.

---

## Task 1: Weekly leakage-safe starters

**Files:**
- Create: `src/data/madden/starters.py`
- Test: `tests/data/madden/test_starters.py`

**Interfaces:**
- Produces:
  - `OUT_STATUSES = ('Out', 'Doubtful')`
  - `available_starters(depth: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame`
    — pure; given one team-week's depth chart + injuries, returns the rank-1-after-injury
    starter per `position`, columns `[gsis_id, position, depth_chart_rank]`.
  - `get_weekly_starters(years) -> pd.DataFrame` — columns `[season, week, team, gsis_id,
    position]`, REG season only.

- [ ] **Step 1: Failing test — next-man-up when the starter is Out**

```python
# tests/data/madden/test_starters.py
import unittest
from unittest import mock
import pandas as pd
from src.data.madden import starters


def _depth(rows):
    return pd.DataFrame(rows)


class TestAvailableStarters(unittest.TestCase):
    def _two_qbs(self):
        return _depth([
            {'club_code': 'KC', 'season': 2023, 'week': 1, 'position': 'QB',
             'depth_team': '1', 'gsis_id': 'STARTER'},
            {'club_code': 'KC', 'season': 2023, 'week': 1, 'position': 'QB',
             'depth_team': '2', 'gsis_id': 'BACKUP'},
        ])

    def test_healthy_starter_chosen(self):
        out = starters.available_starters(self._two_qbs(), pd.DataFrame(
            columns=['gsis_id', 'report_status']))
        qb = out[out['position'] == 'QB'].iloc[0]
        self.assertEqual(qb['gsis_id'], 'STARTER')

    def test_out_starter_promotes_backup(self):
        inj = pd.DataFrame([{'gsis_id': 'STARTER', 'report_status': 'Out'}])
        out = starters.available_starters(self._two_qbs(), inj)
        qb = out[out['position'] == 'QB'].iloc[0]
        self.assertEqual(qb['gsis_id'], 'BACKUP')

    def test_questionable_starter_still_starts(self):
        inj = pd.DataFrame([{'gsis_id': 'STARTER', 'report_status': 'Questionable'}])
        out = starters.available_starters(self._two_qbs(), inj)
        self.assertEqual(out[out['position'] == 'QB'].iloc[0]['gsis_id'], 'STARTER')
```

- [ ] **Step 2: Run — verify fails**

Run: `conda run -n nfl-predictions python -m pytest tests/data/madden/test_starters.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `starters.py`**

```python
# src/data/madden/starters.py
"""Leakage-safe weekly starters: depth-chart rank-1 minus Out/Doubtful players,
promoting the next available player at each position. All inputs are pre-kickoff."""
import nfl_data_py as nfl
import pandas as pd
from src.data.transformations import TEAM_ABBR_MAPPINGS

OUT_STATUSES = ('Out', 'Doubtful')


def available_starters(depth, injuries):
    """One team-week. Return the highest-available player per position.

    :param depth: depth rows with [position, depth_team, gsis_id].
    :param injuries: rows with [gsis_id, report_status].
    :returns: [gsis_id, position, depth_chart_rank] one row per position.
    """
    unavailable = set(
        injuries.loc[injuries['report_status'].isin(OUT_STATUSES), 'gsis_id'])
    d = depth.copy()
    d['rank'] = pd.to_numeric(d['depth_team'], errors='coerce')
    d = d[~d['gsis_id'].isin(unavailable)]
    d = d.sort_values(['position', 'rank'])
    chosen = d.groupby('position', as_index=False).first()
    return chosen[['gsis_id', 'position', 'rank']].rename(
        columns={'rank': 'depth_chart_rank'})


def get_weekly_starters(years):
    depth = nfl.import_depth_charts(years)
    depth = depth[depth['game_type'] == 'REG'].copy()
    depth['club_code'] = depth['club_code'].replace(TEAM_ABBR_MAPPINGS)
    injuries = nfl.import_injuries(years)
    injuries['team'] = injuries['team'].replace(TEAM_ABBR_MAPPINGS)

    frames = []
    keys = ['season', 'week', 'club_code']
    for (season, week, club), grp in depth.groupby(keys):
        inj = injuries[(injuries['season'] == season) & (injuries['week'] == week)
                       & (injuries['team'] == club)]
        chosen = available_starters(grp, inj)
        chosen['season'], chosen['week'], chosen['team'] = season, week, club
        frames.append(chosen)
    out = pd.concat(frames, ignore_index=True)
    return out[['season', 'week', 'team', 'gsis_id', 'position']]
```

- [ ] **Step 4: Run — verify pass**, then add a live smoke test:

```python
# append to tests/data/madden/test_starters.py
class TestLiveStarters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = starters.get_weekly_starters([2023])

    def test_one_qb_per_team_week(self):
        qbs = cls = self.df[self.df['position'] == 'QB']
        counts = qbs.groupby(['team', 'week']).size()
        self.assertTrue((counts == 1).all())

    def test_has_keys(self):
        for col in ['season', 'week', 'team', 'gsis_id', 'position']:
            self.assertIn(col, self.df.columns)
```

Run: `conda run -n nfl-predictions python -m pytest tests/data/madden/test_starters.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/madden/starters.py tests/data/madden/test_starters.py
git commit -m "feat(madden): leakage-safe weekly starters from depth charts + injuries"
```

---

## Task 2: Slot & group overalls (team-week)

**Files:**
- Create: `src/data/madden/features.py`
- Test: `tests/data/madden/test_features.py`

**Interfaces:**
- Consumes: starters frame (Task 1) + Plan-1 player-season frame (`gsis_id, overall,
  position, role, side`).
- Produces:
  - `GRANULAR_SLOTS`, `GROUP_ROLES` constants.
  - `build_team_week_overalls(starters: pd.DataFrame, players: pd.DataFrame) ->
    pd.DataFrame` — one row per `(team, season, week)` with `madden_<slot>_ovr` and
    `madden_<group>_ovr` columns. Slot assignment uses the Madden `position`/`side`
    of each starting player; WR1/2/3 and group means rank by `overall` desc.

- [ ] **Step 1: Failing test**

```python
# tests/data/madden/test_features.py
import unittest
import pandas as pd
from src.data.madden import features


class TestBuildTeamWeekOveralls(unittest.TestCase):
    def _starters(self):
        # one team-week: QB, two tackles, three WRs, a left edge
        return pd.DataFrame([
            {'season': 2023, 'week': 1, 'team': 'KC', 'gsis_id': g, 'position': p}
            for g, p in [('QB1', 'QB'), ('T1', 'T'), ('T2', 'T'),
                         ('W1', 'WR'), ('W2', 'WR'), ('W3', 'WR'), ('E1', 'OLB')]
        ])

    def _players(self):
        return pd.DataFrame([
            {'gsis_id': 'QB1', 'overall': 96, 'position': 'QB', 'role': 'qb', 'side': 'none'},
            {'gsis_id': 'T1', 'overall': 90, 'position': 'LT', 'role': 'exterior_ol', 'side': 'left'},
            {'gsis_id': 'T2', 'overall': 78, 'position': 'RT', 'role': 'exterior_ol', 'side': 'right'},
            {'gsis_id': 'W1', 'overall': 99, 'position': 'WR', 'role': 'receiver', 'side': 'none'},
            {'gsis_id': 'W2', 'overall': 84, 'position': 'WR', 'role': 'receiver', 'side': 'none'},
            {'gsis_id': 'W3', 'overall': 72, 'position': 'WR', 'role': 'receiver', 'side': 'none'},
            {'gsis_id': 'E1', 'overall': 88, 'position': 'LOLB', 'role': 'edge', 'side': 'left'},
        ])

    def test_granular_slots_and_groups(self):
        out = features.build_team_week_overalls(self._starters(), self._players())
        row = out.iloc[0]
        self.assertEqual(row['madden_qb_ovr'], 96)
        self.assertEqual(row['madden_lt_ovr'], 90)    # side from Madden position
        self.assertEqual(row['madden_rt_ovr'], 78)
        self.assertEqual(row['madden_wr1_ovr'], 99)   # ranked by overall desc
        self.assertEqual(row['madden_wr2_ovr'], 84)
        self.assertEqual(row['madden_wr3_ovr'], 72)
        self.assertEqual(row['madden_edge_left_ovr'], 88)
        self.assertEqual(row['madden_receivers_ovr'], (99 + 84 + 72) / 3)
        self.assertEqual(row['madden_exterior_ol_ovr'], (90 + 78) / 2)
```

- [ ] **Step 2: Run — fails** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the overall builder in `features.py`**

```python
# src/data/madden/features.py
"""Build team-week Madden overall columns from weekly starters + the player-season
table, then within-season / season-over-season diffs and post-reframe matchup deltas."""
import pandas as pd

# granular single-player slots keyed by (madden_position predicate)
_OL_SIDE = {'left': 'lt', 'right': 'rt'}
GROUP_ROLES = {
    'backfield': 'backfield', 'receivers': 'receiver', 'tight_end': 'tight_end',
    'interior_ol': 'interior_ol', 'exterior_ol': 'exterior_ol', 'edge': 'edge',
    'interior_dl': 'interior_dl', 'linebacker': 'off_ball_lb',
    'cornerback': 'cornerback', 'safety': 'safety',
}


def _slot_of(player):
    """Map a starting player (joined to Madden) to a granular slot name, or None."""
    pos, side, role = player['position'], player['side'], player['role']
    if role == 'qb':
        return 'qb'
    if role == 'backfield':
        return 'rb'
    if role == 'tight_end':
        return 'te'
    if pos in ('LG', 'RG', 'C'):
        return pos.lower()
    if role == 'exterior_ol':
        return _OL_SIDE.get(side)            # lt / rt
    if role == 'edge':
        return 'edge_left' if side == 'left' else 'edge_right' if side == 'right' else None
    return None  # WRs handled by ranking; interior_dl/lb/db only feed groups


def build_team_week_overalls(starters, players):
    joined = starters.merge(
        players[['gsis_id', 'overall', 'position', 'role', 'side']],
        on='gsis_id', how='left')
    joined = joined[joined['overall'].notna()]
    records = []
    for (season, week, team), grp in joined.groupby(['season', 'week', 'team']):
        rec = {'season': season, 'week': week, 'team': team}
        # single-player slots: highest overall among players mapping to that slot
        grp = grp.assign(_slot=[_slot_of(r) for _, r in grp.iterrows()])
        for slot, sub in grp.dropna(subset=['_slot']).groupby('_slot'):
            rec[f'madden_{slot}_ovr'] = sub['overall'].max()
        # WR1/2/3 by overall desc among receiver-role starters
        wrs = grp[grp['role'] == 'receiver'].sort_values('overall', ascending=False)
        for i, ovr in enumerate(wrs['overall'].head(3).tolist(), start=1):
            rec[f'madden_wr{i}_ovr'] = ovr
        # group means by role
        for group, role in GROUP_ROLES.items():
            members = grp[grp['role'] == role]
            if len(members):
                rec[f'madden_{group}_ovr'] = members['overall'].mean()
        records.append(rec)
    return pd.DataFrame(records)
```

- [ ] **Step 4: Run — verify pass.**

Run: `conda run -n nfl-predictions python -m pytest tests/data/madden/test_features.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/madden/features.py tests/data/madden/test_features.py
git commit -m "feat(madden): team-week slot and group overall columns"
```

---

## Task 3: Within-season and season-over-season diffs

**Files:**
- Modify: `src/data/madden/features.py`
- Test: `tests/data/madden/test_features.py`

**Interfaces:**
- Produces:
  - `add_within_season_diffs(df: pd.DataFrame) -> pd.DataFrame` — for every `madden_*_ovr`
    column, add `_diff_prev` (vs the team's previous game) and `_diff_4g` (vs 4 games
    back), ordered by `week` within `(team, season)`.
  - `add_prev_season_diff(df: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame` — add
    `_diff_prev_season` per `_ovr` column, where `prev` is the prior season's same frame.

- [ ] **Step 1: Failing test**

```python
# append to tests/data/madden/test_features.py
class TestWithinSeasonDiffs(unittest.TestCase):
    def test_prev_and_4g_diffs(self):
        df = pd.DataFrame([
            {'season': 2023, 'week': w, 'team': 'KC', 'madden_qb_ovr': ovr}
            for w, ovr in [(1, 96), (2, 96), (3, 70), (4, 70), (5, 96)]
        ])
        out = features.add_within_season_diffs(df).sort_values('week')
        diffs = out['madden_qb_ovr_diff_prev'].tolist()
        self.assertTrue(pd.isna(diffs[0]))            # first game -> NA
        self.assertEqual(diffs[2], -26)               # 70 - 96 (starter went down)
        self.assertEqual(diffs[4], 26)                # 96 - 70 (starter returned)
        fourg = out['madden_qb_ovr_diff_4g'].tolist()
        self.assertTrue(pd.isna(fourg[3]))            # game 4 -> still NA
        self.assertEqual(fourg[4], 0)                 # week5 96 vs week1 96


class TestPrevSeasonDiff(unittest.TestCase):
    def test_prev_season_diff(self):
        cur = pd.DataFrame([{'season': 2024, 'week': 1, 'team': 'KC', 'madden_qb_ovr': 99}])
        prev = pd.DataFrame([{'season': 2023, 'week': 1, 'team': 'KC', 'madden_qb_ovr': 96}])
        out = features.add_prev_season_diff(cur, prev)
        self.assertEqual(out.iloc[0]['madden_qb_ovr_diff_prev_season'], 3)
```

- [ ] **Step 2: Run — fails** (`AttributeError: add_within_season_diffs`).

- [ ] **Step 3: Implement diffs in `features.py`**

```python
# append to src/data/madden/features.py
def _ovr_columns(df):
    return [c for c in df.columns if c.startswith('madden_') and c.endswith('_ovr')]


def add_within_season_diffs(df):
    out = df.sort_values(['team', 'season', 'week']).copy()
    grp = out.groupby(['team', 'season'])
    for col in _ovr_columns(out):
        out[f'{col}_diff_prev'] = out[col] - grp[col].shift(1)
        out[f'{col}_diff_4g'] = out[col] - grp[col].shift(4)
    return out


def add_prev_season_diff(df, prev):
    """Per-team season-over-season launch diff. `prev` is the prior season frame;
    a team's value is constant within a season, so any prior-season row suffices."""
    out = df.copy()
    ovr_cols = _ovr_columns(out)
    prev_by_team = (prev.sort_values('week').groupby('team')[ovr_cols].first())
    for col in ovr_cols:
        mapped = out['team'].map(prev_by_team[col])
        out[f'{col}_diff_prev_season'] = out[col] - mapped
    return out
```

- [ ] **Step 4: Run — verify pass.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/madden/features.py tests/data/madden/test_features.py
git commit -m "feat(madden): games-played and season-over-season overall diffs"
```

---

## Task 4: Orchestrator `get_madden_data(years)`

**Files:**
- Create: `src/data/madden/collect.py`
- Test: `tests/data/madden/test_collect.py`

**Interfaces:**
- Consumes: Plan 1 (`ingest.load_madden_season`, `ids.attach_gsis_id`,
  `roles.classify_roles`) + Tasks 1–3.
- Produces: `get_madden_data(years) -> pd.DataFrame` — one row per `(team, season,
  week)` with `madden_*` columns (overalls + all three diffs).

- [ ] **Step 1: Failing test (mock the season loads + starters)**

```python
# tests/data/madden/test_collect.py
import unittest
from unittest import mock
import pandas as pd
from src.data.madden import collect


class TestGetMaddenData(unittest.TestCase):
    def test_assembles_team_week_with_madden_columns(self):
        players_2023 = pd.DataFrame([
            {'season': 2023, 'full_name': 'A', 'team': 'KC', 'position': 'QB',
             'overall': 96, 'weight': 220, 'power_moves': 0, 'finesse_moves': 0,
             'gsis_id': 'QB1', 'role': 'qb', 'side': 'none'}])
        starters = pd.DataFrame([
            {'season': 2023, 'week': 1, 'team': 'KC', 'gsis_id': 'QB1', 'position': 'QB'}])
        with mock.patch.object(collect, '_player_season', return_value=players_2023), \
             mock.patch.object(collect.starters_mod, 'get_weekly_starters',
                               return_value=starters):
            out = collect.get_madden_data([2023])
        self.assertEqual(set(['team', 'season', 'week']).issubset(out.columns), True)
        self.assertIn('madden_qb_ovr', out.columns)
        self.assertIn('madden_qb_ovr_diff_prev', out.columns)
        self.assertEqual(out.iloc[0]['madden_qb_ovr'], 96)
        # every feature column carries the 'madden' token (partition.py contract)
        feat = [c for c in out.columns if c not in ('team', 'season', 'week')]
        self.assertTrue(all('madden' in c for c in feat))
```

- [ ] **Step 2: Run — fails.**

- [ ] **Step 3: Implement `collect.py`**

```python
# src/data/madden/collect.py
"""Public entry point: assemble team-week Madden features for the given years."""
import pandas as pd
from src.data.madden import ingest, ids, roles, features
from src.data.madden import starters as starters_mod


def _player_season(season):
    """Plan-1 pipeline for one season -> classified player-season frame."""
    df = ingest.load_madden_season(season)
    df = ids.attach_gsis_id(df, season)
    return roles.classify_roles(df)


def get_madden_data(years):
    years = [years] if isinstance(years, int) else list(years)
    players = {s: _player_season(s) for s in years}
    starters = starters_mod.get_weekly_starters(years)

    per_season = []
    for season in years:
        s_rows = starters[starters['season'] == season]
        overalls = features.build_team_week_overalls(s_rows, players[season])
        overalls = features.add_within_season_diffs(overalls)
        if (season - 1) in players:
            prev = features.build_team_week_overalls(
                starters[starters['season'] == season - 1], players[season - 1])
            overalls = features.add_prev_season_diff(overalls, prev)
        per_season.append(overalls)
    return pd.concat(per_season, ignore_index=True)
```

- [ ] **Step 4: Run — verify pass.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/madden/collect.py tests/data/madden/test_collect.py
git commit -m "feat(madden): get_madden_data team-week orchestrator"
```

---

## Task 5: Integrate into `collect_all.py` (double-merge + matchup deltas)

**Files:**
- Modify: `src/data/collect_all.py`
- Modify: `src/data/madden/features.py` (add `add_madden_matchup_columns`)
- Test: `tests/data/test_collect_all.py` (extend)

**Interfaces:**
- Consumes: `madden_collect.get_madden_data(years)`; the post-shift `combined_data`
  carrying `target_team, opp_team, season, week`.
- Produces: `combined_data` gains `target_madden_*`, `opp_madden_*`, and
  `madden_matchup_*` columns.

- [ ] **Step 1: Implement `add_madden_matchup_columns` (TDD first)**

Test (append to `tests/data/madden/test_features.py`):

```python
class TestMatchupColumns(unittest.TestCase):
    def test_matchup_deltas(self):
        df = pd.DataFrame([{
            'target_madden_exterior_ol_ovr': 90, 'opp_madden_edge_ovr': 80,
            'target_madden_interior_ol_ovr': 85, 'opp_madden_interior_dl_ovr': 75,
            'target_madden_receivers_ovr': 88, 'opp_madden_cornerback_ovr': 70,
            'opp_madden_safety_ovr': 72, 'opp_madden_exterior_ol_ovr': 60,
            'target_madden_edge_ovr': 90}])
        out = features.add_madden_matchup_columns(df).iloc[0]
        self.assertEqual(out['madden_matchup_pass_pro'], 10)        # 90-80
        self.assertEqual(out['madden_matchup_interior'], 10)        # 85-75
        self.assertEqual(out['madden_matchup_skill_cover'], 88 - 71)  # 88-(70+72)/2
        self.assertEqual(out['madden_matchup_pass_rush'], -30)      # 60-90
```

Implementation (append to `src/data/madden/features.py`):

```python
def _safe(df, col):
    return df[col] if col in df.columns else pd.Series([pd.NA] * len(df), index=df.index)


def add_madden_matchup_columns(df):
    """Directional trench/coverage talent deltas, computed AFTER the target/opp merge."""
    out = df.copy()
    out['madden_matchup_pass_pro'] = (
        _safe(out, 'target_madden_exterior_ol_ovr') - _safe(out, 'opp_madden_edge_ovr'))
    out['madden_matchup_interior'] = (
        _safe(out, 'target_madden_interior_ol_ovr')
        - _safe(out, 'opp_madden_interior_dl_ovr'))
    out['madden_matchup_skill_cover'] = (
        _safe(out, 'target_madden_receivers_ovr')
        - (_safe(out, 'opp_madden_cornerback_ovr')
           + _safe(out, 'opp_madden_safety_ovr')) / 2)
    out['madden_matchup_pass_rush'] = (
        _safe(out, 'opp_madden_exterior_ol_ovr') - _safe(out, 'target_madden_edge_ovr'))
    return out
```

- [ ] **Step 2: Run the matchup test — verify pass.**

- [ ] **Step 3: Wire into `collect_all.py`** — add import near the other collector
imports, and append this AFTER `_shift_data(...)` (line ~85), before `return`:

```python
# src/data/collect_all.py  (import block)
from src.data.madden import collect as madden_collect
from src.data.madden import features as madden_features

# ... inside get_schedule_and_weekly_data, after `combined_data = _shift_data(combined_data)`:
madden_tw = madden_collect.get_madden_data(years)
_madden_cols = [c for c in madden_tw.columns if c not in ('team', 'season', 'week')]

target_madden = madden_tw.rename(
    columns={'team': 'target_team',
             **{c: f'target_{c}' for c in _madden_cols}})
opp_madden = madden_tw.rename(
    columns={'team': 'opp_team',
             **{c: f'opp_{c}' for c in _madden_cols}})

combined_data = combined_data.merge(
    target_madden, on=['target_team', 'season', 'week'], how='left', validate='many_to_one')
combined_data = combined_data.merge(
    opp_madden, on=['opp_team', 'season', 'week'], how='left', validate='many_to_one')
combined_data = madden_features.add_madden_matchup_columns(combined_data)
```

Note: merged AFTER the shift so Madden week-N features align to the predicted game week
(they are pre-game-N info, not cumulative-through-N stats). `validate='many_to_one'`
because each game row maps to one team-week Madden row.

- [ ] **Step 4: Extend the integration test**

```python
# append to tests/data/test_collect_all.py (inside the existing test class, new method)
    def test_madden_columns_present_and_paired(self):
        cols = self.all_data.columns
        self.assertIn('target_madden_qb_ovr', cols)
        self.assertIn('opp_madden_qb_ovr', cols)
        self.assertIn('madden_matchup_pass_pro', cols)
        # target QB overall should be a plausible Madden value where present
        vals = self.all_data['target_madden_qb_ovr'].dropna()
        self.assertTrue(vals.between(40, 99).all())
```

- [ ] **Step 5: Run + commit**

```bash
conda run -n nfl-predictions python -m pytest tests/data/test_collect_all.py -v
git add src/data/collect_all.py src/data/madden/features.py tests/data/
git commit -m "feat(madden): integrate team-game features via post-shift double-merge"
```

---

## Task 6: Register the `madden_ratings` feature group

**Files:**
- Modify: `data_science_utilities/feature_groups/partition.py`
- Test: `tests/data_science_utilities/test_feature_group_partition.py` (extend)

**Interfaces:**
- Produces: `content_family('…madden…')` → `'madden_ratings'`; `'madden_ratings'` added
  to `CONTENT_FAMILIES`.

- [ ] **Step 1: Failing test**

```python
# append to tests/data_science_utilities/test_feature_group_partition.py
    def test_madden_columns_classify_as_madden_ratings(self):
        for col in ['target_madden_qb_ovr', 'opp_madden_edge_ovr',
                    'target_madden_lt_ovr_diff_prev', 'madden_matchup_pass_pro',
                    'opp_madden_safety_ovr_diff_prev_season']:
            self.assertEqual(content_family(col), 'madden_ratings')
```

- [ ] **Step 2: Run — fails** (`ValueError: unclassified feature column`).

- [ ] **Step 3: Implement** — in `partition.py`:

Add `'madden_ratings'` to the `CONTENT_FAMILIES` tuple (append at end):

```python
CONTENT_FAMILIES = (
    'context_rest', 'market', 'schedule_points', 'box_score',
    'pbp_phase1', 'pbp_phase2_directional', 'pbp_phase3',
    'situational_playcall', 'snap_share', 'madden_ratings',
)
```

Add this check at the **top** of `content_family`, before the `CONTEXT_REST_COLUMNS`
check (madden columns never collide with other families, and this avoids the
`PERSPECTIVE_PREFIXES` path raising):

```python
def content_family(column):
    if 'madden' in column:
        return 'madden_ratings'
    if column in CONTEXT_REST_COLUMNS:
        ...
```

- [ ] **Step 4: Run the full partition suite** (the exhaustive "every column classifies"
test must still pass with the new madden columns in `xgb_features_list.csv` after Task 7;
for now the unit test passes):

Run: `conda run -n nfl-predictions python -m pytest tests/data_science_utilities/test_feature_group_partition.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data_science_utilities/feature_groups/partition.py tests/data_science_utilities/test_feature_group_partition.py
git commit -m "feat(feature-groups): add madden_ratings content family"
```

---

## Task 7: Rebuild dataset + run base-vs-madden ablation

**Files:**
- Run: `scripts/data_assembly/predict_game_winner/schedule_and_weekly.py` (no edit
  expected — madden columns flow through `get_schedule_and_weekly_data`; the feature
  list is regenerated automatically).
- Use: the `feature_groups/` ablation harness
  (`notebooks/.../cross_validation/run_group_ablation.py`).

- [ ] **Step 1: Rebuild the input parquet**

Run: `conda run -n nfl-predictions python scripts/data_assembly/predict_game_winner/schedule_and_weekly.py`
Expected: writes `data/predict_games/input_data/schedule_and_weekly.parquet` and a
regenerated `xgb_features_list.csv` that now contains `*_madden_*` columns.

- [ ] **Step 2: Verify the exhaustive partition test passes on the new feature list**

Run: `conda run -n nfl-predictions python -m pytest tests/data_science_utilities/test_feature_group_partition.py -v`
Expected: PASS — every column (including the new madden columns) classifies with zero
unknowns. If a madden column raises `unclassified`, its name is missing the `madden`
token — fix the column construction in Task 2/3, not the classifier.

- [ ] **Step 3: Sanity-check coverage of the new columns**

```bash
conda run -n nfl-predictions python - <<'PY'
import pandas as pd
df = pd.read_parquet('data/predict_games/input_data/schedule_and_weekly.parquet')
mc = [c for c in df.columns if 'madden' in c]
print('madden columns:', len(mc))
sub = df[['season'] + [c for c in mc if c.endswith('_ovr')]]
print(sub.groupby('season').apply(lambda g: g.drop(columns='season').notna().mean().mean()))
PY
```
Expected: per-season non-null fraction for madden `_ovr` columns is high (≳0.8) in the
target window; if early seasons are sparse, set the training start year accordingly
(spec §8).

- [ ] **Step 4: Run the ablation: base vs. base + madden_ratings**

Run: `conda run -n nfl-predictions python notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/run_group_ablation.py`
(Configure it to compare the full feature set with and without the `madden_ratings`
group, scored with the same AUROC + hold-out metric as README Runs 21–24.)
Expected: a recorded metric delta for adding `madden_ratings`.

- [ ] **Step 5: Record the result in the README runs log + commit**

```bash
git add README.md data/predict_games/group_ablation/  # or wherever the ablation writes
git commit -m "experiment(madden): record base vs base+madden_ratings ablation (Run NN)"
```

---

## Self-Review

**Spec coverage (design doc §4–§9):**
- §4 stages 3–5 (starters, feature assembly, merge) → Tasks 1, 2–4, 5. ✓
- §6 leakage-safe starters (depth chart + injuries, next-man-up) → Task 1. ✓
- §7 A granular (incl. side-aware edges), B groups, C matchup deltas → Tasks 2 & 5. ✓
- §7 diff semantics (games-played prev / 4g) + §7.3 season-over-season → Task 3. ✓
- §9 single `madden_ratings` group + base-vs-madden ablation → Tasks 6 & 7. ✓
- §8 coverage/missing → Task 7 Step 3 (per-season coverage check; NA-native). ✓
- Leakage alignment (merge after shift) → Task 5 (explicit, with rationale). ✓

**Placeholder scan:** none — each step has runnable test/impl/command. Task 7 Step 4
references the existing ablation runner rather than re-specifying it (it's pre-existing
infrastructure, not new code).

**Type consistency:** `get_weekly_starters` columns (`season,week,team,gsis_id,position`)
feed `build_team_week_overalls(starters, players)`; its `madden_*_ovr` outputs feed
`_ovr_columns()` in both diff functions; `get_madden_data` returns those columns; Task 5
renames them to `target_/opp_` and `add_madden_matchup_columns` reads those exact names.
The `'madden'` substring invariant (Task 4 test) guarantees Task 6 classification.

**Known follow-ups (not gaps):**
- Slot reconciliation assumes a starting player's Madden `position` carries the side
  (LT/RT/LE/RE). A starter with no Madden match → that slot is NA (acceptable; §8).
- If `validate='many_to_one'` raises in Task 5, the Madden frame has duplicate
  `(team,season,week)` rows — dedupe in `get_madden_data` before returning.
