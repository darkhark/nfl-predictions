# Madden Launch-Ratings Data Layer Implementation Plan (Phase 0)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make 2025+ season-launch Madden ratings flow through the existing `src/data/madden/` pipeline and contribute real model signal (2025 `_ovr` coverage 0% → >30%), without changing any ≤2024 behavior.

**Architecture:** Additive. A new launch-`players.json` ingest adapter and a new 2025 depth-chart normalizer feed the *existing* `ids → roles → features` chain through the unchanged `ingest.OUTPUT_COLUMNS` contract. `collect.py` routes by season (≤2024 = theedgepredictor + processed/ gsis bridge; ≥2025 = madden-tools CDN JSON + nflverse-rosters gsis bridge). `starters.py` dispatches the depth-chart load by season and reuses the unchanged `available_starters` core. A coverage-report script proves the result.

**Tech Stack:** Python 3.11, pandas, `nfl_data_py` (nflverse), `requests` (CDN GET), `pytest`/`unittest`. Conda env `nfl-predictions`.

## Global Constraints

- **Additive only:** no ≤2024 code path changes behavior. Every new path is season-gated `>= 2025`.
- **Contract:** every Madden source must emit exactly `ingest.OUTPUT_COLUMNS = ['season','full_name','team','position','overall','weight','power_moves','finesse_moves']`. Downstream `ids/roles/features` are untouched.
- **Feature family unchanged:** the `madden_ratings` family (any column containing `madden`) and its **188 columns** are unchanged — only their **2025 values** go NaN → populated. Do not add, rename, or drop columns.
- **Leakage / meta columns NEVER fed as features:** `game_id, season, season_type, opp_team, opp_score, target_team, target_score, h_win`.
- **Leakage conventions (unchanged):** launch ratings are season-start-known → valid every in-season week; the one-week shift is already baked into the parquet (do NOT re-apply); `_ovr` is already z-scored within season in `features.zscore_overalls_within_season` (do NOT double); REG-only.
- **2025 starter parity:** map the new granular `pos_abb` to the **pre-2025 coarse `position` vocabulary** so `available_starters` selects one starter per coarse slot — matching the all-old-schema training distribution. Do NOT extract the fuller granular lineup (it would make 2025 test features out-of-distribution vs training).
- **Run-log:** Phase 0 is a data phase — a short data-quality note only. The numbered **Run 28 is reserved for Phase 1**.
- **Tests:** `unittest` TestCases with inline-built DataFrames/dicts (house style); no live network in unit tests (mock `nfl_data_py` and CDN fetches). Run with `conda run --no-capture-output -n nfl-predictions python -m pytest <path> -v`.

---

## File Structure

- **Create** `src/data/madden/launch.py` — madden-tools launch `players.json`/`teams.json` → `OUTPUT_COLUMNS`; launch-iteration selection; CDN loader.
- **Create** `src/data/madden/depth_2025.py` — normalize the nflverse 2025+ depth schema to the legacy week-keyed contract (leakage-safe pre-kickoff snapshot + coarse-position mapping).
- **Modify** `src/data/madden/starters.py` — dispatch depth load by season; reuse `available_starters`.
- **Modify** `src/data/madden/ids.py` — add `attach_gsis_id_from_rosters`.
- **Modify** `src/data/madden/collect.py` — season-routing of source + gsis bridge; `season_to_game_version`.
- **Create** `scripts/experiments/madden_coverage_report.py` — per-season gsis-match + `_ovr` coverage report (the data-quality deliverable).
- **Create** `tests/data/madden/test_launch.py`, `tests/data/madden/test_depth_2025.py`.
- **Modify** `tests/data/madden/test_starters.py`, `test_ids.py`, `test_collect.py` — extend.
- **Modify** `docs/madden-features-and-vlm.md` (§3c/§3e coverage table) and `README.md` (run-log data-quality note).

---

## Task 1: Launch adapter — pure parse + launch-iteration selection

**Files:**
- Create: `src/data/madden/launch.py`
- Test: `tests/data/madden/test_launch.py`

**Interfaces:**
- Consumes: `ingest.OUTPUT_COLUMNS`, `src.data.transformations.TEAM_ABBR_MAPPINGS`.
- Produces: `parse_launch_ratings(players: list[dict], teams: list[dict], season: int) -> DataFrame` (exactly `OUTPUT_COLUMNS`); `select_launch_iteration(iterations: list[dict]) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/madden/test_launch.py
import unittest
import pandas as pd
from src.data.madden import launch
from src.data.madden.ingest import OUTPUT_COLUMNS


def _players():
    return [
        {'first_name': 'Lamar', 'last_name': 'Jackson', 'position': 'QB',
         'team_id': 26, 'rating_overall': 94, 'weight': 215,
         'rating_power_moves': 40, 'rating_finesse_moves': 35},
        {'first_name': 'Kyle', 'last_name': 'Juszczyk', 'position': 'FB',
         'team_id': 25, 'rating_overall': 88, 'weight': 235,
         'rating_power_moves': 30, 'rating_finesse_moves': 35},
    ]


def _teams():
    return [
        {'id': 26, 'name': 'Ravens', 'acronym': 'BAL'},
        {'id': 25, 'name': '49ers', 'acronym': 'SF'},
    ]


class TestParseLaunchRatings(unittest.TestCase):
    def test_maps_to_output_columns(self):
        out = launch.parse_launch_ratings(_players(), _teams(), 2025)
        self.assertEqual(list(out.columns), OUTPUT_COLUMNS)
        lamar = out[out['full_name'] == 'Lamar Jackson'].iloc[0]
        self.assertEqual(lamar['team'], 'BAL')
        self.assertEqual(lamar['position'], 'QB')
        self.assertEqual(lamar['overall'], 94)
        self.assertEqual(lamar['power_moves'], 40)
        self.assertEqual(lamar['finesse_moves'], 35)
        self.assertEqual(lamar['season'], 2025)

    def test_empty_players_returns_empty_contract_frame(self):
        out = launch.parse_launch_ratings([], _teams(), 2025)
        self.assertEqual(list(out.columns), OUTPUT_COLUMNS)
        self.assertEqual(len(out), 0)

    def test_unknown_team_id_yields_na_team(self):
        players = [{'first_name': 'A', 'last_name': 'B', 'position': 'WR',
                    'team_id': 999, 'rating_overall': 70, 'weight': 190,
                    'rating_power_moves': 10, 'rating_finesse_moves': 60}]
        out = launch.parse_launch_ratings(players, _teams(), 2025)
        self.assertTrue(pd.isna(out.iloc[0]['team']))


class TestSelectLaunchIteration(unittest.TestCase):
    def _iterations(self):
        return [
            {'id': 0, 'label': 'Launch', 'release_date': '2025-08-14', 'active': True},
            {'id': 1, 'label': 'Week 1', 'release_date': '2025-09-04', 'active': True},
        ]

    def test_selects_launch(self):
        self.assertEqual(launch.select_launch_iteration(self._iterations())['id'], 0)

    def test_raises_when_no_launch(self):
        with self.assertRaises(ValueError):
            launch.select_launch_iteration([{'id': 1, 'label': 'Week 1'}])

    def test_raises_when_multiple_launch(self):
        dup = [{'id': 0, 'label': 'Launch'}, {'id': 5, 'label': 'launch'}]
        with self.assertRaises(ValueError):
            launch.select_launch_iteration(dup)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data/madden/test_launch.py -v`
Expected: FAIL — `ModuleNotFoundError`/`AttributeError: module 'src.data.madden.launch' has no attribute 'parse_launch_ratings'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/madden/launch.py
"""Adapt the madden-tools per-iteration `players.json` (+ `teams.json`) launch
snapshot into the shared `ingest.OUTPUT_COLUMNS` contract, so 2025+ launch ratings
flow through the same ids/roles/features pipeline as the historical theedgepredictor
source. Launch is the iteration whose `iterations.json` label == "Launch"."""
import logging
import pandas as pd
from src.data.madden.ingest import OUTPUT_COLUMNS
from src.data.transformations import TEAM_ABBR_MAPPINGS

logger = logging.getLogger(__name__)


def select_launch_iteration(iterations):
    """Return the single launch iteration entry (label == 'Launch', case-insensitive)."""
    launch = [it for it in iterations
              if str(it.get('label', '')).strip().lower() == 'launch']
    if len(launch) != 1:
        raise ValueError(f'expected exactly one Launch iteration, found {len(launch)}')
    return launch[0]


def _team_id_to_abbr(teams):
    """madden-tools team_id -> standard NFL abbreviation via teams.json `acronym`."""
    return {t['id']: str(t['acronym']).strip() for t in teams}


def parse_launch_ratings(players, teams, season):
    """Map a launch `players.json` (+ `teams.json`) to ingest.OUTPUT_COLUMNS.

    Empty `players` -> empty frame with OUTPUT_COLUMNS (mirrors ingest's resilience)."""
    if not players:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    id_to_abbr = _team_id_to_abbr(teams)
    rows = []
    for p in players:
        first = str(p.get('first_name', '') or '').strip()
        last = str(p.get('last_name', '') or '').strip()
        rows.append({
            'season': season,
            'full_name': (first + ' ' + last).strip(),
            'team': id_to_abbr.get(p.get('team_id')),
            'position': p.get('position'),
            'overall': p.get('rating_overall'),
            'weight': p.get('weight'),
            'power_moves': p.get('rating_power_moves'),
            'finesse_moves': p.get('rating_finesse_moves'),
        })
    out = pd.DataFrame(rows)
    out['team'] = out['team'].replace(TEAM_ABBR_MAPPINGS)
    return out[OUTPUT_COLUMNS]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data/madden/test_launch.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data/madden/launch.py tests/data/madden/test_launch.py
git commit -m "feat(madden): launch players.json -> OUTPUT_COLUMNS adapter + iteration select"
```

---

## Task 2: Launch CDN loader

**Files:**
- Modify: `src/data/madden/launch.py`
- Test: `tests/data/madden/test_launch.py`

**Interfaces:**
- Consumes: `parse_launch_ratings`, `select_launch_iteration` (Task 1).
- Produces: `load_madden_launch(game_version: str, season: int, *, cdn_base: str | None = None) -> DataFrame`. Reads `{base}/{game_version}/json/iterations.json`, selects Launch, reads `.../json/iterations/{id}/players.json` and `.../teams.json`, returns `OUTPUT_COLUMNS`. Any failure (no base / network / parse) → empty `OUTPUT_COLUMNS` frame + warning. CDN base resolves from the `cdn_base` arg, else env `MADDEN_TOOLS_CDN_BASE`. Internal `_cdn_json(url) -> obj` is the single I/O seam (mock it in tests).

- [ ] **Step 1: Write the failing test** (append to `tests/data/madden/test_launch.py`)

```python
from unittest import mock


class TestLoadMaddenLaunch(unittest.TestCase):
    def _fake_cdn(self):
        store = {
            'https://cdn.test/madden-26/json/iterations.json': [
                {'id': 0, 'label': 'Launch', 'active': True},
                {'id': 1, 'label': 'Week 1', 'active': True},
            ],
            'https://cdn.test/madden-26/json/iterations/0/players.json': _players(),
            'https://cdn.test/madden-26/json/iterations/0/teams.json': _teams(),
        }
        return lambda url: store[url]

    def test_loads_launch_iteration(self):
        with mock.patch.object(launch, '_cdn_json', side_effect=self._fake_cdn()):
            out = launch.load_madden_launch('madden-26', 2025, cdn_base='https://cdn.test')
        self.assertEqual(set(out['full_name']), {'Lamar Jackson', 'Kyle Juszczyk'})
        self.assertEqual(out[out['full_name'] == 'Lamar Jackson'].iloc[0]['team'], 'BAL')

    def test_network_failure_returns_empty(self):
        with mock.patch.object(launch, '_cdn_json', side_effect=OSError('boom')):
            out = launch.load_madden_launch('madden-26', 2025, cdn_base='https://cdn.test')
        self.assertEqual(list(out.columns), OUTPUT_COLUMNS)
        self.assertEqual(len(out), 0)

    def test_missing_base_returns_empty(self):
        with mock.patch.dict('os.environ', {}, clear=True):
            out = launch.load_madden_launch('madden-26', 2025, cdn_base=None)
        self.assertEqual(len(out), 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data/madden/test_launch.py::TestLoadMaddenLaunch -v`
Expected: FAIL — `AttributeError: ... has no attribute 'load_madden_launch'` / `_cdn_json`.

- [ ] **Step 3: Write minimal implementation** (append to `src/data/madden/launch.py`)

```python
import os
import requests

# madden-tools DigitalOcean Spaces CDN. The roster JSON is public-read (the webapp
# fetches it client-side), so a plain GET works — no credentials for reads. Path
# layout mirrors json-generator output: {base}/{version}/json/iterations.json and
# {base}/{version}/json/iterations/{id}/{players,teams}.json
# (verified in madden-tools webapp GuideMetadataCache.ts / draft-genius CDNDataRepository.ts).
# Set MADDEN_TOOLS_CDN_BASE to the confirmed CDN domain (e.g. the value of the
# webapp's NEXT_PUBLIC_CDN_DOMAIN) before the real-data run (Task 7).


def _cdn_json(url):
    """GET a JSON document. The single network seam (patched in tests)."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def load_madden_launch(game_version, season, *, cdn_base=None):
    """Fetch + parse the launch-iteration players/teams JSON for `game_version`.

    Returns an empty OUTPUT_COLUMNS frame on any failure (so collect proceeds with
    NaN Madden features for the season)."""
    base = (cdn_base if cdn_base is not None
            else os.environ.get('MADDEN_TOOLS_CDN_BASE', '')).rstrip('/')
    if not base:
        logger.warning('season %s: MADDEN_TOOLS_CDN_BASE unset — skipping launch load', season)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    try:
        root = f'{base}/{game_version}/json'
        iterations = _cdn_json(f'{root}/iterations.json')
        launch_id = select_launch_iteration(iterations)['id']
        it_dir = f'{root}/iterations/{launch_id}'
        players = _cdn_json(f'{it_dir}/players.json')
        teams = _cdn_json(f'{it_dir}/teams.json')
    except Exception as exc:
        logger.warning('season %s: failed to load madden-tools launch (%s)', season, exc)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return parse_launch_ratings(players, teams, season)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data/madden/test_launch.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data/madden/launch.py tests/data/madden/test_launch.py
git commit -m "feat(madden): CDN loader for launch-iteration players/teams JSON"
```

---

## Task 3: 2025+ depth-chart normalizer

**Files:**
- Create: `src/data/madden/depth_2025.py`
- Test: `tests/data/madden/test_depth_2025.py`

**Interfaces:**
- Consumes: nothing project-internal (pure transform of two DataFrames).
- Produces: `normalize_2025_depth(depth_raw: DataFrame, schedule: DataFrame, season: int) -> DataFrame` with columns `['season','week','club_code','game_type','position','depth_team','gsis_id']` — one row per (team-week, snapshot player), using the **latest snapshot strictly before each team's game day** (leakage-safe) and the **coarse-position** map. Also exports `POS_ABB_TO_COARSE: dict`.
- `depth_raw` columns (nflverse 2025+): `dt, team, gsis_id, pos_abb, pos_rank, ...`. `schedule` columns: `season, week, game_type, gameday, gametime, home_team, away_team`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/madden/test_depth_2025.py
import unittest
import pandas as pd
from src.data.madden import depth_2025


def _schedule():
    # BAL plays week 1 on 2025-09-07, week 2 on 2025-09-14.
    return pd.DataFrame([
        {'season': 2025, 'week': 1, 'game_type': 'REG', 'gameday': '2025-09-07',
         'gametime': '13:00', 'home_team': 'BAL', 'away_team': 'CLE'},
        {'season': 2025, 'week': 2, 'game_type': 'REG', 'gameday': '2025-09-14',
         'gametime': '13:00', 'home_team': 'CIN', 'away_team': 'BAL'},
        {'season': 2025, 'week': 1, 'game_type': 'PRE', 'gameday': '2025-08-10',
         'gametime': '13:00', 'home_team': 'BAL', 'away_team': 'IND'},
    ])


def _depth():
    # Two snapshots: one before wk1 (09-03) and a fresher one also before wk1 (09-05);
    # a post-wk1 snapshot (09-10) must NOT leak into week 1 but feeds week 2.
    rows = []
    for dt, qb in [('2025-09-03T10:00:00Z', 'OLD_QB'),
                   ('2025-09-05T10:00:00Z', 'NEW_QB'),
                   ('2025-09-10T10:00:00Z', 'WK2_QB')]:
        rows += [
            {'dt': dt, 'team': 'BAL', 'pos_abb': 'QB', 'pos_rank': 1, 'gsis_id': qb},
            {'dt': dt, 'team': 'BAL', 'pos_abb': 'WR', 'pos_rank': 1, 'gsis_id': 'WR1'},
            {'dt': dt, 'team': 'BAL', 'pos_abb': 'WR', 'pos_rank': 2, 'gsis_id': 'WR2'},
            {'dt': dt, 'team': 'BAL', 'pos_abb': 'LT', 'pos_rank': 1, 'gsis_id': 'LTACK'},
            {'dt': dt, 'team': 'BAL', 'pos_abb': 'KR', 'pos_rank': 1, 'gsis_id': 'RETURNER'},
        ]
    return pd.DataFrame(rows)


class TestNormalize2025Depth(unittest.TestCase):
    def test_picks_latest_pre_gameday_snapshot(self):
        out = depth_2025.normalize_2025_depth(_depth(), _schedule(), 2025)
        wk1 = out[(out['week'] == 1) & (out['position'] == 'QB')]
        self.assertEqual(list(wk1['gsis_id']), ['NEW_QB'])  # 09-05, not 09-03, not 09-10

    def test_coarsens_positions_and_keeps_rank(self):
        out = depth_2025.normalize_2025_depth(_depth(), _schedule(), 2025)
        wk1 = out[out['week'] == 1]
        self.assertEqual(set(wk1['position']), {'QB', 'WR', 'T'})  # LT->T, KR dropped
        lt = wk1[wk1['position'] == 'T'].iloc[0]
        self.assertEqual(lt['depth_team'], 1)

    def test_reg_only_and_contract_columns(self):
        out = depth_2025.normalize_2025_depth(_depth(), _schedule(), 2025)
        self.assertEqual(set(out['game_type']), {'REG'})
        self.assertEqual(list(out.columns),
                         ['season', 'week', 'club_code', 'game_type',
                          'position', 'depth_team', 'gsis_id'])
        self.assertEqual(set(out['week']), {1, 2})

    def test_empty_depth_returns_empty_contract(self):
        out = depth_2025.normalize_2025_depth(pd.DataFrame(), _schedule(), 2025)
        self.assertEqual(len(out), 0)
        self.assertIn('depth_team', out.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data/madden/test_depth_2025.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.madden.depth_2025'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/madden/depth_2025.py
"""Normalize nflverse's 2025+ depth-chart schema (dated snapshots, granular
side-aware `pos_abb`, no week/game_type) into the legacy week-keyed contract that
starters.py consumes. For each team-week we take the latest snapshot strictly
before the game day (leakage-safe) and coarsen positions to the pre-2025 vocabulary
so `available_starters` selects one starter per coarse slot — matching the
all-old-schema training distribution."""
import logging
import pandas as pd

logger = logging.getLogger(__name__)

_CONTRACT = ['season', 'week', 'club_code', 'game_type',
             'position', 'depth_team', 'gsis_id']

# new granular pos_abb -> pre-2025 coarse `position`. Unmapped (H/KR/PR returners,
# kept off the starter set just like the old vocabulary) drop out.
POS_ABB_TO_COARSE = {
    'QB': 'QB', 'RB': 'RB', 'FB': 'FB', 'TE': 'TE', 'WR': 'WR', 'C': 'C',
    'LT': 'T', 'RT': 'T', 'LG': 'G', 'RG': 'G',
    'LCB': 'CB', 'RCB': 'CB', 'NB': 'CB', 'FS': 'FS', 'SS': 'SS',
    'LDE': 'DE', 'RDE': 'DE', 'LDT': 'DT', 'RDT': 'DT', 'NT': 'NT',
    'LILB': 'ILB', 'RILB': 'ILB', 'MLB': 'MLB', 'SLB': 'OLB', 'WLB': 'OLB',
    'PK': 'K', 'P': 'P', 'LS': 'LS',
}


def _empty():
    return pd.DataFrame(columns=_CONTRACT)


def _team_gameday(schedule, season):
    """Long frame [week, team, gameday] for REG games (gameday as midnight Timestamp)."""
    reg = schedule[(schedule['season'] == season) & (schedule['game_type'] == 'REG')]
    gameday = pd.to_datetime(reg['gameday'], errors='coerce')
    parts = []
    for side in ('home_team', 'away_team'):
        parts.append(pd.DataFrame({'week': reg['week'].values,
                                   'team': reg[side].values,
                                   'gameday': gameday.values}))
    return pd.concat(parts, ignore_index=True)


def normalize_2025_depth(depth_raw, schedule, season):
    """See module docstring. Returns the `_CONTRACT` frame (possibly empty)."""
    if depth_raw is None or depth_raw.empty or 'pos_abb' not in depth_raw.columns:
        if depth_raw is not None and not depth_raw.empty:
            logger.warning('season %s: depth frame lacks 2025 schema (no pos_abb) '
                           '— quarantining', season)
        return _empty()
    d = depth_raw.copy()
    d['position'] = d['pos_abb'].map(POS_ABB_TO_COARSE)
    d = d[d['position'].notna()].copy()
    d['_dt'] = pd.to_datetime(d['dt'], errors='coerce', utc=True).dt.tz_localize(None)
    schedule_days = _team_gameday(schedule, season)
    frames = []
    for team, team_days in schedule_days.groupby('team'):
        snaps = d[d['team'] == team]
        if snaps.empty:
            continue
        for _, gw in team_days.iterrows():
            pre = snaps[snaps['_dt'] < gw['gameday']]
            if pre.empty:
                continue
            latest = pre[pre['_dt'] == pre['_dt'].max()].copy()
            latest['season'] = season
            latest['week'] = gw['week']
            latest['club_code'] = team
            latest['game_type'] = 'REG'
            latest['depth_team'] = latest['pos_rank']
            frames.append(latest[_CONTRACT])
    return pd.concat(frames, ignore_index=True) if frames else _empty()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data/madden/test_depth_2025.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data/madden/depth_2025.py tests/data/madden/test_depth_2025.py
git commit -m "feat(madden): normalize 2025+ depth schema to legacy week-keyed contract"
```

---

## Task 4: Season-dispatched starters

**Files:**
- Modify: `src/data/madden/starters.py`
- Test: `tests/data/madden/test_starters.py`

**Interfaces:**
- Consumes: `depth_2025.normalize_2025_depth` (Task 3); existing `available_starters` (unchanged), `OUT_STATUSES`, `TEAM_ABBR_MAPPINGS`.
- Produces: `get_weekly_starters(years)` unchanged signature/return (`['season','week','team','gsis_id','position']`), now routing ≤2024 → old `import_depth_charts` schema and ≥2025 → `import_depth_charts([y])` + `import_schedules([y])` → `normalize_2025_depth`, then the shared `available_starters` loop. Mixed `years` lists are supported (concatenated).

- [ ] **Step 1: Write the failing test** (append a class to `tests/data/madden/test_starters.py`)

```python
from src.data.madden import depth_2025  # noqa: E402  (top of file with the others)


class TestGetWeeklyStarters2025Dispatch(unittest.TestCase):
    def _new_depth(self):
        return pd.DataFrame([
            {'dt': '2025-09-03T10:00:00Z', 'team': 'BAL', 'pos_abb': 'QB',
             'pos_rank': 1, 'gsis_id': 'LAMAR'},
            {'dt': '2025-09-03T10:00:00Z', 'team': 'BAL', 'pos_abb': 'QB',
             'pos_rank': 2, 'gsis_id': 'BACKUP_QB'},
        ])

    def _schedule(self):
        return pd.DataFrame([
            {'season': 2025, 'week': 1, 'game_type': 'REG', 'gameday': '2025-09-07',
             'gametime': '13:00', 'home_team': 'BAL', 'away_team': 'CLE'},
        ])

    def test_2025_routes_through_new_schema(self):
        with mock.patch.object(starters.nfl, 'import_depth_charts',
                               return_value=self._new_depth()), \
             mock.patch.object(starters.nfl, 'import_schedules',
                               return_value=self._schedule()), \
             mock.patch.object(starters.nfl, 'import_injuries',
                               return_value=pd.DataFrame(
                                   columns=['gsis_id', 'report_status', 'team',
                                            'season', 'week'])):
            out = starters.get_weekly_starters([2025])
        qb = out[(out['team'] == 'BAL') & (out['position'] == 'QB')].iloc[0]
        self.assertEqual(qb['gsis_id'], 'LAMAR')
        self.assertEqual(qb['week'], 1)

    def test_2025_quarantines_on_unexpected_schema(self):
        # An old-shaped frame for a 2025 request -> no pos_abb -> empty (quarantine).
        old_shape = pd.DataFrame([
            {'season': 2025, 'week': 1, 'game_type': 'REG', 'club_code': 'BAL',
             'depth_team': '1', 'position': 'QB', 'gsis_id': 'LAMAR'}])
        with mock.patch.object(starters.nfl, 'import_depth_charts',
                               return_value=old_shape), \
             mock.patch.object(starters.nfl, 'import_schedules',
                               return_value=self._schedule()), \
             mock.patch.object(starters.nfl, 'import_injuries',
                               return_value=pd.DataFrame(
                                   columns=['gsis_id', 'report_status'])):
            out = starters.get_weekly_starters([2025])
        self.assertEqual(len(out), 0)
        self.assertEqual(list(out.columns),
                         ['season', 'week', 'team', 'gsis_id', 'position'])
```

Note: also confirm the existing ≤2024 tests in this file still pass unchanged.

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data/madden/test_starters.py::TestGetWeeklyStarters2025Dispatch -v`
Expected: FAIL — 2025 path not implemented (returns empty for both, or errors).

- [ ] **Step 3: Write minimal implementation**

Replace `get_weekly_starters` in `src/data/madden/starters.py` with a season-dispatched version and add helpers (keep `available_starters`, `OUT_STATUSES`, imports unchanged; add `from src.data.madden import depth_2025`):

```python
_OLD_REQUIRED = {'game_type', 'club_code', 'depth_team', 'gsis_id', 'position'}
_DEPTH_CONTRACT = ['season', 'week', 'club_code', 'game_type',
                   'position', 'depth_team', 'gsis_id']


def _normalized_old_depth(years):
    """Pre-2025 nflverse depth schema -> the shared depth contract (REG only)."""
    try:
        depth = nfl.import_depth_charts(years)
    except Exception as exc:
        logger.warning('import_depth_charts(%s) unavailable (%s)', years, exc)
        return None
    if depth is None or depth.empty:
        return None
    missing = _OLD_REQUIRED - set(depth.columns)
    if missing:
        logger.warning('import_depth_charts(%s) unexpected schema (missing %s)',
                       years, missing)
        return None
    depth = depth[depth['game_type'] == 'REG'].copy()
    depth['club_code'] = depth['club_code'].replace(TEAM_ABBR_MAPPINGS)
    return depth[_DEPTH_CONTRACT]


def _normalized_new_depth(year):
    """2025+ nflverse depth schema -> the shared depth contract via depth_2025."""
    try:
        depth_raw = nfl.import_depth_charts([year])
        schedule = nfl.import_schedules([year])
    except Exception as exc:
        logger.warning('2025+ depth/schedule import for %s failed (%s)', year, exc)
        return None
    norm = depth_2025.normalize_2025_depth(depth_raw, schedule, year)
    return norm if not norm.empty else None


def _load_injuries(years):
    try:
        injuries = nfl.import_injuries(years)
        injuries['team'] = injuries['team'].replace(TEAM_ABBR_MAPPINGS)
        return injuries
    except Exception as exc:
        logger.warning('import_injuries(%s) unavailable (%s) — all players available',
                       years, exc)
        return pd.DataFrame(columns=['gsis_id', 'report_status', 'team', 'season', 'week'])


def _starters_from_depth(depth, injuries):
    frames = []
    for (season, week, club), grp in depth.groupby(['season', 'week', 'club_code']):
        inj = injuries[(injuries['season'] == season) & (injuries['week'] == week)
                       & (injuries['team'] == club)]
        chosen = available_starters(grp, inj)
        chosen['season'], chosen['week'], chosen['team'] = season, week, club
        frames.append(chosen)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    return out[['season', 'week', 'team', 'gsis_id', 'position']]


def get_weekly_starters(years):
    years = [years] if isinstance(years, int) else list(years)
    _empty_starters = pd.DataFrame(
        columns=['season', 'week', 'team', 'gsis_id', 'position'])
    parts = []
    old_years = [y for y in years if y < 2025]
    if old_years:
        od = _normalized_old_depth(old_years)
        if od is not None:
            parts.append(od)
    for y in (y for y in years if y >= 2025):
        nd = _normalized_new_depth(y)
        if nd is not None:
            parts.append(nd)
    if not parts:
        logger.warning('get_weekly_starters(%s): no usable depth charts — empty', years)
        return _empty_starters
    depth = pd.concat(parts, ignore_index=True)
    starters = _starters_from_depth(depth, _load_injuries(years))
    return _empty_starters if starters is None else starters
```

- [ ] **Step 4: Run test to verify it passes (incl. the unchanged ≤2024 suite)**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data/madden/test_starters.py -v`
Expected: PASS — new dispatch tests + all pre-existing starters tests.

- [ ] **Step 5: Commit**

```bash
git add src/data/madden/starters.py tests/data/madden/test_starters.py
git commit -m "feat(madden): season-dispatch weekly starters (2025+ new depth schema)"
```

---

## Task 5: nflverse-rosters gsis bridge

**Files:**
- Modify: `src/data/madden/ids.py`
- Test: `tests/data/madden/test_ids.py`

**Interfaces:**
- Consumes: existing `normalize_name`, `TEAM_ABBR_MAPPINGS`; `nfl_data_py.import_seasonal_rosters`.
- Produces: `attach_gsis_id_from_rosters(df, season, rosters=None) -> DataFrame` — adds a `gsis_id` column by matching `df.full_name`+`df.team` to nflverse seasonal-roster `player_name`+`team` (whose `player_id` IS the gsis id); name-only fallback with ambiguity→None. `rosters` injectable for tests.

- [ ] **Step 1: Write the failing test** (append to `tests/data/madden/test_ids.py`)

```python
class TestAttachGsisFromRosters(unittest.TestCase):
    def _rosters(self):
        return pd.DataFrame([
            {'player_name': 'Lamar Jackson', 'player_id': '00-0034796',
             'team': 'BAL', 'position': 'QB'},
            {'player_name': 'Josh Allen', 'player_id': '00-0034857',
             'team': 'BUF', 'position': 'QB'},
            {'player_name': 'Josh Allen', 'player_id': '00-0035000',
             'team': 'JAX', 'position': 'LB'},  # name collision -> ambiguous
        ])

    def test_primary_name_team_match(self):
        df = pd.DataFrame([{'full_name': 'Lamar Jackson', 'team': 'BAL', 'position': 'QB'}])
        out = ids.attach_gsis_id_from_rosters(df, 2025, rosters=self._rosters())
        self.assertEqual(out.iloc[0]['gsis_id'], '00-0034796')

    def test_team_match_disambiguates_name_collision(self):
        df = pd.DataFrame([{'full_name': 'Josh Allen', 'team': 'BUF', 'position': 'QB'}])
        out = ids.attach_gsis_id_from_rosters(df, 2025, rosters=self._rosters())
        self.assertEqual(out.iloc[0]['gsis_id'], '00-0034857')

    def test_unmatched_team_falls_back_to_name_only_when_unambiguous(self):
        df = pd.DataFrame([{'full_name': 'Lamar Jackson', 'team': 'XXX', 'position': 'QB'}])
        out = ids.attach_gsis_id_from_rosters(df, 2025, rosters=self._rosters())
        self.assertEqual(out.iloc[0]['gsis_id'], '00-0034796')

    def test_ambiguous_name_only_yields_na(self):
        df = pd.DataFrame([{'full_name': 'Josh Allen', 'team': 'XXX', 'position': 'QB'}])
        out = ids.attach_gsis_id_from_rosters(df, 2025, rosters=self._rosters())
        self.assertTrue(pd.isna(out.iloc[0]['gsis_id']))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data/madden/test_ids.py::TestAttachGsisFromRosters -v`
Expected: FAIL — `AttributeError: ... has no attribute 'attach_gsis_id_from_rosters'`.

- [ ] **Step 3: Write minimal implementation** (append to `src/data/madden/ids.py`; add `import nfl_data_py as nfl` at top)

```python
def attach_gsis_id_from_rosters(df, season, rosters=None):
    """Attach gsis_id by matching to nflverse seasonal rosters (player_id IS gsis).

    Primary key normalized_name|team; fallback name-only (ambiguous -> None). Used
    for the 2025+ madden-tools source, whose JSON carries no gsis/pfr id."""
    if rosters is None:
        rosters = nfl.import_seasonal_rosters([season])
    proc = rosters.copy()
    proc['team'] = proc['team'].replace(TEAM_ABBR_MAPPINGS)
    proc['_norm_name'] = proc['player_name'].map(normalize_name)

    proc['_key'] = proc['_norm_name'] + '|' + proc['team'].astype(str)
    lookup_primary = dict(zip(proc['_key'], proc['player_id']))

    name_lookup = {}
    for _, r in proc.iterrows():
        n = r['_norm_name']
        if n in name_lookup:
            if name_lookup[n] != r['player_id']:
                name_lookup[n] = None  # ambiguous — multiple distinct gsis
        else:
            name_lookup[n] = r['player_id']

    out = df.copy()
    out['_norm_name'] = out['full_name'].map(normalize_name)
    out['gsis_id'] = (out['_norm_name'] + '|' + out['team'].astype(str)).map(lookup_primary)
    mask = out['gsis_id'].isna()
    if mask.any():
        out.loc[mask, 'gsis_id'] = out.loc[mask, '_norm_name'].map(name_lookup)
    return out.drop(columns=['_norm_name'])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data/madden/test_ids.py -v`
Expected: PASS — new bridge tests + existing ids tests.

- [ ] **Step 5: Commit**

```bash
git add src/data/madden/ids.py tests/data/madden/test_ids.py
git commit -m "feat(madden): nflverse-rosters gsis bridge for 2025+ source"
```

---

## Task 6: Season-routed collection

**Files:**
- Modify: `src/data/madden/collect.py`
- Test: `tests/data/madden/test_collect.py`

**Interfaces:**
- Consumes: `launch.load_madden_launch` (Task 2), `ids.attach_gsis_id_from_rosters` (Task 5), existing `ingest.load_madden_season`, `ids.attach_gsis_id`, `roles.classify_roles`, `features.*`, `starters.get_weekly_starters` (Task 4).
- Produces: `season_to_game_version(season) -> str` (`f"madden-{season-1999}"`, e.g. 2025→`madden-26`); `_player_season(season)` routes ≥2025 → launch loader + rosters bridge, ≤2024 → unchanged theedgepredictor + processed bridge. `get_madden_data` signature/return unchanged.

- [ ] **Step 1: Write the failing test** (append to `tests/data/madden/test_collect.py`)

```python
class TestSeasonRouting(unittest.TestCase):
    def test_game_version_formula(self):
        self.assertEqual(collect.season_to_game_version(2025), 'madden-26')
        self.assertEqual(collect.season_to_game_version(2026), 'madden-27')

    def test_2025_routes_to_launch_and_rosters(self):
        launch_df = pd.DataFrame([{
            'season': 2025, 'full_name': 'Lamar Jackson', 'team': 'BAL',
            'position': 'QB', 'overall': 94, 'weight': 215,
            'power_moves': 40, 'finesse_moves': 35}])
        with mock.patch.object(collect.launch, 'load_madden_launch',
                               return_value=launch_df) as mlaunch, \
             mock.patch.object(collect.ids, 'attach_gsis_id_from_rosters',
                               side_effect=lambda d, s: d.assign(gsis_id='G')) as mbridge, \
             mock.patch.object(collect.ingest, 'load_madden_season') as mold:
            out = collect._player_season(2025)
        mlaunch.assert_called_once_with('madden-26', 2025)
        mbridge.assert_called_once()
        mold.assert_not_called()
        self.assertEqual(out.iloc[0]['gsis_id'], 'G')
        self.assertIn('role', out.columns)

    def test_pre2025_uses_theedgepredictor_path(self):
        old_df = pd.DataFrame([{
            'season': 2023, 'full_name': 'X Y', 'team': 'KC', 'position': 'QB',
            'overall': 99, 'weight': 230, 'power_moves': 20, 'finesse_moves': 20}])
        with mock.patch.object(collect.ingest, 'load_madden_season',
                               return_value=old_df) as mold, \
             mock.patch.object(collect.ids, 'attach_gsis_id',
                               side_effect=lambda d, s: d.assign(gsis_id='G')) as mbridge, \
             mock.patch.object(collect.launch, 'load_madden_launch') as mlaunch:
            collect._player_season(2023)
        mold.assert_called_once_with(2023)
        mbridge.assert_called_once()
        mlaunch.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data/madden/test_collect.py::TestSeasonRouting -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'season_to_game_version'` / `launch`.

- [ ] **Step 3: Write minimal implementation**

In `src/data/madden/collect.py`, add `from src.data.madden import launch` to the imports and replace `_player_season` + add the helper:

```python
def season_to_game_version(season):
    """NFL season -> madden-tools game version. Madden NFL N covers season N-1, so
    the version number is season-1999 (2025 -> 'madden-26')."""
    return f'madden-{season - 1999}'


def _player_season(season):
    """Plan-1 pipeline for one season -> classified player-season frame.

    Routes by season: >=2025 uses the madden-tools launch JSON + nflverse-rosters
    gsis bridge; <=2024 keeps the theedgepredictor raw CSV + processed/ bridge.
    Returns an empty OUTPUT_COLUMNS frame if the season is unavailable."""
    if season >= 2025:
        df = launch.load_madden_launch(season_to_game_version(season), season)
    else:
        df = ingest.load_madden_season(season)
    if df.empty:
        logger.warning('season %s: empty Madden ingest — skipping', season)
        return df
    try:
        if season >= 2025:
            df = ids.attach_gsis_id_from_rosters(df, season)
        else:
            df = ids.attach_gsis_id(df, season)
        return roles.classify_roles(df)
    except Exception as exc:
        logger.warning('season %s: Madden id/role pipeline failed (%s) — skipping',
                       season, exc)
        return pd.DataFrame()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data/madden/test_collect.py -v`
Expected: PASS — routing tests + existing collect tests.

- [ ] **Step 5: Run the full madden suite (regression gate)**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data/madden/ -v`
Expected: PASS — every madden test, old and new.

- [ ] **Step 6: Commit**

```bash
git add src/data/madden/collect.py tests/data/madden/test_collect.py
git commit -m "feat(madden): season-route ingest + gsis bridge (2025+ via madden-tools)"
```

---

## Task 7: Coverage report + live verification + docs

This is the data-quality deliverable. It runs the full chain against **live data**
(needs network: madden-tools CDN + nflverse). First confirm the CDN base.

**Files:**
- Create: `scripts/experiments/madden_coverage_report.py`
- Create: `data/predict_games/madden_coverage/coverage_report.json` (output)
- Modify: `docs/madden-features-and-vlm.md` (§3c/§3e), `README.md` (run-log note)

**Interfaces:**
- Consumes: `src.data.madden.collect.get_madden_data`, `feature_groups.partition` (optional).
- Produces: `coverage_report(seasons) -> dict` and a `__main__` that writes the JSON and prints a per-season table of gsis-match-rate + `_ovr` non-null coverage.

- [ ] **Step 1: Confirm the CDN base, then write the failing test**

Confirm the live CDN domain (the value of the webapp's `NEXT_PUBLIC_CDN_DOMAIN`):
read `~/ClaudeProjects/MaddenTools/madden-tools/apps/webapp/.env.local` (or ask the
user). Export it for the run: `export MADDEN_TOOLS_CDN_BASE="https://<confirmed-domain>"`.
Sanity-check one object resolves:
`curl -sI "$MADDEN_TOOLS_CDN_BASE/madden-26/json/iterations.json" | head -1` → expect `200`.

Write the unit test for the pure aggregation (no network):

```python
# tests/data/madden/test_coverage_report.py
import unittest
import numpy as np
import pandas as pd
from scripts.experiments import madden_coverage_report as rpt


class TestCoverageAgg(unittest.TestCase):
    def test_ovr_coverage_and_match_rate(self):
        tw = pd.DataFrame({
            'season': [2025, 2025],
            'madden_qb_ovr': [1.2, np.nan],
            'madden_rb_ovr': [0.3, 0.4],
            'team': ['BAL', 'SF'], 'week': [1, 1],
        })
        cov = rpt.season_ovr_coverage(tw)
        self.assertAlmostEqual(cov, 0.75)  # 3 of 4 _ovr cells non-null
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data/madden/test_coverage_report.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the report script**

```python
# scripts/experiments/madden_coverage_report.py
"""Phase-0 data-quality report: per-season Madden `_ovr` coverage (proves 2025
0% -> populated) and gsis match rate. Run after the season-routing changes land."""
import json
import os
import numpy as np
import pandas as pd
from src.data.madden import collect


def season_ovr_coverage(team_week):
    ovr = [c for c in team_week.columns
           if c.startswith('madden_') and c.endswith('_ovr')]
    if not ovr or team_week.empty:
        return 0.0
    return float(team_week[ovr].notna().to_numpy().mean())


def coverage_report(seasons):
    rows = []
    for s in seasons:
        tw = collect.get_madden_data([s])
        rows.append({'season': s, 'n_team_weeks': int(len(tw)),
                     'ovr_coverage': round(season_ovr_coverage(tw), 4)})
    return {'seasons': rows}


def main():
    seasons = list(range(2021, 2026))
    report = coverage_report(seasons)
    out_dir = 'data/predict_games/madden_coverage'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'coverage_report.json'), 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f"{'season':>6} {'team_weeks':>11} {'ovr_coverage':>13}")
    for r in report['seasons']:
        print(f"{r['season']:>6} {r['n_team_weeks']:>11} {r['ovr_coverage']:>13.4f}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the unit test, then the live report**

Run unit: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data/madden/test_coverage_report.py -v` → PASS.
Run live (network): `MADDEN_TOOLS_CDN_BASE="$MADDEN_TOOLS_CDN_BASE" conda run --no-capture-output -n nfl-predictions python -m scripts.experiments.madden_coverage_report`
Expected: a table where **2025 `ovr_coverage` > 0.30** (target the 0.66–0.78 band) and 2021–2024 are unchanged from history. If 2025 is still 0.0, inspect: CDN base resolved? launch iteration found? 2025 gsis match rate? depth normalization producing weeks? (Quarantine fallback is acceptable only if the CDN truly lacks 2025 launch data — document it.)

- [ ] **Step 5: Update docs + run-log, then commit**

- In `docs/madden-features-and-vlm.md` §3c and §3f, replace the "⚠ 2025 = 0% → FAILS" lines with the measured 2025 coverage and note the fix (new depth schema + madden-tools launch source). Update §3e match-rate note with the measured 2025 gsis rate.
- In `README.md`, add a brief **data-quality note** under the run-log narrative (NOT a numbered run): "Phase 0 (madden-launch-ratings): wired 2025 launch ratings (madden-tools CDN `players.json` + nflverse-rosters gsis bridge) and fixed the 2025 nflverse depth-chart schema; 2025 `_ovr` coverage 0% → <measured>%. Run 28 reserved for the first XGBoost-RFE modeling result."

```bash
git add scripts/experiments/madden_coverage_report.py \
        tests/data/madden/test_coverage_report.py \
        data/predict_games/madden_coverage/coverage_report.json \
        docs/madden-features-and-vlm.md README.md
git commit -m "experiment(madden): Phase 0 coverage report — 2025 launch ratings 0% -> populated"
```

---

## Self-Review

**Spec coverage:**
- Workstream 1 (launch adapter) → Tasks 1–2. ✓
- Workstream 2 (source routing) → Task 6. ✓
- Workstream 3 (starters 2025 fix) → Tasks 3–4. ✓
- Workstream 4 (gsis bridge) → Task 5; (wire 2025 + coverage report + VLM table) → Task 7. ✓
- Decisions: CDN file-contract (Task 2), fix-starters w/ quarantine fallback (Tasks 3–4 incl. quarantine test), nflverse-rosters bridge (Task 5), data-quality note w/ Run 28 reserved (Task 7). ✓
- Leakage/parity constraints → Global Constraints + Task 3 (pre-gameday snapshot, coarse map). ✓
- Out-of-scope (parquet upgrade, modeling) → not planned. ✓

**Placeholder scan:** No TBD/TODO; every code/test step has complete code. The one external unknown (live CDN domain) is handled as an env var with an explicit confirm-and-curl step in Task 7-Step 1, not a code placeholder.

**Type consistency:** `OUTPUT_COLUMNS` frame is the shared contract across launch.py / collect.py. `_DEPTH_CONTRACT`/`_CONTRACT` columns match between `depth_2025.normalize_2025_depth` (Task 3) and starters' `_normalized_old_depth`/`_starters_from_depth` (Task 4). `attach_gsis_id_from_rosters(df, season, rosters=None)` signature matches its call in `collect._player_season` (Task 6) — note `collect` calls it with the two positional args only. `load_madden_launch(game_version, season)` call in Task 6 matches the Task 2 signature. `season_to_game_version` formula (2025→madden-26) is consistent in Task 6 test and impl.
