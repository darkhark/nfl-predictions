# Madden Ratings — Data Layer Implementation Plan (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the public Madden `raw/` ratings files into a clean, tested,
gsis-keyed, role-classified **player-season table** that Plan 2 consumes to build
team-game features.

**Architecture:** A new `src/data/madden/` package with three focused modules —
`ingest.py` (download + era-aware schema normalization), `ids.py` (bridge to nflverse
`gsis_id`), `roles.py` (vocabulary map + attribute classifier ported from the pre-EDGE
guide-generator logic). Each is independently unit-tested with `unittest` and synthetic
fixtures (matching the repo's existing test style).

**Tech Stack:** Python 3.13, pandas, `nfl_data_py` (conda env `nfl-predictions`),
`unittest`, data from `github.com/theedgepredictor/nfl-madden-data` (`raw/` + `processed/`).

## Global Constraints

- **Source of record:** `raw/` stage CSVs at
  `https://raw.githubusercontent.com/theedgepredictor/nfl-madden-data/main/data/madden/raw/{season}.csv`.
  Verified launch/preseason ratings (leakage-safe), 2001–2025, ~0% null on Overall /
  Power Moves / Finesse Moves / Weight.
- **Season key = NFL season-start year** (file `2025.csv` = Madden 26 = NFL 2025). No
  off-by-one.
- **Schema detection is per-file, not per-year:** classic Title-Case vs nested
  (`stats/*/value`, split `firstName`/`lastName`) — detect by presence of any column
  containing `stats/`.
- **Team codes** normalized with the repo's existing `TEAM_ABBR_MAPPINGS`
  (`src/data/transformations.py:6-10`: `STL→LA, SD→LAC, OAK→LV`).
- **Test framework:** `unittest` (stdlib), synthetic in-memory DataFrames, network reads
  mocked via `mock.patch.object(<module>.pd, 'read_csv', ...)` — mirror
  `tests/data/weekly/test_collect.py`.
- **Pre-EDGE classifier thresholds (verbatim from guide-generator history):**
  edge if `OLB and power_moves + finesse_moves >= 130`, or `DE and weight <= 280`;
  interior DL if `DT or (DE and weight >= 280)`; off-ball LB if
  `OLB and power_moves + finesse_moves < 130`, or `MLB`.

---

## File Structure

- `src/data/madden/__init__.py` — package marker.
- `src/data/madden/ingest.py` — `load_madden_season(season)` → tidy player-season frame.
  Owns schema detection + per-schema column maps + team normalization + 2025 position
  backfill flag.
- `src/data/madden/ids.py` — `attach_gsis_id(df, season)` → adds `gsis_id` via the
  `processed/` bridge with a name+position+team fallback.
- `src/data/madden/roles.py` — `assign_role(position, weight, power_moves, finesse_moves)`
  and `classify_roles(df)` → adds `role` + `side` columns. Pure functions, no I/O.
- `tests/data/madden/__init__.py`, `tests/data/madden/test_ingest.py`,
  `test_ids.py`, `test_roles.py`.

**Plan 1 output schema (the contract Plan 2 consumes)** — one row per rated player-season:

```
season:int, full_name:str, team:str(normalized), position:str(Madden side-aware:
  LT/LG/C/RG/RT, LE/RE/DT, LOLB/ROLB/MLB, CB/FS/SS, QB/HB/FB/WR/TE, K/P/LS),
overall:int, weight:float, power_moves:float, finesse_moves:float,
gsis_id:str|NA, role:str(one of: edge|interior_dl|off_ball_lb|cornerback|safety|
  qb|backfield|receiver|tight_end|interior_ol|exterior_ol|specialist|unknown),
side:str(left|right|none)
```

---

## Task 1: Madden raw ingestion + era-aware normalization

**Files:**
- Create: `src/data/madden/__init__.py` (empty)
- Create: `src/data/madden/ingest.py`
- Test: `tests/data/madden/__init__.py` (empty), `tests/data/madden/test_ingest.py`

**Interfaces:**
- Produces:
  - `RAW_URL_TEMPLATE: str` = the raw URL with `{season}` placeholder.
  - `CLASSIC_COLUMN_MAP: dict[str,str]`, `NESTED_COLUMN_MAP: dict[str,str]`.
  - `is_nested_schema(columns: Iterable[str]) -> bool`
  - `normalize_madden_frame(df: pd.DataFrame, season: int) -> pd.DataFrame` — pure;
    maps either schema to columns `[season, full_name, team, position, overall, weight,
    power_moves, finesse_moves]`, applies `TEAM_ABBR_MAPPINGS`.
  - `load_madden_season(season: int) -> pd.DataFrame` — downloads then normalizes.

- [ ] **Step 1: Write the failing test for schema detection + normalization**

```python
# tests/data/madden/test_ingest.py
import unittest
from unittest import mock
import pandas as pd
from src.data.madden import ingest


class TestIsNestedSchema(unittest.TestCase):
    def test_nested_detected_by_stats_prefix(self):
        self.assertTrue(ingest.is_nested_schema(
            ['firstName', 'lastName', 'stats/overall/value', 'weight']))

    def test_classic_not_nested(self):
        self.assertFalse(ingest.is_nested_schema(
            ['Team', 'Name', 'Position', 'Overall', 'Power Moves']))


class TestNormalizeClassic(unittest.TestCase):
    def _classic(self):
        # Column order mirrors raw/2018.csv (Name col, Overall, Power/Finesse Moves, Weight)
        return pd.DataFrame([
            {'Team': 'OAK', 'Name': 'Khalil Mack', 'Position': 'LOLB',
             'Overall': 96, 'Power Moves': 90, 'Finesse Moves': 91, 'Weight': 252},
        ])

    def test_classic_maps_and_normalizes_team(self):
        out = ingest.normalize_madden_frame(self._classic(), 2018)
        row = out.iloc[0]
        self.assertEqual(list(out.columns), ['season', 'full_name', 'team', 'position',
                                             'overall', 'weight', 'power_moves',
                                             'finesse_moves'])
        self.assertEqual(row['team'], 'LV')          # OAK -> LV via TEAM_ABBR_MAPPINGS
        self.assertEqual(row['full_name'], 'Khalil Mack')
        self.assertEqual(row['position'], 'LOLB')
        self.assertEqual(row['overall'], 96)
        self.assertEqual(row['power_moves'], 90)
        self.assertEqual(row['season'], 2018)


class TestNormalizeNested(unittest.TestCase):
    def _nested(self):
        return pd.DataFrame([
            {'firstName': 'Maxx', 'lastName': 'Crosby', 'team': 'Raiders',
             'Position': 'RE', 'stats/overall/value': 89, 'weight': 255,
             'stats/powerMoves/value': 88, 'stats/finesseMoves/value': 90},
        ])

    def test_nested_builds_full_name_and_maps_stats(self):
        out = ingest.normalize_madden_frame(self._nested(), 2024)
        row = out.iloc[0]
        self.assertEqual(row['full_name'], 'Maxx Crosby')
        self.assertEqual(row['overall'], 89)
        self.assertEqual(row['finesse_moves'], 90)
        self.assertEqual(row['position'], 'RE')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n nfl-predictions python -m pytest tests/data/madden/test_ingest.py -v`
Expected: FAIL (`ModuleNotFoundError: src.data.madden.ingest`).

- [ ] **Step 3: Implement `ingest.py`**

```python
# src/data/madden/ingest.py
"""Download + era-normalize the edgepredictor Madden `raw/` ratings files.

`raw/` mixes two schemas; detection is per-file (presence of any `stats/` column),
never per-year (2023 classic, 2024 nested, 2025 classic). Output is one tidy frame
with a fixed column contract regardless of source schema."""
import pandas as pd
from src.data.transformations import TEAM_ABBR_MAPPINGS

RAW_URL_TEMPLATE = (
    'https://raw.githubusercontent.com/theedgepredictor/nfl-madden-data/'
    'main/data/madden/raw/{season}.csv'
)

# Team nicknames used by the nested (EA-API) files -> standard abbreviations.
NICKNAME_TO_ABBR = {
    'Cardinals': 'ARI', 'Falcons': 'ATL', 'Ravens': 'BAL', 'Bills': 'BUF',
    'Panthers': 'CAR', 'Bears': 'CHI', 'Bengals': 'CIN', 'Browns': 'CLE',
    'Cowboys': 'DAL', 'Broncos': 'DEN', 'Lions': 'DET', 'Packers': 'GB',
    'Texans': 'HOU', 'Colts': 'IND', 'Jaguars': 'JAX', 'Chiefs': 'KC',
    'Raiders': 'LV', 'Chargers': 'LAC', 'Rams': 'LA', 'Dolphins': 'MIA',
    'Vikings': 'MIN', 'Patriots': 'NE', 'Saints': 'NO', 'Giants': 'NYG',
    'Jets': 'NYJ', 'Eagles': 'PHI', 'Steelers': 'PIT', '49ers': 'SF',
    'Seahawks': 'SEA', 'Buccaneers': 'TB', 'Titans': 'TEN', 'Commanders': 'WAS',
}

OUTPUT_COLUMNS = ['season', 'full_name', 'team', 'position', 'overall', 'weight',
                  'power_moves', 'finesse_moves']

CLASSIC_COLUMN_MAP = {
    'Position': 'position', 'Overall': 'overall', 'Overall Rating': 'overall',
    'Weight': 'weight', 'Power Moves': 'power_moves', 'Finesse Moves': 'finesse_moves',
}
NESTED_COLUMN_MAP = {
    'Position': 'position', 'stats/overall/value': 'overall', 'weight': 'weight',
    'stats/powerMoves/value': 'power_moves', 'stats/finesseMoves/value': 'finesse_moves',
}


def is_nested_schema(columns):
    """Nested (EA-API) files carry flattened `stats/<attr>/value` columns."""
    return any('stats/' in str(c) for c in columns)


def _resolve_column(df, candidates):
    """Return the first candidate column present in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_madden_frame(df, season):
    """Map either raw schema to OUTPUT_COLUMNS, normalize team to standard abbr."""
    nested = is_nested_schema(df.columns)
    out = pd.DataFrame()
    out['season'] = [season] * len(df)
    if nested:
        out['full_name'] = (df['firstName'].astype(str).str.strip() + ' '
                            + df['lastName'].astype(str).str.strip()).str.strip()
        out['team'] = df['team'].map(NICKNAME_TO_ABBR).fillna(df['team'])
        colmap = NESTED_COLUMN_MAP
    else:
        name_col = _resolve_column(df, ['Name', 'full_name'])
        out['full_name'] = df[name_col].astype(str).str.strip()
        out['team'] = df['Team']
        colmap = CLASSIC_COLUMN_MAP
    for src_col, dst_col in colmap.items():
        if src_col in df.columns and dst_col not in out.columns:
            out[dst_col] = df[src_col].values
    out['team'] = out['team'].replace(TEAM_ABBR_MAPPINGS)
    for required in ('position', 'overall', 'weight', 'power_moves', 'finesse_moves'):
        if required not in out.columns:
            out[required] = pd.NA
    return out[OUTPUT_COLUMNS]


def load_madden_season(season):
    """Download and normalize one season's raw Madden ratings."""
    df = pd.read_csv(RAW_URL_TEMPLATE.format(season=season))
    return normalize_madden_frame(df, season)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n nfl-predictions python -m pytest tests/data/madden/test_ingest.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Add a live smoke check for the 2025 position-quality caveat**

Add to `tests/data/madden/test_ingest.py`:

```python
class TestPositionQualityFlag(unittest.TestCase):
    def test_normalize_tolerates_missing_position(self):
        # 2025 raw has `position_short_label == "False"` for ~88% of rows; once renamed
        # to `position`, normalization must not crash and must leave those as-is for the
        # downstream depth-chart backfill (Plan 2).
        df = pd.DataFrame([{'Team': 'KC', 'Name': 'Patrick Mahomes',
                            'Position': 'False', 'Overall': 99, 'Weight': 225,
                            'Power Moves': 50, 'Finesse Moves': 60}])
        out = ingest.normalize_madden_frame(df, 2025)
        self.assertEqual(out.iloc[0]['position'], 'False')
        self.assertEqual(out.iloc[0]['overall'], 99)
```

- [ ] **Step 6: Run + commit**

```bash
conda run -n nfl-predictions python -m pytest tests/data/madden/test_ingest.py -v
git add src/data/madden/__init__.py src/data/madden/ingest.py tests/data/madden/
git commit -m "feat(madden): era-aware ingestion + normalization of raw ratings"
```

---

## Task 2: Bridge Madden players to nflverse `gsis_id`

**Files:**
- Create: `src/data/madden/ids.py`
- Test: `tests/data/madden/test_ids.py`

**Interfaces:**
- Consumes: tidy frame from Task 1 (`season, full_name, team, position, …`).
- Produces:
  - `PROCESSED_URL_TEMPLATE: str`
  - `normalize_name(name: str) -> str` — lowercase, strip punctuation/suffixes for joining.
  - `attach_gsis_id(df: pd.DataFrame, season: int, processed: pd.DataFrame|None=None) -> pd.DataFrame`
    — adds a `gsis_id` column (string or `pd.NA`). `processed` injectable for tests.

The `processed/` file carries both `fullname` and `player_id` (the gsis id, `00-00xxxxx`)
plus `team`. Join key: normalized name + team (+ season is implicit per file). Fallback:
normalized name + position + team. Unmatched → `pd.NA`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/madden/test_ids.py
import unittest
import pandas as pd
from src.data.madden import ids


class TestNormalizeName(unittest.TestCase):
    def test_strips_case_punct_suffix(self):
        self.assertEqual(ids.normalize_name('A.J. Brown'), 'aj brown')
        self.assertEqual(ids.normalize_name('Michael Pittman Jr.'), 'michael pittman')


class TestAttachGsisId(unittest.TestCase):
    def _madden(self):
        return pd.DataFrame([
            {'season': 2023, 'full_name': 'Patrick Mahomes', 'team': 'KC',
             'position': 'QB'},
            {'season': 2023, 'full_name': 'Nobody Here', 'team': 'KC',
             'position': 'QB'},
        ])

    def _processed(self):
        return pd.DataFrame([
            {'fullname': 'Patrick Mahomes', 'team': 'KC', 'player_id': '00-0033873'},
        ])

    def test_matches_on_name_and_team(self):
        out = ids.attach_gsis_id(self._madden(), 2023, processed=self._processed())
        self.assertEqual(out.iloc[0]['gsis_id'], '00-0033873')

    def test_unmatched_is_na(self):
        out = ids.attach_gsis_id(self._madden(), 2023, processed=self._processed())
        self.assertTrue(pd.isna(out.iloc[1]['gsis_id']))
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n nfl-predictions python -m pytest tests/data/madden/test_ids.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `ids.py`**

```python
# src/data/madden/ids.py
"""Attach nflverse gsis_id to Madden players by bridging through the edgepredictor
`processed/` file (which carries both `fullname` and the gsis `player_id`)."""
import re
import pandas as pd
from src.data.transformations import TEAM_ABBR_MAPPINGS

PROCESSED_URL_TEMPLATE = (
    'https://raw.githubusercontent.com/theedgepredictor/nfl-madden-data/'
    'main/data/madden/processed/{season}.csv'
)
_SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}


def normalize_name(name):
    """Lowercase, drop punctuation and generational suffixes for joining."""
    cleaned = re.sub(r'[^a-z ]', '', str(name).lower())
    tokens = [t for t in cleaned.split() if t not in _SUFFIXES]
    return ' '.join(tokens)


def attach_gsis_id(df, season, processed=None):
    if processed is None:
        processed = pd.read_csv(PROCESSED_URL_TEMPLATE.format(season=season))
    proc = processed.copy()
    proc['team'] = proc['team'].replace(TEAM_ABBR_MAPPINGS)
    proc['_key'] = proc['fullname'].map(normalize_name) + '|' + proc['team'].astype(str)
    lookup = dict(zip(proc['_key'], proc['player_id']))

    out = df.copy()
    keys = out['full_name'].map(normalize_name) + '|' + out['team'].astype(str)
    out['gsis_id'] = keys.map(lookup)
    out['gsis_id'] = out['gsis_id'].where(out['gsis_id'].notna(), other=pd.NA)
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n nfl-predictions python -m pytest tests/data/madden/test_ids.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/madden/ids.py tests/data/madden/test_ids.py
git commit -m "feat(madden): bridge players to nflverse gsis_id via processed file"
```

---

## Task 3: Role + side classification (port pre-EDGE guide-generator logic)

**Files:**
- Create: `src/data/madden/roles.py`
- Test: `tests/data/madden/test_roles.py`

**Interfaces:**
- Consumes: tidy frame (`position, weight, power_moves, finesse_moves`).
- Produces:
  - `POSITION_TO_ROLE: dict[str,str]` — unambiguous label → role.
  - `EDGE_PASS_RUSH_THRESHOLD = 130`, `INTERIOR_DL_WEIGHT_THRESHOLD = 280`.
  - `assign_role(position, weight, power_moves, finesse_moves) -> str`
  - `assign_side(position) -> str` (`left|right|none`)
  - `classify_roles(df: pd.DataFrame) -> pd.DataFrame` — adds `role` and `side` columns.

Roles vocabulary (matches Plan 1 output contract): `edge, interior_dl, off_ball_lb,
cornerback, safety, qb, backfield, receiver, tight_end, interior_ol, exterior_ol,
specialist, unknown`.

- [ ] **Step 1: Write the failing test (the ambiguous defenders are the point)**

```python
# tests/data/madden/test_roles.py
import unittest
import pandas as pd
from src.data.madden import roles


class TestAssignRole(unittest.TestCase):
    def test_olb_high_passrush_is_edge(self):
        # 3-4 edge: LOLB with power+finesse >= 130
        self.assertEqual(roles.assign_role('LOLB', 250, 88, 90), 'edge')

    def test_olb_low_passrush_is_off_ball(self):
        # 4-3 weakside backer: OLB that can't rush
        self.assertEqual(roles.assign_role('ROLB', 240, 40, 45), 'off_ball_lb')

    def test_light_de_is_edge_heavy_de_is_interior(self):
        self.assertEqual(roles.assign_role('LE', 270, 80, 85), 'edge')
        self.assertEqual(roles.assign_role('RE', 295, 70, 60), 'interior_dl')

    def test_dt_and_mlb_unambiguous(self):
        self.assertEqual(roles.assign_role('DT', 310, 70, 40), 'interior_dl')
        self.assertEqual(roles.assign_role('MLB', 240, 30, 35), 'off_ball_lb')

    def test_modern_labels_map_directly(self):
        self.assertEqual(roles.assign_role('LEDGE', 250, 0, 0), 'edge')
        self.assertEqual(roles.assign_role('MIKE', 240, 0, 0), 'off_ball_lb')
        self.assertEqual(roles.assign_role('SAM', 240, 0, 0), 'off_ball_lb')

    def test_offense_and_dbs(self):
        self.assertEqual(roles.assign_role('LT', 320, 0, 0), 'exterior_ol')
        self.assertEqual(roles.assign_role('C', 300, 0, 0), 'interior_ol')
        self.assertEqual(roles.assign_role('CB', 190, 0, 0), 'cornerback')
        self.assertEqual(roles.assign_role('FS', 200, 0, 0), 'safety')
        self.assertEqual(roles.assign_role('QB', 220, 0, 0), 'qb')


class TestAssignSide(unittest.TestCase):
    def test_side_from_label(self):
        self.assertEqual(roles.assign_side('LE'), 'left')
        self.assertEqual(roles.assign_side('REDGE'), 'right')
        self.assertEqual(roles.assign_side('DT'), 'none')


class TestClassifyRoles(unittest.TestCase):
    def test_adds_role_and_side_columns(self):
        df = pd.DataFrame([{'position': 'LOLB', 'weight': 250,
                            'power_moves': 88, 'finesse_moves': 90}])
        out = roles.classify_roles(df)
        self.assertEqual(out.iloc[0]['role'], 'edge')
        self.assertEqual(out.iloc[0]['side'], 'left')
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n nfl-predictions python -m pytest tests/data/madden/test_roles.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `roles.py`**

```python
# src/data/madden/roles.py
"""Classify a Madden player into a scheme/era-invariant positional role.

Unambiguous labels (offense, DBs, DT/MLB, and the modern LEDGE/REDGE/SAM/MIKE/WILL)
map directly. The ambiguous classic labels -- DE (LE/RE) and OLB (LOLB/ROLB) -- are
split by attributes, porting the pre-EDGE guide-generator thresholds:
  edge        = OLB with power_moves+finesse_moves >= 130, or DE with weight <= 280
  interior_dl = DT, or DE with weight >= 280
  off_ball_lb = OLB with power_moves+finesse_moves < 130, or MLB
"""
import pandas as pd

EDGE_PASS_RUSH_THRESHOLD = 130
INTERIOR_DL_WEIGHT_THRESHOLD = 280

POSITION_TO_ROLE = {
    'QB': 'qb', 'HB': 'backfield', 'RB': 'backfield', 'FB': 'backfield',
    'WR': 'receiver', 'TE': 'tight_end',
    'LT': 'exterior_ol', 'RT': 'exterior_ol',
    'LG': 'interior_ol', 'RG': 'interior_ol', 'C': 'interior_ol',
    'DT': 'interior_dl', 'NT': 'interior_dl',
    'MLB': 'off_ball_lb', 'MIKE': 'off_ball_lb', 'WILL': 'off_ball_lb',
    'SAM': 'off_ball_lb', 'ILB': 'off_ball_lb',
    'LEDGE': 'edge', 'REDGE': 'edge',
    'CB': 'cornerback', 'FS': 'safety', 'SS': 'safety',
    'K': 'specialist', 'P': 'specialist', 'LS': 'specialist',
}
_OLB_LABELS = {'LOLB', 'ROLB', 'OLB'}
_DE_LABELS = {'LE', 'RE', 'DE'}


def _num(value):
    return 0.0 if pd.isna(value) else float(value)


def assign_role(position, weight, power_moves, finesse_moves):
    if position in POSITION_TO_ROLE:
        return POSITION_TO_ROLE[position]
    if position in _OLB_LABELS:
        if _num(power_moves) + _num(finesse_moves) >= EDGE_PASS_RUSH_THRESHOLD:
            return 'edge'
        return 'off_ball_lb'
    if position in _DE_LABELS:
        if _num(weight) <= INTERIOR_DL_WEIGHT_THRESHOLD:
            return 'edge'
        return 'interior_dl'
    return 'unknown'


def assign_side(position):
    p = str(position)
    if p.startswith('L'):
        return 'left'
    if p.startswith('R'):
        return 'right'
    return 'none'


def classify_roles(df):
    out = df.copy()
    out['role'] = [
        assign_role(r['position'], r['weight'], r['power_moves'], r['finesse_moves'])
        for _, r in out.iterrows()
    ]
    out['side'] = out['position'].map(assign_side)
    return out
```

Note: `assign_side` keys off the L/R prefix, which is correct for `LE/RE/LEDGE/REDGE/
LOLB/ROLB/LT/RT/LG/RG`. Center (`C`) and side-less labels return `none`. Plan 2 owns
the depth-chart reconciliation for any slot the depth chart lists without a side.

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n nfl-predictions python -m pytest tests/data/madden/test_roles.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/madden/roles.py tests/data/madden/test_roles.py
git commit -m "feat(madden): port pre-EDGE attribute role/side classifier"
```

---

## Task 4: End-to-end data-layer smoke test (live, one season)

**Files:**
- Test: `tests/data/madden/test_ingest.py` (append a live integration class)

**Interfaces:** Consumes Tasks 1–3 together. No new production code — this proves the
contract on real 2023 data before Plan 2 builds on it.

- [ ] **Step 1: Write the live integration test**

```python
# append to tests/data/madden/test_ingest.py
from src.data.madden import ids as madden_ids
from src.data.madden import roles as madden_roles


class TestLiveSeasonContract(unittest.TestCase):
    """Live network test (2023). Verifies the Plan 1 output contract end-to-end."""
    @classmethod
    def setUpClass(cls):
        df = ingest.load_madden_season(2023)
        df = madden_ids.attach_gsis_id(df, 2023)
        cls.df = madden_roles.classify_roles(df)

    def test_has_contract_columns(self):
        for col in ['season', 'full_name', 'team', 'position', 'overall', 'weight',
                    'power_moves', 'finesse_moves', 'gsis_id', 'role', 'side']:
            self.assertIn(col, self.df.columns)

    def test_overall_complete_and_plausible(self):
        self.assertEqual(self.df['overall'].isna().sum(), 0)
        self.assertTrue(self.df['overall'].between(20, 99).all())

    def test_majority_of_players_get_gsis_id(self):
        match_rate = self.df['gsis_id'].notna().mean()
        self.assertGreater(match_rate, 0.85)

    def test_every_player_classified(self):
        self.assertEqual((self.df['role'] == 'unknown').sum(), 0)

    def test_known_edge_classified_as_edge(self):
        mack = self.df[self.df['full_name'] == 'Khalil Mack']
        self.assertFalse(mack.empty)
        self.assertEqual(mack.iloc[0]['role'], 'edge')
```

- [ ] **Step 2: Run it**

Run: `conda run -n nfl-predictions python -m pytest tests/data/madden/test_ingest.py::TestLiveSeasonContract -v`
Expected: PASS. If `test_majority_of_players_get_gsis_id` fails, tighten `normalize_name`
(add more suffix/punctuation handling) before proceeding — Plan 2 depends on this join.

- [ ] **Step 3: Commit**

```bash
git add tests/data/madden/test_ingest.py
git commit -m "test(madden): live end-to-end data-layer contract on 2023"
```

---

## Self-Review

**Spec coverage (Plan 1 portion of the design doc):**
- §3.1 raw source + §3.3 schema drift → Task 1 (`is_nested_schema`, dual column maps). ✓
- §3.2 season alignment → Global Constraints + Task 4 uses `season` directly. ✓
- §3.4 gsis bridge + name/pos/team fallback → Task 2. ✓ (fallback-by-position is a noted
  extension if 85% match gate fails; base bridge is name+team.)
- §5 role classifier + vocabulary map → Task 3 (exact thresholds 130 / 280). ✓
- 2025 position `"False"` caveat → Task 1 Step 5 (tolerated, deferred to Plan 2 backfill). ✓
- Team normalization → reuse `TEAM_ABBR_MAPPINGS`. ✓

**Deferred to Plan 2 (not gaps):** weekly starter identification (depth charts +
injuries), per-position/grouped/matchup feature columns + games-played diffs,
`collect_all`/`partition.py` wiring, parquet rebuild + ablation. Plan 1 stops at the
clean player-season table.

**Placeholder scan:** none — every step has runnable test + implementation code.

**Type consistency:** `normalize_madden_frame` → `attach_gsis_id` → `classify_roles` all
operate on the same column set; `OUTPUT_COLUMNS` (Task 1) feeds `gsis_id` (Task 2) feeds
`role`/`side` (Task 3); Task 4 asserts the union. Thresholds named identically in
constants and `assign_role`.

**Open item to resolve during execution:** if the live gsis match rate is below ~0.85,
implement the name+position+team fallback inside `attach_gsis_id` (a second lookup keyed
on `name|position|team`) before Plan 2.
