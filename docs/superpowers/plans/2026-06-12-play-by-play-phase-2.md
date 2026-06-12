# Play-by-Play Features Phase 2 Implementation Plan (Directional)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add directional run/pass features — average yards and explosive-play rate for 7 run buckets (`run_location` × `run_gap`) and 6 pass buckets (`pass_location` × `pass_length`) — riding the Phase 1 architecture unchanged.

**Architecture:** Phase 1's pipeline (`src/data/play_by_play/collect.py`) already does everything structural: wp-context splitting, component-sum aggregation, per-season caching, ratio-of-cumsums cumulative rates with the defense context swap, and shared ranks. Phase 2 only (a) adds 5 raw columns and 39 bucket component columns to the play aggregation, and (b) extends `RATE_METRICS` with 26 generated directional metrics. Everything downstream (pivot, cache, cumulative rates, ranks, collect_all merge, leakage shift) picks the new columns up with zero changes. The caches must be regenerated because the cached schema changes.

**Tech Stack:** Python 3.11 (conda env `nfl-predictions`), pandas, nfl_data_py, unittest.

**Spec:** `docs/superpowers/specs/2026-06-12-play-by-play-features-design.md` (Phase 2 section). Baselines to beat: XGBoost run 7 (AUROC 0.696 / acc 0.647), BART run 8 (0.705 / 0.660), both on the 2024+2025 pooled hold-out.

---

## Conventions and verified facts

- Run everything through `conda run -n nfl-predictions python ...` from the repo root; single test module form: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect -v` (no `unittest discover`).
- **Verified against real downloads (2026-06-12):** on rush plays, `run_location` is ~95–96% non-null in ALL eras (2003 included) with values `left/middle/right`; `run_gap` is ~67–72% non-null with values `end/tackle/guard` (middle runs have NaN gap by definition). On pass plays, `pass_location`/`pass_length` are **0% non-null 2003–2005** and ~82% from 2006+ (values `left/middle/right` and `short/deep`; sacks, scrambles, and throwaways are unlabeled). `yards_gained` is ~97% non-null overall (gaps are no-plays outside our pass/rush universe).
- Era consequence: directional **pass** features are all-NaN before 2006 — same handling as PROE/CPOE (NaN features, NaN ranks over teams without values; XGBoost handles natively). Directional **run** features work for the full 2003+ span.
- Unlabeled plays (NaN location/gap/length) simply contribute to no bucket; they remain in the Phase 1 aggregate metrics. Per-bucket denominators are **bucket attempt counts**, so rates are over labeled plays only.
- Explosive thresholds (locked in the spec): rush ≥ 10 yards, pass ≥ 20 yards.
- Naming traps: no new name fragment may contain `off`, `opp`, `home`, `away`, `target`, or start with `team_` (collect_all renames by substring). All bucket and metric names below were checked: `run_left_end`, …, `pass_deep_middle`, `_yards_per_attempt`, `_explosive_rate` are all clean.
- New feature math: 26 new metrics × 3 contexts × 2 sides × 3 kinds (rate/rank/rank_change) = **468 new feature columns**; `get_play_by_play_features` grows 180 → **648** features (651 columns with keys). Cache schema grows from 51 to **168** component columns (56 components × 3 contexts).
- **Cache invalidation:** Phase 1 caches in `data/play_by_play/aggregated/` lack the new components and are read back verbatim — they MUST be deleted and regenerated (Task 4). Until then, only unit tests (which mock the download) are valid.

## File structure

| File | Action | Responsibility |
|---|---|---|
| `src/data/play_by_play/collect.py` | Modify | New constants + bucket assignment helpers + directional components in `_aggregate_play_components` |
| `tests/data/play_by_play/test_collect.py` | Modify | Extend `make_play`; new bucket/component tests; update feature-count assertion |
| `data/play_by_play/aggregated/*.parquet` | Regenerate | New 168-component schema |
| `data/predict_games/*` | Regenerate | Dataset + feature list with 468 new candidates |
| `notebooks/.../cross_validation/rfe.ipynb`, `grid_search.ipynb`, `bart.ipynb` | Re-run | Runs 9 (XGBoost) and 10 (BART); `BEST_NUM_FEATS` updated to the new RFE recommendation |
| `README.md` | Modify | Runs 9–10 in the experiment log |

---

### Task 1: Directional constants and extended play fixture

**Files:**
- Modify: `src/data/play_by_play/collect.py` (constants region, after `RATE_METRICS`)
- Modify: `tests/data/play_by_play/test_collect.py` (`make_play`)

- [ ] **Step 1: Extend `make_play` in the test file**

In `tests/data/play_by_play/test_collect.py`, replace the `make_play` function with this version (adds the five new fields; all existing call sites keep working because the new parameters default to unlabeled/zero):

```python
def make_play(posteam='AAA', defteam='BBB', season=2023, week=1, season_type='REG',
              play_id=1, game_id='2023_01_AAA_BBB', is_pass=0, is_rush=0,
              down=1, yardline_100=75.0, third_down_converted=0.0, success=0.0,
              epa=0.0, wp=0.5, xpass=None, fixed_drive=1, fixed_drive_result='Punt',
              yards_gained=0.0, run_location=None, run_gap=None,
              pass_location=None, pass_length=None):
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
    }
```

(Location/gap/length columns are object dtype with None for unlabeled, matching real nflfastR string-or-NaN columns.)

- [ ] **Step 2: Write the failing constants test**

Append to `tests/data/play_by_play/test_collect.py`:

NOTE ON TASK BOUNDARIES: Task 1 defines the bucket constants and the generated
`DIRECTIONAL_*` lists but does NOT yet append them to `PLAY_COMPONENT_COLUMNS` /
`RATE_METRICS` — the aggregation doesn't build those columns until Task 2, and
appending early would make the component groupby KeyError across the whole suite.
Task 2 wires them in together with the implementation, keeping every commit green.

```python
class TestDirectionalConstants(unittest.TestCase):

    def test_thirteen_buckets(self):
        self.assertEqual(len(collect.RUN_BUCKETS), 7)
        self.assertEqual(len(collect.PASS_BUCKETS), 6)
        self.assertEqual(
            collect.DIRECTIONAL_BUCKETS, collect.RUN_BUCKETS + collect.PASS_BUCKETS)

    def test_directional_generated_lists(self):
        # 3 components per bucket and 2 metrics per bucket, generated from the
        # bucket lists; wired into the aggregate lists in the next task
        self.assertEqual(len(collect.DIRECTIONAL_COMPONENT_COLUMNS), 39)
        self.assertEqual(len(collect.DIRECTIONAL_RATE_METRICS), 26)
        self.assertIn('run_left_end_attempt_count', collect.DIRECTIONAL_COMPONENT_COLUMNS)
        self.assertIn(
            ('pass_deep_right_explosive_rate', 'pass_deep_right_explosive_count',
             'pass_deep_right_attempt_count'),
            collect.DIRECTIONAL_RATE_METRICS)

    def test_directional_required_columns(self):
        for column in ('yards_gained', 'run_location', 'run_gap',
                       'pass_location', 'pass_length'):
            self.assertIn(column, collect.REQUIRED_PBP_COLUMNS)
```

- [ ] **Step 3: Run to verify failure**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect.TestDirectionalConstants -v`
Expected: FAIL with `AttributeError: ... no attribute 'RUN_BUCKETS'`

- [ ] **Step 4: Add the constants**

In `src/data/play_by_play/collect.py`:

(a) Extend `REQUIRED_PBP_COLUMNS` (replace the list — five new entries at the end):

```python
# Verified present for every season 2003-2025. epa/wp/success/fixed_drive are fully
# populated on pass/rush plays back to 2003; xpass is fully null before 2006, which
# makes PROE NaN there by the zero-denominator rule. pass_location/pass_length are
# fully null before 2006 too (directional pass features go NaN there the same way);
# run_location is ~95% populated on rushes in all eras, run_gap ~70% (middle runs
# have no gap by definition).
REQUIRED_PBP_COLUMNS = [
    'posteam', 'defteam', 'season', 'week', 'season_type', 'play_id', 'game_id',
    'pass', 'rush', 'down', 'yardline_100', 'third_down_converted', 'success',
    'epa', 'wp', 'xpass', 'fixed_drive', 'fixed_drive_result',
    'yards_gained', 'run_location', 'run_gap', 'pass_location', 'pass_length',
]
```

(b) Insert directly after the `AGGREGATION_KEY_COLUMNS` line and BEFORE
`PLAY_COMPONENT_COLUMNS` (Task 2 appends these generated lists to the aggregate
lists, so they must be defined above them in the file):

```python
# Phase 2: directional buckets. Runs split by run_location x run_gap (middle has no
# gap by definition); passes by pass_length x pass_location. Plays with unlabeled
# direction contribute to no bucket (they remain in the aggregate Phase 1 metrics);
# per-bucket denominators are bucket attempt counts, so rates cover labeled plays only.
RUN_BUCKETS = [
    'run_left_end', 'run_left_tackle', 'run_left_guard', 'run_middle',
    'run_right_guard', 'run_right_tackle', 'run_right_end',
]
PASS_BUCKETS = [
    'pass_short_left', 'pass_short_middle', 'pass_short_right',
    'pass_deep_left', 'pass_deep_middle', 'pass_deep_right',
]
DIRECTIONAL_BUCKETS = RUN_BUCKETS + PASS_BUCKETS

EXPLOSIVE_RUSH_YARDS = 10
EXPLOSIVE_PASS_YARDS = 20

DIRECTIONAL_COMPONENT_COLUMNS = [
    f'{bucket}_{component}'
    for bucket in DIRECTIONAL_BUCKETS
    for component in ('attempt_count', 'yards_sum', 'explosive_count')
]

DIRECTIONAL_RATE_METRICS = (
    [(f'{bucket}_yards_per_attempt', f'{bucket}_yards_sum', f'{bucket}_attempt_count')
     for bucket in DIRECTIONAL_BUCKETS]
    + [(f'{bucket}_explosive_rate', f'{bucket}_explosive_count', f'{bucket}_attempt_count')
       for bucket in DIRECTIONAL_BUCKETS]
)
```

Do NOT touch `PLAY_COMPONENT_COLUMNS`, `COMPONENT_COLUMNS`, or `RATE_METRICS` in this
task — that happens in Task 2 together with the component construction, so the suite
stays green at every commit.

- [ ] **Step 5: Run to verify pass, and that nothing else broke**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect -v`
Expected: ALL 33 tests PASS (30 existing + 3 new; nothing consumes the new constants yet).

- [ ] **Step 6: Commit**

```bash
git add src/data/play_by_play/collect.py tests/data/play_by_play/test_collect.py
git commit -m "Register directional buckets, components, and rate metrics"
```

---

### Task 2: Bucket assignment and directional components

**Files:**
- Modify: `src/data/play_by_play/collect.py` (`_aggregate_play_components` + two new helpers)
- Test: `tests/data/play_by_play/test_collect.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/play_by_play/test_collect.py`:

```python
class TestDirectionalComponents(unittest.TestCase):

    def _aggregate(self, plays):
        return collect._aggregate_play_components(pd.DataFrame(plays))

    def test_run_bucket_assignment(self):
        result = self._aggregate([
            make_play(play_id=1, is_rush=1, run_location='left', run_gap='end',
                      yards_gained=12.0, epa=0.5),
            make_play(play_id=2, is_rush=1, run_location='middle', run_gap=None,
                      yards_gained=3.0, epa=0.1),
            # left run with no gap label: contributes to NO bucket
            make_play(play_id=3, is_rush=1, run_location='left', run_gap=None,
                      yards_gained=5.0, epa=0.2),
            # unlabeled run: contributes to NO bucket
            make_play(play_id=4, is_rush=1, yards_gained=4.0, epa=0.1),
        ])
        row = result.iloc[0]
        self.assertEqual(row['run_left_end_attempt_count'], 1)
        self.assertEqual(row['run_left_end_yards_sum'], 12.0)
        self.assertEqual(row['run_left_end_explosive_count'], 1)  # 12 >= 10
        self.assertEqual(row['run_middle_attempt_count'], 1)
        self.assertEqual(row['run_middle_yards_sum'], 3.0)
        self.assertEqual(row['run_middle_explosive_count'], 0)
        # all plays still count in the aggregate universe
        self.assertEqual(row['play_count'], 4)
        # the two unlabeled runs landed in no bucket
        bucket_attempts = sum(
            row[f'{bucket}_attempt_count'] for bucket in collect.RUN_BUCKETS)
        self.assertEqual(bucket_attempts, 2)

    def test_pass_bucket_assignment(self):
        result = self._aggregate([
            make_play(play_id=1, is_pass=1, pass_location='right', pass_length='deep',
                      yards_gained=25.0, epa=1.5),
            make_play(play_id=2, is_pass=1, pass_location='middle', pass_length='short',
                      yards_gained=19.0, epa=0.8),
            # sack/scramble/throwaway: pass==1 but no location -> no bucket
            make_play(play_id=3, is_pass=1, yards_gained=-7.0, epa=-1.2),
        ])
        row = result.iloc[0]
        self.assertEqual(row['pass_deep_right_attempt_count'], 1)
        self.assertEqual(row['pass_deep_right_yards_sum'], 25.0)
        self.assertEqual(row['pass_deep_right_explosive_count'], 1)  # 25 >= 20
        self.assertEqual(row['pass_short_middle_attempt_count'], 1)
        self.assertEqual(row['pass_short_middle_explosive_count'], 0)  # 19 < 20
        self.assertEqual(row['play_count'], 3)

    def test_explosive_thresholds_differ_for_run_and_pass(self):
        result = self._aggregate([
            # 12-yard run IS explosive (>= 10)...
            make_play(play_id=1, is_rush=1, run_location='middle',
                      yards_gained=12.0, epa=0.5),
            # ...but a 12-yard pass is NOT (< 20)
            make_play(play_id=2, is_pass=1, pass_location='left', pass_length='short',
                      yards_gained=12.0, epa=0.5),
        ])
        row = result.iloc[0]
        self.assertEqual(row['run_middle_explosive_count'], 1)
        self.assertEqual(row['pass_short_left_explosive_count'], 0)

    def test_buckets_split_by_wp_context(self):
        result = self._aggregate([
            make_play(play_id=1, is_rush=1, run_location='middle',
                      yards_gained=5.0, epa=0.2, wp=0.5),
            make_play(play_id=2, is_rush=1, run_location='middle',
                      yards_gained=30.0, epa=1.0, wp=0.97),
        ])
        competitive = result[result[collect.CONTEXT_COL] == collect.COMPETITIVE].iloc[0]
        leading = result[result[collect.CONTEXT_COL] == collect.GARBAGE_LEADING].iloc[0]
        self.assertEqual(competitive['run_middle_attempt_count'], 1)
        self.assertEqual(competitive['run_middle_explosive_count'], 0)
        self.assertEqual(leading['run_middle_attempt_count'], 1)
        self.assertEqual(leading['run_middle_explosive_count'], 1)
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect.TestDirectionalComponents -v`
Expected: FAIL — `KeyError: 'run_left_end_attempt_count'` (groupby selects PLAY_COMPONENT_COLUMNS which now include directional names that `_aggregate_play_components` never created).

- [ ] **Step 3: Wire the generated lists into the aggregates and implement**

In `src/data/play_by_play/collect.py`:

(a) Append the generated lists to the two aggregate lists (this is what makes the
pivot/cache/cumulative/rank machinery pick the new columns up automatically):

```python
PLAY_COMPONENT_COLUMNS = [
    'play_count', 'epa_sum', 'success_sum',
    'dropback_count', 'dropback_epa_sum', 'dropback_success_sum',
    'rush_count', 'rush_epa_sum', 'rush_success_sum',
    'early_down_count', 'early_down_success_sum',
    'third_down_count', 'third_down_conversion_sum',
    'xpass_play_count', 'pass_minus_xpass_sum',
] + DIRECTIONAL_COMPONENT_COLUMNS
```

```python
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
] + DIRECTIONAL_RATE_METRICS
```

(`DRIVE_COMPONENT_COLUMNS` and `COMPONENT_COLUMNS = PLAY_COMPONENT_COLUMNS +
DRIVE_COMPONENT_COLUMNS` stay untouched; `COMPONENT_COLUMNS` grows to 56 automatically.)

(b) In the SAME commit, update the two count-sensitive test assertions:

In `TestGetPlayByPlayFeatures.test_feature_count_and_naming`, change

```python
        # 10 metrics x 3 contexts x 2 sides x 3 column kinds (rate, rank, rank_change)
        self.assertEqual(len(feature_cols), 180)
```

to

```python
        # 36 metrics (10 aggregate + 26 directional) x 3 contexts x 2 sides
        # x 3 column kinds (rate, rank, rank_change)
        self.assertEqual(len(feature_cols), 648)
```

and append to `TestDirectionalConstants`:

```python
    def test_directional_lists_wired_into_aggregates(self):
        self.assertEqual(len(collect.COMPONENT_COLUMNS), 56)
        self.assertEqual(len(collect.RATE_METRICS), 36)
        for bucket in collect.DIRECTIONAL_BUCKETS:
            self.assertIn(f'{bucket}_attempt_count', collect.PLAY_COMPONENT_COLUMNS)
```

(c) Add the two helpers directly above `_aggregate_play_components`:

```python
def _assign_run_bucket(plays):
    """
    Label each rush with its directional bucket. Middle runs have no gap by definition;
    left/right runs need a gap label. Unlabeled runs (NaN location, or sided runs with
    NaN gap, ~5% and ~30% of rushes respectively) get no bucket — they still count in
    the aggregate Phase 1 metrics, and per-bucket denominators only cover labeled runs.
    """
    bucket = pd.Series(None, index=plays.index, dtype='object')
    is_rush = plays['rush'] == 1
    bucket[is_rush & (plays['run_location'] == 'middle')] = 'run_middle'
    sided = is_rush & plays['run_location'].isin(['left', 'right']) & plays['run_gap'].notna()
    bucket[sided] = 'run_' + plays.loc[sided, 'run_location'] + '_' + plays.loc[sided, 'run_gap']
    return bucket


def _assign_pass_bucket(plays):
    """
    Label each located pass with its depth x direction bucket. Sacks, scrambles, and
    throwaways carry pass == 1 with no location/length and get no bucket; before 2006
    nflfastR has no pass charting at all, so every pass is unlabeled there and the
    directional pass features are NaN for those seasons (like PROE/CPOE).
    """
    bucket = pd.Series(None, index=plays.index, dtype='object')
    located = (
        (plays['pass'] == 1)
        & plays['pass_location'].notna()
        & plays['pass_length'].notna()
    )
    bucket[located] = (
        'pass_' + plays.loc[located, 'pass_length'] + '_' + plays.loc[located, 'pass_location']
    )
    return bucket
```

(d) Inside `_aggregate_play_components`, add the directional component construction
after the `pass_minus_xpass_sum` line and before the final `return`:

```python
    directional_bucket = _assign_run_bucket(plays)
    directional_bucket = directional_bucket.where(directional_bucket.notna(),
                                                  _assign_pass_bucket(plays))
    yards = plays['yards_gained'].fillna(0)
    is_explosive = (
        ((plays['rush'] == 1) & (yards >= EXPLOSIVE_RUSH_YARDS))
        | ((plays['pass'] == 1) & (yards >= EXPLOSIVE_PASS_YARDS))
    )
    for bucket in DIRECTIONAL_BUCKETS:
        in_bucket = (directional_bucket == bucket).astype(int)
        plays[f'{bucket}_attempt_count'] = in_bucket
        plays[f'{bucket}_yards_sum'] = yards * in_bucket
        plays[f'{bucket}_explosive_count'] = is_explosive.astype(int) * in_bucket
```

- [ ] **Step 4: Run the full suite**

Run: `conda run -n nfl-predictions python -m unittest tests.data.play_by_play.test_collect -v`
Expected: all 38 tests PASS (33 from Task 1 + 4 new component tests + 1 new wiring
test; the Phase 1 component tests pass because unlabeled fixture plays produce
all-zero directional components, and the cumulative/rank tests built on
`make_component_row` already iterate `COMPONENT_COLUMNS` so they generate the
56-component grid automatically; `test_feature_count_and_naming` now asserts 648).

- [ ] **Step 5: Commit**

```bash
git add src/data/play_by_play/collect.py tests/data/play_by_play/test_collect.py
git commit -m "Add directional run/pass bucket components to play aggregation"
```

---

### Task 3: End-to-end feature check on real cached-free data

**Files:**
- Test only (throwaway verification, not committed as a test)

- [ ] **Step 1: Verify the public API on one real season (downloads 2023, ~1 min; bypasses stale caches with refresh)**

```bash
conda run -n nfl-predictions python -c "
from src.data.play_by_play import collect
import pandas as pd
feats = collect.get_play_by_play_features([2023], refresh=True)
cols = list(feats.columns)
print('total columns:', len(cols))  # expect 651 = 3 keys + 648
directional = [c for c in cols if 'run_left_end' in c or 'pass_deep_right' in c]
print('sample directional cols:', len(directional))  # 2 buckets x 2 metrics x 3 ctx x 2 sides x 3 kinds = 72... printed for eyeballing
kc = feats[(feats['team'] == 'KC')].sort_values('week')
col = 'off_run_middle_yards_per_attempt_competitive_cumulative_average'
print(kc[['week', col]].head(8).to_string(index=False))
print('non-null fraction week>=3:', round(feats[feats['week']>=3][col].notna().mean(), 3))
"
```

Expected: 651 total columns; the KC `run_middle` cumulative yards-per-attempt series is
plausible (roughly 3–6 yards) and monotone-smoothing across weeks; non-null fraction
near 1.0 for competitive-context run buckets by week 3.

NOTE: `refresh=True` regenerated and OVERWROTE `data/play_by_play/aggregated/2023.parquet`
with the new 168-component schema — the other 22 seasons are now stale relative to it.
Task 4 regenerates everything; do not run the assembly script or integration tests
between Tasks 3 and 4.

- [ ] **Step 2: Commit nothing** (verification only). Proceed immediately to Task 4.

---

### Task 4: Regenerate caches and the model dataset

**Files:**
- Regenerate: `data/play_by_play/aggregated/*.parquet` (all 23 seasons)
- Regenerate: `data/predict_games/input_data/schedule_and_weekly.parquet`, `data/predict_games/model_features_in/xgb_features_list.csv`

- [ ] **Step 1: Delete stale caches so the assembly run rebuilds every season (LONG: ~25 min of downloads)**

```bash
rm data/play_by_play/aggregated/*.parquet
conda run -n nfl-predictions python -m scripts.data_assembly.predict_game_winner.schedule_and_weekly
```

Expected: per-season "**** done." progress lines for 2003–2025, exit 0.

- [ ] **Step 2: Verify the regenerated dataset**

```bash
conda run -n nfl-predictions python -c "
import pandas as pd
df = pd.read_parquet('data/predict_games/input_data/schedule_and_weekly.parquet')
print('total columns:', len(df.columns))  # expect 916 + 936 = 1852 (468 new x target/opp)
directional = [c for c in df.columns if '_yards_per_attempt_' in c or '_explosive_rate_' in c]
print('directional feature columns:', len(directional))  # expect 936
late = df[df['week'] >= 3]
run_col = 'off_target_run_middle_yards_per_attempt_competitive_cumulative_average_rank'
pass_col = 'off_target_pass_deep_right_explosive_rate_competitive_cumulative_average_rank'
print('run col 2003+ non-null (weeks>=3):', round(late[run_col].notna().mean(), 3))
print('pass col pre-2006 non-null:', round(late[late['season'] < 2006][pass_col].notna().mean(), 3))  # expect 0.0
print('pass col 2006+ non-null:', round(late[late['season'] >= 2006][pass_col].notna().mean(), 3))
features = pd.read_csv('data/predict_games/model_features_in/xgb_features_list.csv')
print('features list length:', len(features))  # expect 910 + 936 = 1846
"
```

Expected: 1852 total columns, 936 directional features, run ranks ~fully populated for
weeks ≥ 3 in all eras, pass directional 0.0 pre-2006 and high 2006+, feature list 1846.

- [ ] **Step 3: Run the full integration suite (network, uses fresh caches)**

Run: `conda run -n nfl-predictions python -m unittest tests.data.test_collect_all -v`
Expected: all 8 PASS.

- [ ] **Step 4: Commit**

```bash
git add data/play_by_play/aggregated data/predict_games/input_data/schedule_and_weekly.parquet data/predict_games/model_features_in/xgb_features_list.csv
git commit -m "Regenerate caches and dataset with Phase 2 directional features"
```

---

### Task 5: Run 9 — cross-validated RFE + grid search (XGBoost)

**Files:**
- Re-run: `notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/rfe.ipynb`
- Modify + re-run: `notebooks/.../cross_validation/grid_search.ipynb` (`BEST_NUM_FEATS`)
- Modify: `README.md` (run 9 row + changelog)

- [ ] **Step 1: Execute RFE (LONG, ~30–60 min; the rank-only pool grows ~569 → ~880)**

```bash
PYTHONPATH=$(pwd) conda run -n nfl-predictions jupyter nbconvert --to notebook --execute --inplace notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/rfe.ipynb
```

- [ ] **Step 2: Read the recommendation** — extract `get_best_num_features(.005)` from the executed notebook's outputs (cell after `features_df`); call it `N9`.

```bash
conda run -n nfl-predictions python -c "
import json
nb = json.load(open('notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/rfe.ipynb'))
for cell in nb['cells']:
    if 'get_best_num_features' in ''.join(cell.get('source', [])):
        print(''.join(cell['outputs'][0]['data']['text/plain']))
"
```

- [ ] **Step 3: Point grid search at N9** — in `grid_search.ipynb` cell 1, set `BEST_NUM_FEATS = <N9>` and update its comment to reference the Phase 2 pool (edit the JSON the same way as the Phase 1 run: replace the literal in `cells[1].source`).

- [ ] **Step 4: Execute grid search (LONG, ~30–60 min)**

```bash
PYTHONPATH=$(pwd) conda run -n nfl-predictions jupyter nbconvert --to notebook --execute --inplace notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/grid_search.ipynb
```

- [ ] **Step 5: Extract hold-out metrics** — pooled AUROC and accuracy, computed exactly as for run 7:

```bash
conda run -n nfl-predictions python -c "
import pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
feats = list(pd.read_csv('data/predict_games/model_features_in/rfe_features_kfolds.csv', index_col=0).loc[<N9>].dropna().values)
df = pd.read_parquet('data/predict_games/input_data/schedule_and_weekly.parquet')
df = df.sample(frac=1, random_state=32).reset_index(drop=True)
holdout = df[df['season'] >= 2024]
model = xgb.XGBClassifier(); model.load_model('models/best_random_xgb_model.json')
print('pooled AUROC:', round(roc_auc_score(holdout['target_win'], model.predict_proba(holdout[feats])[:, 1]), 4))
print('accuracy:', round(model.score(holdout[feats], holdout['target_win']), 4))
"
```

- [ ] **Step 6: Record run 9 in README.md** — add the table row (`Rank-only + directional play-by-play (run/pass buckets), XGBoost | <N9> | <auroc> | <acc>`) and a `Run 8 → 9` changelog bullet stating: pool size, selected count, how many directional features were selected (count metric-name matches for `_yards_per_attempt`/`_explosive_rate` in the selected row), the hold-out deltas vs runs 5/7, and an honest interpretation either way.

- [ ] **Step 7: Commit**

```bash
git add README.md notebooks data/predict_games/model_features_in models/best_random_xgb_model.json
git commit -m "Record Run 9: XGBoost with Phase 2 directional play-by-play features"
```

---

### Task 6: Run 10 — BART on the run-9 feature set

**Files:**
- Modify + re-run: `notebooks/.../cross_validation/bart.ipynb` (`BEST_NUM_FEATS = <N9>`)
- Modify: `README.md` (run 10 row + changelog)

- [ ] **Step 1: Update `bart.ipynb`** — set `BEST_NUM_FEATS = <N9>` in `cells[1].source` and adjust the intro markdown to reference the run-9 selection (same JSON-edit pattern as run 8).

- [ ] **Step 2: Execute (a few minutes: 4 chains + posterior predictive)**

```bash
PYTHONPATH=$(pwd) conda run -n nfl-predictions jupyter nbconvert --to notebook --execute --inplace notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/bart.ipynb
```

- [ ] **Step 3: Extract the headline cell** (prints validation AUROC, hold-out AUROC, hold-out accuracy) and the Brier/log-loss cell from the executed notebook outputs.

- [ ] **Step 4: Record run 10 in README.md** — table row (`BART re-trained on the run-9 directional feature set | <N9> | <auroc> | <acc>`) + `Run 9 → 10` changelog bullet comparing against run 8 (0.705/0.660) and run 6.

- [ ] **Step 5: Commit**

```bash
git add README.md notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/bart.ipynb
git commit -m "Record Run 10: BART on Phase 2 directional feature set"
```

---

### Task 7: PR

- [ ] **Step 1: Push and open the PR**

```bash
git push -u origin add-directional-pbp-features
gh pr create --base master --head add-directional-pbp-features \
  --title "Phase 2 play-by-play: directional run/pass features (runs 9-10)" \
  --body "<summary of: 13 buckets x avg-yards/explosive-rate x 3 wp contexts = 468 new candidates; bucket-assignment rules incl. era caveats (no pass charting pre-2006); run 9 XGBoost and run 10 BART hold-out results vs baselines; test plan: 38-test pbp suite + 8-test integration suite green>

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```
