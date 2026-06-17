# Situational Play-Call PBP Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add down×distance play-call tendency + play-type-split execution features (and a wp-context snap-share family) to the play-by-play collector, flowing through the existing aggregation → cumulative-average → rank pipeline.

**Architecture:** Mirror the existing directional-bucket family in `src/data/play_by_play/collect.py` — constants + a per-play bucket-assignment helper + per-play `(numerator, denominator)` component columns + registered `(rate, num, den)` tuples. Ranking and the 4-perspective duplication happen automatically by column-naming convention. The snap-share family is a small dedicated block (cross-context ratio) in the cumulative-rate function.

**Tech Stack:** Python, pandas, nfl_data_py; unittest (`python -m unittest`, no pytest).

---

## Conventions

- Repo root **ROOT** = `/Users/joshuaharkness/ClaudeProjects/nfl-predictions`; `cd` there first.
- **Env Python**: `/opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/python` (bare `python3` lacks pandas).
- **Run a test file** (no `__init__.py` in tests → use dotted module path, not `discover -p`):
  ```bash
  cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
  PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/python \
    -m unittest tests.data.play_by_play.test_collect -v
  ```
- Target module: `src/data/play_by_play/collect.py`. Tests: `tests/data/play_by_play/test_collect.py` (synthetic `make_play(...)` row helper; `_aggregate_play_components` / `_add_cumulative_rate_columns` are unit-tested directly).
- **No cache-version constant exists.** Cache invalidation is manual: `get_play_by_play_features(years, refresh=True)` re-downloads + re-aggregates. The new columns appear only after a refresh (Task 5).

## File Structure

| File | Change |
|---|---|
| `src/data/play_by_play/collect.py` | Add `ydstogo`/`goal_to_go` to `REQUIRED_PBP_COLUMNS`; situational constants + `_assign_situational_bucket`; per-play components in `_aggregate_play_components`; register rates; snap-share block in `_add_cumulative_rate_columns` |
| `tests/data/play_by_play/test_collect.py` | Extend `make_play` with `ydstogo`/`goal_to_go`; add bucket / component / rate / snap-share tests |
| `data/play_by_play/aggregated/*.parquet` | Regenerated (Task 5, `refresh=True`) |
| `data/predict_games/input_data/schedule_and_weekly.parquet`, `.../model_features_in/xgb_features_list.csv` | Rebuilt (Task 5) |

---

## Task 1: Situational constants, raw columns, and bucket helper

**Files:** Modify `src/data/play_by_play/collect.py`; modify `tests/data/play_by_play/test_collect.py`

- [ ] **Step 1: Extend the test `make_play` helper** so synthetic rows carry the new fields. In `tests/data/play_by_play/test_collect.py`, add `ydstogo=10, goal_to_go=0` to the `make_play` signature and to its returned dict (`'ydstogo': ydstogo, 'goal_to_go': goal_to_go`).

- [ ] **Step 2: Write the failing test** — append a new class:
```python
class TestAssignSituationalBucket(unittest.TestCase):

    def _bucket(self, **kw):
        plays = pd.DataFrame([collect.make_play(**kw)]) if False else None
        # build directly from kwargs via a one-row frame
        import pandas as pd
        row = {'down': kw['down'], 'ydstogo': kw['ydstogo'], 'goal_to_go': kw.get('goal_to_go', 0)}
        return collect._assign_situational_bucket(pd.DataFrame([row])).iloc[0]

    def test_first_down_single_bucket(self):
        self.assertEqual(self._bucket(down=1, ydstogo=10), 'down1')
        self.assertEqual(self._bucket(down=1, ydstogo=4), 'down1')   # 1st-and-short still down1

    def test_goal_to_go_overrides_distance_and_down(self):
        self.assertEqual(self._bucket(down=1, ydstogo=3, goal_to_go=1), 'goalToGo')
        self.assertEqual(self._bucket(down=3, ydstogo=1, goal_to_go=1), 'goalToGo')

    def test_down2_distance_bins_and_boundaries(self):
        self.assertEqual(self._bucket(down=2, ydstogo=2), 'down2_short')   # <=2 short
        self.assertEqual(self._bucket(down=2, ydstogo=3), 'down2_med')     # 3..6 medium
        self.assertEqual(self._bucket(down=2, ydstogo=6), 'down2_med')
        self.assertEqual(self._bucket(down=2, ydstogo=7), 'down2_long')    # >=7 long

    def test_down3_distance_bins(self):
        self.assertEqual(self._bucket(down=3, ydstogo=1), 'down3_short')
        self.assertEqual(self._bucket(down=3, ydstogo=5), 'down3_med')
        self.assertEqual(self._bucket(down=3, ydstogo=12), 'down3_long')

    def test_fourth_down_gets_no_bucket(self):
        self.assertIsNone(self._bucket(down=4, ydstogo=1))
```

- [ ] **Step 3: Run it, verify it fails**
```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/python \
  -m unittest tests.data.play_by_play.test_collect -v
```
Expected: `AttributeError: module ... has no attribute '_assign_situational_bucket'`.

- [ ] **Step 4: Implement** in `collect.py`:
  - Add to `REQUIRED_PBP_COLUMNS` (the list at line ~36): `'ydstogo'`, `'goal_to_go'`.
  - Add constants near the directional-bucket block (after line ~95):
```python
# Situational play-call buckets: down x distance for downs 2/3 (1st down is ~always
# 1st-and-10, so it is a single bucket; goal_to_go overrides distance; 4th down excluded).
SITUATIONAL_SHORT_MAX = 2          # short  = ydstogo <= 2
SITUATIONAL_MEDIUM_MAX = 6         # medium = 3..6 ; long = >= 7
SITUATIONAL_BUCKETS = [
    'down1',
    'down2_short', 'down2_med', 'down2_long',
    'down3_short', 'down3_med', 'down3_long',
    'goalToGo',
]
_SITUATIONAL_DOWN3 = {'down3_short', 'down3_med', 'down3_long'}

SITUATIONAL_COMPONENT_COLUMNS = []
for _b in SITUATIONAL_BUCKETS:
    SITUATIONAL_COMPONENT_COLUMNS += [f'{_b}_play_count', f'{_b}_pass_count', f'{_b}_run_count']
    if _b in _SITUATIONAL_DOWN3:
        SITUATIONAL_COMPONENT_COLUMNS += [f'{_b}_pass_conversion_sum', f'{_b}_run_conversion_sum']
    else:
        SITUATIONAL_COMPONENT_COLUMNS += [f'{_b}_pass_success_sum', f'{_b}_run_success_sum']

SITUATIONAL_RATE_METRICS = []
for _b in SITUATIONAL_BUCKETS:
    SITUATIONAL_RATE_METRICS.append((f'{_b}_pass_rate', f'{_b}_pass_count', f'{_b}_play_count'))
    if _b in _SITUATIONAL_DOWN3:
        SITUATIONAL_RATE_METRICS.append((f'{_b}_conversion_rate_pass', f'{_b}_pass_conversion_sum', f'{_b}_pass_count'))
        SITUATIONAL_RATE_METRICS.append((f'{_b}_conversion_rate_run', f'{_b}_run_conversion_sum', f'{_b}_run_count'))
    else:
        SITUATIONAL_RATE_METRICS.append((f'{_b}_success_rate_pass', f'{_b}_pass_success_sum', f'{_b}_pass_count'))
        SITUATIONAL_RATE_METRICS.append((f'{_b}_success_rate_run', f'{_b}_run_success_sum', f'{_b}_run_count'))
```
  - Add the helper (near `_assign_run_bucket`, ~line 172):
```python
def _assign_situational_bucket(plays):
    """Label each play with its down x distance situational bucket (or None). goal_to_go
    overrides down/distance; 1st down is a single bucket; 4th down is unlabeled. Distance
    bins: short <= 2, medium 3..6, long >= 7. Plays outside any bucket still count in the
    aggregate Phase 1/2/3 metrics."""
    bucket = pd.Series(None, index=plays.index, dtype='object')
    goal = plays['goal_to_go'] == 1
    bucket[goal] = 'goalToGo'
    rest = ~goal
    down = plays['down']
    ytg = plays['ydstogo']
    bucket[rest & (down == 1)] = 'down1'
    for d, prefix in ((2, 'down2'), (3, 'down3')):
        sel = rest & (down == d)
        bucket[sel & (ytg <= SITUATIONAL_SHORT_MAX)] = f'{prefix}_short'
        bucket[sel & (ytg > SITUATIONAL_SHORT_MAX) & (ytg <= SITUATIONAL_MEDIUM_MAX)] = f'{prefix}_med'
        bucket[sel & (ytg > SITUATIONAL_MEDIUM_MAX)] = f'{prefix}_long'
    return bucket
```

- [ ] **Step 5: Run tests, verify pass** (same command as Step 3). Expected: the 5 new tests + all existing tests pass.

- [ ] **Step 6: Commit**
```bash
git add src/data/play_by_play/collect.py tests/data/play_by_play/test_collect.py
git commit -m "PBP: situational play-call buckets + ydstogo/goal_to_go columns + assignment helper"
```

---

## Task 2: Per-play situational components

**Files:** Modify `src/data/play_by_play/collect.py`; modify `tests/data/play_by_play/test_collect.py`

- [ ] **Step 1: Write the failing test** — append:
```python
class TestSituationalComponents(unittest.TestCase):

    def setUp(self):
        # all competitive (wp=0.5). 3rd-and-short: one converted pass, one stuffed run.
        # 2nd-and-long: one successful pass. 1st down: one failed run.
        self.agg = collect._aggregate_play_components(pd.DataFrame([
            make_play(play_id=1, is_pass=1, down=3, ydstogo=1, third_down_converted=1.0, success=1.0, epa=0.4),
            make_play(play_id=2, is_rush=1, down=3, ydstogo=1, third_down_converted=0.0, success=0.0, epa=-0.3),
            make_play(play_id=3, is_pass=1, down=2, ydstogo=9, success=1.0, epa=0.5),
            make_play(play_id=4, is_rush=1, down=1, ydstogo=10, success=0.0, epa=-0.1),
        ]))

    def _val(self, col):
        return self.agg.loc[self.agg[collect.CONTEXT_COL] == collect.COMPETITIVE, col].iloc[0]

    def test_bucket_play_pass_run_counts(self):
        self.assertEqual(self._val('down3_short_play_count'), 2)
        self.assertEqual(self._val('down3_short_pass_count'), 1)
        self.assertEqual(self._val('down3_short_run_count'), 1)
        self.assertEqual(self._val('down1_play_count'), 1)
        self.assertEqual(self._val('down2_long_pass_count'), 1)

    def test_down3_conversion_components_split_by_play_type(self):
        self.assertEqual(self._val('down3_short_pass_conversion_sum'), 1)  # the converted pass
        self.assertEqual(self._val('down3_short_run_conversion_sum'), 0)   # stuffed run

    def test_non_down3_uses_success_components(self):
        self.assertEqual(self._val('down2_long_pass_success_sum'), 1)
        self.assertEqual(self._val('down1_run_success_sum'), 0)

    def test_play_count_equals_pass_plus_run(self):
        for b in collect.SITUATIONAL_BUCKETS:
            self.assertEqual(self._val(f'{b}_play_count'),
                             self._val(f'{b}_pass_count') + self._val(f'{b}_run_count'))
```

- [ ] **Step 2: Run it, verify it fails** (same command). Expected: `KeyError: 'down3_short_play_count'` (component columns not produced yet).

- [ ] **Step 3: Implement** in `_aggregate_play_components` (after the directional-bucket loop ends, ~line 260, before the `plays['sack_count'] = ...` block):
```python
    situational_bucket = _assign_situational_bucket(plays)
    is_pass = plays['pass'] == 1
    is_rush = plays['rush'] == 1
    is_success = plays['success'] == 1
    converted = plays['third_down_converted'].fillna(0) == 1
    for bucket in SITUATIONAL_BUCKETS:
        in_bucket = situational_bucket == bucket
        plays[f'{bucket}_play_count'] = in_bucket.astype(int)
        plays[f'{bucket}_pass_count'] = (in_bucket & is_pass).astype(int)
        plays[f'{bucket}_run_count'] = (in_bucket & is_rush).astype(int)
        if bucket in _SITUATIONAL_DOWN3:
            plays[f'{bucket}_pass_conversion_sum'] = (in_bucket & is_pass & converted).astype(int)
            plays[f'{bucket}_run_conversion_sum'] = (in_bucket & is_rush & converted).astype(int)
        else:
            plays[f'{bucket}_pass_success_sum'] = (in_bucket & is_pass & is_success).astype(int)
            plays[f'{bucket}_run_success_sum'] = (in_bucket & is_rush & is_success).astype(int)
```
  - Extend `PLAY_COMPONENT_COLUMNS` (line ~149) to append `+ SITUATIONAL_COMPONENT_COLUMNS`:
```python
] + DIRECTIONAL_COMPONENT_COLUMNS + PHASE3_PLAY_COMPONENT_COLUMNS + SITUATIONAL_COMPONENT_COLUMNS
```

- [ ] **Step 4: Run tests, verify pass** (same command). Expected: PASS, including existing `TestAggregatePlayComponents`.

- [ ] **Step 5: Commit**
```bash
git add src/data/play_by_play/collect.py tests/data/play_by_play/test_collect.py
git commit -m "PBP: per-play situational components (tendency + play-type-split execution)"
```

---

## Task 3: Register situational rates

**Files:** Modify `src/data/play_by_play/collect.py`; modify `tests/data/play_by_play/test_collect.py`

- [ ] **Step 1: Write the failing test** — append:
```python
class TestSituationalRates(unittest.TestCase):

    def test_situational_rate_columns_exist_off_and_def(self):
        # one team-week of competitive plays so cumulative rates are well-defined
        components = collect._aggregate_season(pd.DataFrame([
            make_play(play_id=1, is_pass=1, down=3, ydstogo=1, third_down_converted=1.0, success=1.0, epa=0.4, wp=0.5),
            make_play(play_id=2, is_rush=1, down=3, ydstogo=1, third_down_converted=0.0, success=0.0, epa=-0.2, wp=0.5),
            make_play(play_id=3, is_pass=1, down=2, ydstogo=9, success=1.0, epa=0.5, wp=0.5),
        ]))
        components = collect._add_game_count_columns(components).reset_index(drop=True)
        df = collect._add_cumulative_rate_columns(components)
        # tendency + play-type-split execution, offense and defense, competitive context
        for col in [
            'off_down3_short_pass_rate_competitive_cumulative_average',
            'off_down3_short_conversion_rate_pass_competitive_cumulative_average',
            'off_down3_short_conversion_rate_run_competitive_cumulative_average',
            'off_down2_long_success_rate_pass_competitive_cumulative_average',
            'def_opp_down3_short_pass_rate_competitive_cumulative_average',
        ]:
            self.assertIn(col, df.columns)

    def test_situational_rate_values(self):
        components = collect._aggregate_season(pd.DataFrame([
            make_play(play_id=1, is_pass=1, down=3, ydstogo=1, third_down_converted=1.0, success=1.0, epa=0.4, wp=0.5),
            make_play(play_id=2, is_rush=1, down=3, ydstogo=1, third_down_converted=0.0, success=0.0, epa=-0.2, wp=0.5),
        ]))
        components = collect._add_game_count_columns(components).reset_index(drop=True)
        df = collect._add_cumulative_rate_columns(components)
        row = df.iloc[0]
        # 3rd-and-short: 1 pass of 2 plays -> pass_rate 0.5; the pass converted -> 1.0
        self.assertAlmostEqual(row['off_down3_short_pass_rate_competitive_cumulative_average'], 0.5)
        self.assertAlmostEqual(row['off_down3_short_conversion_rate_pass_competitive_cumulative_average'], 1.0)
        self.assertAlmostEqual(row['off_down3_short_conversion_rate_run_competitive_cumulative_average'], 0.0)
```

- [ ] **Step 2: Run it, verify it fails** (same command). Expected: the rate columns are missing (`AssertionError` / `KeyError`).

- [ ] **Step 3: Implement** — extend the `RATE_METRICS` list (line ~169) to append `+ SITUATIONAL_RATE_METRICS`:
```python
] + DIRECTIONAL_RATE_METRICS + PHASE3_RATE_METRICS + PENALTY_RATE_METRICS + PACE_RATE_METRICS + SITUATIONAL_RATE_METRICS
```
(No other change: the `_add_cumulative_rate_columns` loop already iterates `RATE_METRICS × WP_CONTEXTS` building `off_*` and `def_opp_*` cumulative averages, and `get_play_by_play_features` auto-ranks every `*_cumulative_average` column by naming convention.)

- [ ] **Step 4: Run tests, verify pass** (same command). Expected: PASS, including existing `test_all_rate_metric_columns_exist`.

- [ ] **Step 5: Commit**
```bash
git add src/data/play_by_play/collect.py tests/data/play_by_play/test_collect.py
git commit -m "PBP: register situational rate metrics (auto-ranked, 4 perspectives x 3 contexts)"
```

---

## Task 4: wp-context snap-share family

**Files:** Modify `src/data/play_by_play/collect.py`; modify `tests/data/play_by_play/test_collect.py`

- [ ] **Step 1: Write the failing test** — append:
```python
class TestSnapShare(unittest.TestCase):

    def test_snap_share_sums_to_one_and_matches_context_mix(self):
        # 3 competitive plays, 1 garbage_leading play -> competitive share 0.75
        components = collect._aggregate_season(pd.DataFrame([
            make_play(play_id=1, is_pass=1, epa=0.1, wp=0.5),
            make_play(play_id=2, is_rush=1, epa=0.1, wp=0.5),
            make_play(play_id=3, is_pass=1, epa=0.1, wp=0.5),
            make_play(play_id=4, is_rush=1, epa=0.1, wp=0.99),  # garbage_leading
        ]))
        components = collect._add_game_count_columns(components).reset_index(drop=True)
        df = collect._add_cumulative_rate_columns(components)
        row = df.iloc[0]
        shares = [
            row['off_snap_share_competitive_cumulative_average'],
            row['off_snap_share_garbage_leading_cumulative_average'],
            row['off_snap_share_garbage_trailing_cumulative_average'],
        ]
        self.assertAlmostEqual(row['off_snap_share_competitive_cumulative_average'], 0.75)
        self.assertAlmostEqual(row['off_snap_share_garbage_leading_cumulative_average'], 0.25)
        self.assertAlmostEqual(sum(shares), 1.0)

    def test_defense_snap_share_context_swapped(self):
        # the offense's garbage_leading play is the opponent-defense's garbage_trailing snap
        components = collect._aggregate_season(pd.DataFrame([
            make_play(play_id=1, is_pass=1, epa=0.1, wp=0.5),
            make_play(play_id=2, is_rush=1, epa=0.1, wp=0.99),
        ]))
        components = collect._add_game_count_columns(components).reset_index(drop=True)
        df = collect._add_cumulative_rate_columns(components)
        row = df.iloc[0]
        self.assertAlmostEqual(row['def_opp_snap_share_garbage_trailing_cumulative_average'], 0.5)
```

- [ ] **Step 2: Run it, verify it fails** (same command). Expected: snap-share columns missing.

- [ ] **Step 3: Implement** — in `_add_cumulative_rate_columns`, after the `for metric, numerator, denominator in RATE_METRICS:` loop completes and before `return pd.concat(...)` (line ~490), add:
```python
    # wp-context snap-share: a CROSS-context ratio (each context's play_count over the
    # season-to-date total across all contexts), so it cannot be a within-context
    # RATE_METRICS entry. Named *_cumulative_average so it is auto-ranked like the rest.
    off_total_plays = sum(off_cumulative[f'play_count_{c}'] for c in WP_CONTEXTS)
    def_total_plays = sum(def_cumulative[f'play_count_{c}'] for c in WP_CONTEXTS)
    for context in WP_CONTEXTS:
        rate_columns[f'off_snap_share_{context}_cumulative_average'] = (
            off_cumulative[f'play_count_{context}'] / off_total_plays.where(off_total_plays != 0)
        )
        defense_context = DEFENSE_CONTEXT_SWAP[context]
        rate_columns[f'def_opp_snap_share_{defense_context}_cumulative_average'] = (
            def_cumulative[f'play_count_{context}'] / def_total_plays.where(def_total_plays != 0)
        )
```

- [ ] **Step 4: Run tests, verify pass** (same command). Expected: PASS.

- [ ] **Step 5: Run the WHOLE pbp test file once more** to confirm no regressions (same command). Expected: all classes pass.

- [ ] **Step 6: Commit**
```bash
git add src/data/play_by_play/collect.py tests/data/play_by_play/test_collect.py
git commit -m "PBP: wp-context snap-share family (cross-context, auto-ranked, context-swapped on defense)"
```

---

## Task 5: Regenerate caches + rebuild model dataset

**Files:** regenerates `data/play_by_play/aggregated/*.parquet`, `data/predict_games/input_data/schedule_and_weekly.parquet`, `data/predict_games/model_features_in/xgb_features_list.csv`

> **Network + slow + coordinate with the user.** This re-downloads ~20 seasons of nflverse PBP and rebuilds the merged dataset. There are pre-existing uncommitted local edits to `src/data/collect_all.py` and `scripts/data_assembly/predict_game_winner/schedule_and_weekly.py` — confirm with the user whether those should be in place before rebuilding (they feed the same pipeline).

- [ ] **Step 1: Smoke-test one season end-to-end (fast, still downloads one year)** — confirms the new columns materialize through `get_play_by_play_features`:
```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/python -c "
from src.data.play_by_play import collect
df = collect.get_play_by_play_features([2023], refresh=True)
sit = [c for c in df.columns if 'down3_short_conversion_rate_pass' in c]
share = [c for c in df.columns if 'snap_share' in c]
print('situational sample cols:', len(sit), sit[:4])
print('snap_share cols:', len(share), share[:4])
print('have rank + rank_change:', any(c.endswith('_rank') for c in df.columns) and any(c.endswith('_rank_change') for c in df.columns))
"
```
Expected: nonzero situational + snap-share columns, each with `_rank` / `_rank_change` siblings.

- [ ] **Step 2: Identify and run the dataset-build entry point.** Locate how `schedule_and_weekly.parquet` is produced:
```bash
grep -rn "schedule_and_weekly.parquet\|get_play_by_play_features\|to_parquet" scripts/ src/data/collect_all.py | head
```
Run that build (the `scripts/data_assembly/predict_game_winner/schedule_and_weekly.py` assembly, with PBP `refresh=True` so all season caches regenerate). Use the env Python with `PYTHONPATH=.`. Capture stdout to a log; expect it to download every season once.

- [ ] **Step 3: Regenerate the candidate feature list.** Confirm how `xgb_features_list.csv` is generated (it enumerates the dataset's feature columns) and rebuild it so the new situational + snap-share columns are present:
```bash
PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/python -c "
import pandas as pd
feats = pd.read_csv('data/predict_games/model_features_in/xgb_features_list.csv')
n_sit = feats['feature'].str.contains('down3_short|down2_long|goalToGo|snap_share').sum()
print('feature list rows:', len(feats), '| new-family rows:', int(n_sit))
"
```
Expected: row count grew (~2,349 → ~3,200+); new-family rows > 0.

- [ ] **Step 4: Verify the model input carries the new columns**
```bash
PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/python -c "
import pandas as pd
df = pd.read_parquet('data/predict_games/input_data/schedule_and_weekly.parquet')
cols = [c for c in df.columns if 'down3_short_conversion_rate_pass' in c or 'snap_share' in c]
print('situational/snap-share columns in model input:', len(cols))
print('rows:', len(df))
"
```
Expected: columns present (across off/def × target/opp perspectives from collect_all's duplication); row count unchanged from before.

- [ ] **Step 5: Commit the regenerated artifacts** (large binary parquet — confirm with the user before committing data files, per repo convention):
```bash
git add data/play_by_play/aggregated/ data/predict_games/input_data/schedule_and_weekly.parquet \
        data/predict_games/model_features_in/xgb_features_list.csv
git commit -m "Regenerate PBP caches + model dataset with situational play-call + snap-share features"
```

---

## Out of scope (separate follow-up)

Evaluation run — rank-only RFE + grid/BART on the expanded pool, recorded as the next README run. Not part of this plan; start it once Task 5's dataset is verified.

## Self-Review notes

- **Spec coverage:** raw columns + buckets → Task 1; per-play components (tendency + play-type-split execution) → Task 2; rate registration + auto-rank + 4 perspectives → Task 3; snap-share cross-context family → Task 4; cache regen + dataset rebuild + feature-list → Task 5; eval explicitly deferred. All covered.
- **Naming consistency:** `_assign_situational_bucket`, `SITUATIONAL_BUCKETS`, `_SITUATIONAL_DOWN3`, `SITUATIONAL_COMPONENT_COLUMNS`, `SITUATIONAL_RATE_METRICS`, and the `{bucket}_pass_rate` / `{bucket}_success_rate_pass` / `{bucket}_conversion_rate_pass` / `off_snap_share_{context}` column names are used identically across tasks and match the spec.
- **Placeholders:** none — every code/test step has literal code; Task 5's two "identify the entry point" steps include the discovery `grep` plus exact verification commands (the build entry point genuinely must be read from the repo, not guessed).
- **Spec correction:** the spec said "bump the cache version"; there is no version constant — invalidation is `refresh=True`. Task 5 uses that; update the spec line if desired.
