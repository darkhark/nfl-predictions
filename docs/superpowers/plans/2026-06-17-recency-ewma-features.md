# Recency (EWMA + Rolling) Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add EWMA (halflife 3) and rolling (last 4) recency versions of every cumulative-average stat — weekly box stats, schedule points, PBP rates — alongside the existing season-to-date averages, each ranked, so RFE/BART can choose current-form vs season-average per stat.

**Architecture:** Mirror each module's existing cumulative-average computation with two added recency columns per stat; extend rank discovery (suffix-based, weekly/PBP) and the schedule's bespoke rank logic; generalize the rank-only filter's raw-value marker test. Recency is derived post-cache, so regeneration is a dataset rebuild (no PBP re-download).

**Tech Stack:** Python, pandas (`ewm`, `rolling`); unittest (`python -m unittest`, no pytest).

---

## Conventions

- Repo root **ROOT** = `/Users/joshuaharkness/ClaudeProjects/nfl-predictions`; `cd` there first.
- **Env Python**: `/opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/python` (bare `python3` lacks pandas).
- **Run a test file** (tests/ has no `__init__.py` → dotted module path, not `discover -p`):
  ```bash
  cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
  PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/python \
    -m unittest tests.data.<pkg>.test_collect -v
  ```
- **Recency parameters** (use everywhere): `EWMA_HALFLIFE = 3`, `ROLLING_WINDOW = 4`, `min_periods=1`, `adjust=True`.
- **Naming** (must contain the markers `ewma` / `rolling` so the rank-only filter catches them):
  weekly/PBP `{...}_ewma_average`, `{...}_rolling_average` (+ `_rank`, `_rank_change`);
  schedule `{...}_ewma_avg_score` / `..._rolling_avg_score` (mirroring `cumulative_avg_score`).
- No `_change` variants for recency families (YAGNI).

## File Structure

| File | Change |
|---|---|
| `src/data/weekly/collect.py` | recency cols in `create_cumulative_columns`; extend `add_rank_columns` discovery |
| `src/data/schedule/collect.py` | recency avg-score in `_calculate_cumulative_avg_score`; ranks in `_add_score_rank_columns`; renames in `_merge_team_df` |
| `src/data/play_by_play/collect.py` | recency rate ratios + snap-share in `_add_cumulative_rate_columns`; extend `get_play_by_play_features` discovery |
| `notebooks/.../cross_validation/rfe.ipynb` | generalize the rank-only marker filter (cell `5f1372f2bb7f751e`) |
| tests under `tests/data/{weekly,schedule,play_by_play}/test_collect.py` | recency unit tests |

---

## Task 1: Weekly recency (box stats)

**Files:** Modify `src/data/weekly/collect.py`; Test `tests/data/weekly/test_collect.py`

- [ ] **Step 1: Write the failing test** — append. (Uses the module's existing synthetic-frame
  conventions; if a helper exists, reuse it — otherwise build the small frame inline as below.)
```python
class TestWeeklyRecency(unittest.TestCase):

    def _frame(self):
        # one team, one season, 5 games, ascending team_game_count; a single stat column
        # from ONLY_NON_IDENTIFIER_COLUMNS[5:] to exercise the loop.
        import pandas as pd
        stat = collect.ONLY_NON_IDENTIFIER_COLUMNS[5]
        rows = []
        for i, val in enumerate([10.0, 20.0, 30.0, 40.0, 50.0], start=1):
            r = {c: 0 for c in collect.ONLY_NON_IDENTIFIER_COLUMNS}
            r.update({'team': 'AAA', 'opp_team': 'BBB', 'season': 2023, 'week': i,
                      'team_game_count': i, 'opp_game_count': i, stat: val})
            rows.append(r)
        return pd.DataFrame(rows), stat

    def test_ewma_and_rolling_match_pandas(self):
        import pandas as pd
        df, stat = self._frame()
        out = collect.create_cumulative_columns(df, ['team', 'season'], 'off', 'team_game_count')
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        expected_ewma = s.ewm(halflife=3, adjust=True).mean().to_numpy()
        expected_roll = s.rolling(4, min_periods=1).mean().to_numpy()
        self.assertTrue(
            (abs(out[f'off_{stat}_ewma_average'].to_numpy() - expected_ewma) < 1e-9).all())
        self.assertTrue(
            (abs(out[f'off_{stat}_rolling_average'].to_numpy() - expected_roll) < 1e-9).all())

    def test_recency_resets_across_seasons(self):
        import pandas as pd
        df, stat = self._frame()
        df2 = df.copy(); df2['season'] = 2024; df2['team_game_count'] = range(1, 6)
        both = pd.concat([df, df2], ignore_index=True)
        out = collect.create_cumulative_columns(both, ['team', 'season'], 'off', 'team_game_count')
        # 2024 game 1 ewma == its own value (no carryover from 2023)
        first_2024 = out[(out['season'] == 2024)].sort_values('team_game_count').iloc[0]
        self.assertAlmostEqual(first_2024[f'off_{stat}_ewma_average'], 10.0)
```

- [ ] **Step 2: Run it, verify it fails** (`-m unittest tests.data.weekly.test_collect -v`).
  Expected: `KeyError: 'off_<stat>_ewma_average'`.

- [ ] **Step 3: Implement** in `create_cumulative_columns`, inside the
  `for column in ONLY_NON_IDENTIFIER_COLUMNS[5:]:` loop, after the cumulative-average lines
  (after the `_cumulative_average_change` fillna, before `cols_to_drop.append(column)`):
```python
        grp = df.groupby(groupby_columns)[column]
        new_df[f'{column_prefix}_{column}_ewma_average'] = grp.transform(
            lambda s: s.ewm(halflife=3, adjust=True).mean())
        new_df[f'{column_prefix}_{column}_rolling_average'] = grp.transform(
            lambda s: s.rolling(4, min_periods=1).mean())
```
  (The frame is already `sort_values(groupby_columns + [game_count_col])`, so each group's
  series is in game order; grouping by `(team|opp_team, season)` resets recency per season.)

- [ ] **Step 4: Run tests, verify pass.** Expected: PASS.

- [ ] **Step 5: Extend rank discovery** — in `add_rank_columns`, change both list
  comprehensions' `endswith` to a tuple:
```python
    _avg_suffixes = ('_cumulative_average', '_ewma_average', '_rolling_average')
    off_cols = [col for col in df.columns if col.startswith('off_') and col.endswith(_avg_suffixes)]
    def_cols = [col for col in df.columns if col.startswith('def_opp_') and col.endswith(_avg_suffixes)]
    return transformations.add_rank_and_rank_change_columns(df, off_cols, def_cols)
```
  Add a test asserting recency rank columns appear:
```python
    def test_recency_columns_get_ranked(self):
        import pandas as pd
        df, stat = self._frame()
        out = collect.create_cumulative_columns(df, ['team', 'season'], 'off', 'team_game_count')
        out = collect.add_rank_columns(out)
        self.assertIn(f'off_{stat}_ewma_average_rank', out.columns)
        self.assertIn(f'off_{stat}_rolling_average_rank_change', out.columns)
```
  Run tests, verify pass.

- [ ] **Step 6: Commit**
```bash
git add src/data/weekly/collect.py tests/data/weekly/test_collect.py
git commit -m "Weekly: EWMA + rolling recency versions of cumulative-average box stats (ranked)"
```

---

## Task 2: Schedule recency (points scored/allowed)

**Files:** Modify `src/data/schedule/collect.py`; Test `tests/data/schedule/test_collect.py`

The schedule module is bespoke: `_calculate_cumulative_avg_score` builds
`cumulative_avg_{points}`, `_add_score_rank_columns` ranks it, `_merge_team_df` renames to
`{team_type}_{side}_cumulative_avg_{points}`. Mirror each for `ewma_avg` / `rolling_avg`.

- [ ] **Step 1: Write the failing test** — append (follow the file's existing synthetic-frame
  pattern; build a 4-game one-team frame):
```python
class TestScheduleRecency(unittest.TestCase):

    def test_ewma_rolling_avg_score(self):
        import pandas as pd
        team_df = pd.DataFrame({
            'team': ['AAA'] * 4, 'season': [2023] * 4, 'week': [1, 2, 3, 4],
            'score': [10.0, 20.0, 30.0, 40.0],
        })
        out = collect._calculate_cumulative_avg_score(team_df, offense=True)
        s = pd.Series([10.0, 20.0, 30.0, 40.0])
        self.assertTrue((abs(out['ewma_avg_score'].to_numpy()
                             - s.ewm(halflife=3, adjust=True).mean().to_numpy()) < 1e-9).all())
        self.assertTrue((abs(out['rolling_avg_score'].to_numpy()
                             - s.rolling(4, min_periods=1).mean().to_numpy()) < 1e-9).all())
```

- [ ] **Step 2: Run it, verify it fails** (`-m unittest tests.data.schedule.test_collect -v`).
  Expected: `KeyError: 'ewma_avg_score'`.

- [ ] **Step 3: Implement** in `_calculate_cumulative_avg_score`, after the existing
  `cumulative_avg_{points}` lines (use the existing `points` variable):
```python
    team_df[f'ewma_avg_{points}'] = team_df.groupby(['team', 'season'])['score'].transform(
        lambda s: s.ewm(halflife=3, adjust=True).mean())
    team_df[f'rolling_avg_{points}'] = team_df.groupby(['team', 'season'])['score'].transform(
        lambda s: s.rolling(4, min_periods=1).mean())
```
  Run tests, verify pass.

- [ ] **Step 4: Rank the recency families** — in `_add_score_rank_columns`, mirror the
  existing `cumulative_avg_{points}` rank/rank_change block for `ewma_avg_{points}` and
  `rolling_avg_{points}`. For each `base` in `(f'ewma_avg_{points}', f'rolling_avg_{points}')`:
```python
    for base in (f'ewma_avg_{points}', f'rolling_avg_{points}'):
        ascending = not offense   # offense ranked high-to-low, defense low-to-high (match existing)
        team_df[f'{base}_rank'] = team_df.groupby(['season', 'week'])[base].rank(
            ascending=ascending, method='min')
        team_df[f'{base}_rank_change'] = team_df.groupby(['team', 'season'])[f'{base}_rank'].diff()
        team_df.loc[team_df.groupby(['team', 'season']).cumcount() == 0, f'{base}_rank_change'] = 0
```
  (Match the exact `ascending` orientation the existing `cumulative_avg` rank uses — read
  `_add_score_rank_columns` and copy its direction.) Add a test that `ewma_avg_score_rank`
  exists after `_add_score_rank_columns`. Run, verify pass.

- [ ] **Step 5: Carry the new columns through the merge** — in `_merge_team_df`, extend the
  rename dict to include the recency families so they become
  `{team_type}_{side}_ewma_avg_{score}` etc.:
```python
        f'ewma_avg_{score}': f'{team_type}_{side}_ewma_avg_{score}',
        f'ewma_avg_{score}_rank': f'{team_type}_{side}_ewma_avg_{score}_rank',
        f'ewma_avg_{score}_rank_change': f'{team_type}_{side}_ewma_avg_{score}_rank_change',
        f'rolling_avg_{score}': f'{team_type}_{side}_rolling_avg_{score}',
        f'rolling_avg_{score}_rank': f'{team_type}_{side}_rolling_avg_{score}_rank',
        f'rolling_avg_{score}_rank_change': f'{team_type}_{side}_rolling_avg_{score}_rank_change',
```
  Add a test on the public `add_calculated_values` (or whichever assembles the merged frame)
  asserting a `target_off_ewma_avg_score` column exists. Run, verify pass.

- [ ] **Step 6: Commit**
```bash
git add src/data/schedule/collect.py tests/data/schedule/test_collect.py
git commit -m "Schedule: EWMA + rolling recency avg-score (scored/allowed), ranked and merged"
```

---

## Task 3: PBP recency (rate ratios + snap-share)

**Files:** Modify `src/data/play_by_play/collect.py`; Test `tests/data/play_by_play/test_collect.py`

- [ ] **Step 1: Write the failing test** — append:
```python
class TestPbpRecency(unittest.TestCase):

    def _components(self):
        # one team, 4 competitive games; play_count and success_sum components only needed
        import pandas as pd
        import numpy as np
        base = {c: 0 for c in collect.COMPONENT_COLUMNS}
        rows = []
        for i, (pc, ss) in enumerate([(50, 25), (40, 28), (60, 30), (30, 9)], start=1):
            r = {**base, 'team': 'AAA', 'opp_team': 'BBB', 'season': 2023, 'week': i,
                 'season_type': 'REG', 'team_game_count': i, 'opp_game_count': i}
            r['play_count_competitive'] = pc
            r['success_sum_competitive'] = ss
            # zero the other two contexts for these two components
            r['play_count_garbage_leading'] = 0; r['play_count_garbage_trailing'] = 0
            r['success_sum_garbage_leading'] = 0; r['success_sum_garbage_trailing'] = 0
            rows.append(r)
        df = pd.DataFrame(rows)
        # ensure every component_context column exists (others all zero)
        for comp in collect.COMPONENT_COLUMNS:
            for ctx in collect.WP_CONTEXTS:
                df.setdefault(f'{comp}_{ctx}', 0)
        return df

    def test_ewma_and_rolling_success_rate(self):
        import pandas as pd
        df = collect._add_cumulative_rate_columns(self._components())
        num = pd.Series([25.0, 28, 30, 9]); den = pd.Series([50.0, 40, 60, 30])
        ewma_expected = (num.ewm(halflife=3, adjust=True).mean()
                         / den.ewm(halflife=3, adjust=True).mean()).to_numpy()
        roll_expected = (num.rolling(4, min_periods=1).sum()
                         / den.rolling(4, min_periods=1).sum()).to_numpy()
        got_ewma = df.sort_values('team_game_count')['off_success_rate_competitive_ewma_average'].to_numpy()
        got_roll = df.sort_values('team_game_count')['off_success_rate_competitive_rolling_average'].to_numpy()
        self.assertTrue((abs(got_ewma - ewma_expected) < 1e-9).all())
        self.assertTrue((abs(got_roll - roll_expected) < 1e-9).all())
```
  (If `_add_cumulative_rate_columns` requires game-count columns / unique index, call
  `_add_game_count_columns(...).reset_index(drop=True)` first as the existing rate tests do.)

- [ ] **Step 2: Run it, verify it fails** (`-m unittest tests.data.play_by_play.test_collect -v`).
  Expected: `KeyError: 'off_success_rate_competitive_ewma_average'`.

- [ ] **Step 3: Implement** in `_add_cumulative_rate_columns`. After the existing
  `off_cumulative` / `def_cumulative` cumsum frames are built, add recency frames (same
  grouping + ordering the cumsum frames use):
```python
    off_ewm = df.groupby([TEAM_COL, SEASON_COL])[component_cols].transform(
        lambda s: s.ewm(halflife=3, adjust=True).mean())
    off_roll = df.groupby([TEAM_COL, SEASON_COL])[component_cols].transform(
        lambda s: s.rolling(4, min_periods=1).sum())
    def_ewm = opp_ordered.groupby([OPPONENT_TEAM_COL, SEASON_COL])[component_cols].transform(
        lambda s: s.ewm(halflife=3, adjust=True).mean())
    def_roll = opp_ordered.groupby([OPPONENT_TEAM_COL, SEASON_COL])[component_cols].transform(
        lambda s: s.rolling(4, min_periods=1).sum())
```
  Then inside the existing `for metric, numerator, denominator in RATE_METRICS:` /
  `for context in WP_CONTEXTS:` loop, after the cumulative-average assignments, add (offense
  and defense, mirroring the cumulative direction + `DEFENSE_CONTEXT_SWAP`):
```python
            for tag, off_frame, def_frame in (('ewma', off_ewm, off_ewm),
                                              ('rolling', off_roll, off_roll)):
                pass  # see explicit block below
```
  Concretely (write it out, no loop indirection):
```python
            # EWMA recency (ratio of EWMA-weighted components; shared weights cancel)
            off_n = off_ewm[f'{numerator}_{context}']; off_d = off_ewm[f'{denominator}_{context}']
            rate_columns[f'off_{metric}_{context}_ewma_average'] = off_n / off_d.where(off_d != 0)
            def_n = def_ewm[f'{numerator}_{context}']; def_d = def_ewm[f'{denominator}_{context}']
            rate_columns[f'def_opp_{metric}_{defense_context}_ewma_average'] = def_n / def_d.where(def_d != 0)
            # Rolling recency (ratio of last-4 component sums = last-4-game rate)
            off_n = off_roll[f'{numerator}_{context}']; off_d = off_roll[f'{denominator}_{context}']
            rate_columns[f'off_{metric}_{context}_rolling_average'] = off_n / off_d.where(off_d != 0)
            def_n = def_roll[f'{numerator}_{context}']; def_d = def_roll[f'{denominator}_{context}']
            rate_columns[f'def_opp_{metric}_{defense_context}_rolling_average'] = def_n / def_d.where(def_d != 0)
```
  And mirror the **snap-share** block (added in the situational work) for ewma/rolling — using
  the per-frame play_count totals across contexts:
```python
    for frame, tag in ((off_ewm, 'ewma'), (off_roll, 'rolling')):
        total = sum(frame[f'play_count_{c}'] for c in WP_CONTEXTS)
        for context in WP_CONTEXTS:
            rate_columns[f'off_snap_share_{context}_{tag}_average'] = (
                frame[f'play_count_{context}'] / total.where(total != 0))
    for frame, tag in ((def_ewm, 'ewma'), (def_roll, 'rolling')):
        total = sum(frame[f'play_count_{c}'] for c in WP_CONTEXTS)
        for context in WP_CONTEXTS:
            defense_context = DEFENSE_CONTEXT_SWAP[context]
            rate_columns[f'def_opp_snap_share_{defense_context}_{tag}_average'] = (
                frame[f'play_count_{context}'] / total.where(total != 0))
```
  Run tests, verify pass.

- [ ] **Step 4: Extend rank discovery** — in `get_play_by_play_features`, change the
  `off_cols` / `def_cols` `endswith('_cumulative_average')` to the 3-suffix tuple
  `('_cumulative_average', '_ewma_average', '_rolling_average')` (both lists). Add a test that
  `off_success_rate_competitive_ewma_average_rank` appears in `get_play_by_play_features`
  output for a small synthetic season (follow the existing rank test's construction). Run,
  verify pass.

- [ ] **Step 5: Full pbp suite** (`-m unittest tests.data.play_by_play.test_collect`) — confirm
  no regression (the existing `test_directional_lists_wired_into_aggregates` / feature-count
  assertions may need their magic numbers bumped for the new recency columns; update only the
  numbers, derived consistently, as in the situational work).

- [ ] **Step 6: Commit**
```bash
git add src/data/play_by_play/collect.py tests/data/play_by_play/test_collect.py
git commit -m "PBP: EWMA + rolling recency rate ratios + snap-share (ranked, context-swapped)"
```

---

## Task 4: Generalize the rank-only filter (rfe.ipynb)

**Files:** Modify `notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/rfe.ipynb` (cell `5f1372f2bb7f751e`)

- [ ] **Step 1: Read the notebook** (required before NotebookEdit) and confirm the current
  filter uses `'cumulative' not in f`.

- [ ] **Step 2: Edit the RANK_ONLY filter** (NotebookEdit `cell_id 5f1372f2bb7f751e`) — replace
  the comprehension's condition with the marker-substring rule (keep everything else in the
  cell, including the current `RANK_ONLY` value):
```python
candidate_features = list(MODEL_FEATURES_BEFORE_RFE['feature'])
_RAW_VALUE_MARKERS = ('cumulative', 'ewma', 'rolling')
if RANK_ONLY:
    candidate_features = [
        f for f in candidate_features
        if not any(m in f for m in _RAW_VALUE_MARKERS)
        or f.endswith('_rank') or f.endswith('_rank_change')
    ]
```

- [ ] **Step 3: Regression-verify the filter on the CURRENT feature list** (recency not in the
  data yet, so the rank-only count must be unchanged vs the old `'cumulative'`-only rule):
```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/python -c "
import pandas as pd
feats = list(pd.read_csv('data/predict_games/model_features_in/xgb_features_list.csv')['feature'])
old = [f for f in feats if 'cumulative' not in f or f.endswith('_rank') or f.endswith('_rank_change')]
markers = ('cumulative','ewma','rolling')
new = [f for f in feats if not any(m in f for m in markers) or f.endswith('_rank') or f.endswith('_rank_change')]
print('old rank-only:', len(old), '| new rank-only:', len(new), '| identical:', set(old)==set(new))
"
```
  Expected: identical = True (no `ewma`/`rolling` columns exist yet).

- [ ] **Step 4: Commit**
```bash
git add notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/rfe.ipynb
git commit -m "rfe.ipynb: generalize rank-only filter to drop ewma/rolling raw values too"
```

---

## Task 5: Regenerate dataset + feature list

**Files:** regenerates `data/predict_games/input_data/schedule_and_weekly.parquet`, `.../model_features_in/xgb_features_list.csv` (no PBP cache re-download)

- [ ] **Step 1: Run the full test suite** to confirm all modules green before regenerating:
```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/python -m unittest \
  tests.data.weekly.test_collect tests.data.schedule.test_collect tests.data.play_by_play.test_collect -v 2>&1 | tail -5
```

- [ ] **Step 2: Rebuild the merged dataset** (refetches weekly source, REUSES PBP component
  caches — recency is derived post-cache, so NO `refresh=True`):
```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/python -W ignore \
  scripts/data_assembly/predict_game_winner/schedule_and_weekly.py > /tmp/recency_assembly.out 2>&1
tail -n 3 /tmp/recency_assembly.out   # must NOT contain a Traceback
```

- [ ] **Step 3: Verify the recency families landed + pool size**
```bash
PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/python -W ignore -c "
import pandas as pd
feats = pd.read_csv('data/predict_games/model_features_in/xgb_features_list.csv')['feature']
df = pd.read_parquet('data/predict_games/input_data/schedule_and_weekly.parquet')
print('feature-list rows:', len(feats))
print('ewma cols:', int(feats.str.contains('_ewma_').sum()), '| rolling cols:', int(feats.str.contains('_rolling_').sum()))
print('rank for a recency feat present:', any(feats.str.contains('ewma_average_rank')))
print('parquet rows:', len(df))
"
```
  Expected: pool ~3,249 → ~8,000; ewma + rolling columns present (incl. `_rank` variants);
  parquet rows unchanged.

- [ ] **Step 4: Commit the regenerated artifacts** (confirm with user before committing data
  files, per repo convention):
```bash
git add data/predict_games/input_data/schedule_and_weekly.parquet \
        data/predict_games/model_features_in/xgb_features_list.csv
git commit -m "Regenerate model dataset + feature list with EWMA/rolling recency families (~3,249 -> ~8,000)"
```

---

## Out of scope (separate follow-up)

Evaluation — rank-only RFE + grid/BART on the recency-expanded pool vs the ~0.705 plateau,
recorded as the next README run. Start once Task 5's dataset is verified.

## Self-Review notes

- **Spec coverage:** weekly recency + discovery → Task 1; schedule recency (bespoke calc +
  rank + merge) → Task 2; PBP rate-ratio recency + snap-share + discovery → Task 3; rank-only
  marker filter → Task 4; regen → Task 5; eval deferred. All covered.
- **Naming consistency:** `_ewma_average` / `_rolling_average` (weekly/PBP), `ewma_avg_score` /
  `rolling_avg_score` (schedule) — all contain the markers `ewma`/`rolling` that Task 4's filter
  keys on; `EWMA_HALFLIFE=3` / `ROLLING_WINDOW=4` / `min_periods=1` used identically everywhere.
- **The schedule-vs-suffix subtlety:** rank *discovery* is suffix-based (weekly/PBP columns end
  in `_cumulative_average`/`_ewma_average`/`_rolling_average`); schedule has its own rank logic
  (Task 2 Step 4); the rank-only *filter* is substring-marker-based (Task 4) so it catches the
  schedule `*_avg_score` columns too. These are intentionally different mechanisms.
- **Placeholders:** none — all code literal. (Task 2 Step 4 instructs copying the existing
  `ascending` direction; the engineer reads it from `_add_score_rank_columns` 5 lines above.)
