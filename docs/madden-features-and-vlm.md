# Madden Ratings — Feature Catalog & Validation/Verification Module (VLM) Spec

Companion doc for the `madden_ratings` feature family (PR #39). Two parts:
1. **Catalog** — what every feature is and how it's computed.
2. **VLM spec** — the quality gates the Madden feature block must pass before it reaches
   the model, with thresholds grounded in the actual built dataset
   (`data/predict_games/input_data/schedule_and_weekly.parquet`, 11,898 rows, seasons
   2003–2025).

---

## 1. How to read the 188 columns

The family is **not** 188 distinct ideas. It is **23 base features**, each crossed with
a perspective and a measure, plus 4 matchup deltas:

```
23 base features  ×  2 perspectives (target_, opp_)  ×  4 measures            = 184
+ 4 matchup deltas                                                            =   4
                                                                       TOTAL  = 188
```

**Perspective prefix** — `target_` = the team whose win we're predicting; `opp_` = its
opponent. (Produced by an explicit double-merge in `collect_all.py`, AFTER the one-week
shift, because Madden ratings are pre-game-known season constants — see leakage note below.)

**Measure suffix** (applied to each base, per perspective):

| Suffix | Meaning | Computation |
|---|---|---|
| `_ovr` | the slot/group's Madden overall this game | overall of the starter (slot) or mean overall of the role's starters (group) |
| `_ovr_diff_prev` | change vs the team's **previous game** | `_ovr` − the same slot's `_ovr` one game earlier (by games played, skipping byes) |
| `_ovr_diff_4g` | change vs **4 games** earlier | `_ovr` − the same slot's `_ovr` 4 games earlier |
| `_ovr_diff_prev_season` | change vs **last season's** launch value | `_ovr` − the same player's prior-season launch overall |

Diffs are driven mostly by **starter changes** (injury/trade/benching) since launch
overalls are constant within a season; `_diff_prev_season` captures a player's
year-over-year rise/decline.

**Leakage note:** all inputs are pre-kickoff — starters from the pre-game depth chart
minus Out/Doubtful (next-man-up); overalls are preseason/launch (season-constant);
diffs use only prior games / prior season; the block is merged after `_shift_data`. No
post-kickoff or future information enters any column.

---

## 2. Catalog

### 2a. Granular slots (13) — single starter's overall

The starter at each slot is the weekly depth-chart rank-1 (minus Out/Doubtful), with the
**side** resolved from the player's Madden position (the depth chart is side-less).

| Base feature | Slot | Notes |
|---|---|---|
| `madden_qb` | Starting QB | the dominant single signal |
| `madden_rb` | Lead RB | highest-overall starting back |
| `madden_wr1` / `wr2` / `wr3` | WRs by overall (desc) | top 3 starting receivers, ranked by overall |
| `madden_te` | TE1 | top starting tight end |
| `madden_lt` / `lg` / `c` / `rg` / `rt` | O-line, by position | side/interior from Madden position |
| `madden_edge_left` / `edge_right` | Edge rushers by side | the "good edge vs weak tackle" matchup inputs |

### 2b. Group means (10) — mean overall of a role's starters

Roles are assigned by the ported pre-EDGE attribute classifier (edge vs interior-DL vs
off-ball-LB by weight + power/finesse moves), so they're scheme/era-invariant.

| Base feature | Group | Members |
|---|---|---|
| `madden_backfield` | Backfield | RB + FB starters |
| `madden_receivers` | Receivers | top-3 WR starters |
| `madden_tight_end` | Tight ends | TE starters |
| `madden_interior_ol` | Interior OL | LG, C, RG |
| `madden_exterior_ol` | Exterior OL | LT, RT |
| `madden_edge` | Edge | all edge rushers (L+R) |
| `madden_interior_dl` | Interior DL | DTs/NTs |
| `madden_linebacker` | Off-ball LB | MIKE/WILL/SAM/MLB (non-edge) |
| `madden_cornerback` | Cornerbacks | CB starters |
| `madden_safety` | Safeties | FS + SS |

### 2c. Matchup deltas (4) — directional, one value per row (no perspective/measure split)

| Column | Formula | Reads as |
|---|---|---|
| `madden_matchup_pass_pro` | `target_exterior_ol_ovr − opp_edge_ovr` | can our tackles handle their edge rush? |
| `madden_matchup_interior` | `target_interior_ol_ovr − opp_interior_dl_ovr` | interior O-line vs interior D-line |
| `madden_matchup_skill_cover` | `target_receivers_ovr − (opp_cornerback_ovr + opp_safety_ovr)/2` | our pass-catchers vs their coverage |
| `madden_matchup_pass_rush` | `opp_exterior_ol_ovr − target_edge_ovr` | our edge rush vs their tackles |

### 2d. Worked example — `KC @ BAL`, 2024 Week 1, target = KC

| Column | Value | Reading |
|---|---|---|
| `target_madden_qb_ovr` | 99.0 | Mahomes |
| `target_madden_edge_left_ovr` | 81.0 | KC's left edge |
| `target_madden_exterior_ol_ovr` | 73.0 | KC tackles (mean) |
| `opp_madden_edge_ovr` | 80.0 | BAL edge group |
| `madden_matchup_pass_pro` | −7.0 | 73 − 80: KC tackles slightly outclassed by BAL edge |
| `target_madden_qb_ovr_diff_prev_season` | 0.0 | Mahomes 99 in both Madden 25 and 26 |

---

## 3. VLM / Validation & Verification Module spec

Gates the Madden feature block before train/predict. Severity: **FAIL** = block/quarantine
the affected rows or season; **WARN** = log and proceed (the model tolerates NaN natively).
Baselines below are the observed values in the current dataset.

### 3a. Schema conformity (FAIL on violation)
- Exactly the **188** `madden_*` columns are present, named per the convention in §1.
- Every `madden_*` column classifies as the `madden_ratings` family via
  `partition.content_family` (enforced by the exhaustive partition test).
- **dtype = `float64`** for all 188 (currently true). Object/`pd.NA` columns are a defect
  (XGBoost rejects them) — guarded by `features._safe` returning float64 NaN.

### 3b. Value ranges (per cell)
| Measure | Observed [min, p1 … p99, max] | WARN if outside | FAIL if outside |
|---|---|---|---|
| `_ovr` | [28, 58 … 98, 99] | [25, 99] | [0, 99] |
| `_ovr_diff_prev` | [−66, −21 … 20, 66] | [−70, 70] | [−99, 99] |
| `_ovr_diff_4g` | [−58, −23 … 21, 66] | [−70, 70] | [−99, 99] |
| `_ovr_diff_prev_season` | [−58, −27 … 25, 52] | [−70, 70] | [−99, 99] |
| `madden_matchup_*` | [−40.5, −25 … 24, 44.5] | [−60, 60] | [−99, 99] |

An overall > 99 or < 0, or |diff| > 99, indicates a parsing/join bug, not real data.

### 3c. Coverage / missingness (per season, on `_ovr` columns)
NaN is **allowed** at the cell level (a game with an unmatched backup → XGBoost routes the
NaN). The gate is at the season level:
- **Baseline:** 66–78% non-null across `_ovr` columns for every season 2003–2024.
- **WARN** if a season's coverage < 60%.
- **FAIL / quarantine the season** if coverage < 30%.
- ⚠️ **2025 currently = 0% → FAILS this gate** (nflverse changed the 2025 depth-chart
  schema — missing `game_type`/`club_code`/`depth_team`; `get_weekly_starters` skips it).
  This is the documented open follow-up; until fixed, 2025 contributes ~0 Madden signal.

### 3d. Distribution drift (per season, league-mean `_ovr`)
- **Baseline:** league-mean overall ranges **75.6 (2020) → 85.4 (2008)**. There is a real
  **era drift** — EA recalibrated the scale: ~84 in 2004–2008 vs ~77 in recent years.
  Overalls are therefore **not perfectly comparable across eras**; the within-season and
  YoY diffs are more era-robust than the raw level.
- **WARN** if a season's league-mean `_ovr` is outside **[73, 87]** or is **NaN**.
- 2025 is NaN here → consistent with the §3c coverage failure.

### 3e. Signal quality (upstream)
- Per-season **gsis match rate ≥ 0.85** (the Plan-1 bridge gate; 2023 measured 0.9341).
  Below threshold ⇒ degraded coverage downstream. 2009 (~0%) and pre-2017 SD/LAC team-code
  mismatches are known low-coverage years (outside the strong 2015+ window).

### 3f. What the VLM currently catches
1. **2025 zero coverage** (3c/3d) — the headline gap; quarantine until the depth-chart
   schema mapping lands.
2. **Era drift** (3d) — informational; argues for leaning on diffs over raw level when
   training across eras.
3. **~25–35% baseline NaN** — expected (backups/unmatched/early-season diff windows), not
   a defect; tolerated by XGBoost.

> Status: today the pipeline relies on XGBoost's native NaN handling plus these documented
> checks. Promoting §3 into an executable gating module (run after `get_madden_data`,
> before train/predict) is a recommended follow-up so a future season's schema change
> fails loudly here instead of silently degrading the model.
