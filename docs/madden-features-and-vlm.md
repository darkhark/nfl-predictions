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
| `_ovr` | the slot/group's talent this game, **z-scored within season** | starter overall (slot) or mean overall of the role's starters (group), then `(x − season_mean) / season_std` over that season — removes cross-era scale drift, preserves relative magnitude. Unit = standard deviations. |
| `_ovr_diff_prev` | change vs the team's **previous game**, in **raw overall points** | raw overall − the same slot's raw overall one game earlier (by games played, skipping byes) |
| `_ovr_diff_4g` | change vs **4 games** earlier, in **raw overall points** | raw overall − the same slot's raw overall 4 games earlier |
| `_ovr_diff_prev_season` | change vs **last season's** launch value, in **raw overall points** | raw overall − the same player's prior-season launch overall |

> **Note on units:** the `_ovr` *level* is a within-season **z-score**; the three `_diff_*`
> columns are **raw overall points** (computed from the raw overalls *before* normalization,
> because differences are already era-invariant and "lost 14 overall" is more interpretable
> than a z-shift). Matchup deltas (below) are differences of the z-scored levels.

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

Levels are z-scores (SDs above/below the 2024 league mean for that slot); diffs are raw points.

| Column | Value | Reading |
|---|---|---|
| `target_madden_qb_ovr` | **+2.28** | Mahomes — 2.28 SD above the league's starting QBs (raw 99) |
| `target_madden_edge_left_ovr` | +0.20 | KC's left edge, slightly above average |
| `target_madden_exterior_ol_ovr` | −0.80 | KC tackles, below average |
| `opp_madden_edge_ovr` | +0.02 | BAL edge group, ~average |
| `madden_matchup_pass_pro` | −0.82 | KC tackles (−0.80) − BAL edge (+0.02): KC tackles outclassed |
| `target_madden_qb_ovr_diff_prev_season` | 0.0 | raw points — Mahomes 99 in both Madden 25 and 26 |

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
Levels are now within-season z-scores (≈ N(0,1)); diffs remain raw overall points.

| Measure | Unit | Observed [min, p1 … p99, max] | WARN if outside | FAIL if outside |
|---|---|---|---|---|
| `_ovr` | z-score | [−5.69, −2.3 … 2.2, 4.19] | [−5, 5] | [−10, 10] |
| `_ovr_diff_prev` | raw pts | [−66, −21 … 20, 66] | [−70, 70] | [−99, 99] |
| `_ovr_diff_4g` | raw pts | [−58, −23 … 21, 66] | [−70, 70] | [−99, 99] |
| `_ovr_diff_prev_season` | raw pts | [−58, −27 … 25, 52] | [−70, 70] | [−99, 99] |
| `madden_matchup_*` | z diff | [−5.78, −3.2 … 3.1, 5.79] | [−8, 8] | [−12, 12] |

A `_ovr` z beyond ±10 (or a per-season mean far from 0 / std far from 1) signals a
normalization or parse bug; a raw-point |diff| > 99 is impossible real data.

### 3c. Coverage / missingness (per season, on `_ovr` columns)
NaN is **allowed** at the cell level (a game with an unmatched backup → XGBoost routes the
NaN). The gate is at the season level:
- **Baseline:** 66–78% non-null across `_ovr` columns for every season 2003–2024.
- **WARN** if a season's coverage < 60%.
- **FAIL / quarantine the season** if coverage < 30%.
- ✅ **2025 = 0% → 76.5% (RESOLVED, Phase 0 / `madden-launch-ratings`).** The gap was never
  a ratings problem — it was that nflverse changed the 2025+ depth-chart schema (dated
  snapshots, granular `pos_abb`, no `game_type`/`club_code`/`depth_team`), so the
  weekly-starters detection skipped it. Phase 0 sources 2025 launch ratings from the
  madden-tools CDN `players.json` (launch iteration) bridged to gsis via nflverse seasonal
  rosters, and normalizes the new depth schema (latest pre-game-day snapshot, positions
  coarsened to the pre-2025 vocabulary for cross-era parity). Measured 2025 `_ovr` coverage
  is **0.765** — inside the 66–78% baseline band. See `scripts/experiments/madden_coverage_report.py`
  and `data/predict_games/madden_coverage/coverage_report.json`.

### 3d. Distribution drift
The raw overalls have a real **era drift** — EA recalibrated the scale (league-mean ≈ 84 in
2004–08 vs ≈ 77 recently). The `_ovr` levels are now **z-scored within season**, which
removes that drift by construction; the A/B ablation confirms this helps (Madden lift
+0.0075 raw → **+0.0129** z-scored). Two checks remain:
- **Post-normalization invariant (FAIL):** each season's `_ovr` columns should have mean ≈ 0
  and std ≈ 1 (within float tolerance). A season far off means normalization didn't run or
  the season has too few teams.
- **Pre-normalization source health (WARN):** on the *raw* overalls (before z-scoring), a
  season's league-mean far outside **[73, 87]** or **NaN** signals a data-source change.
  (2025 was NaN/empty here while the §3c gap was open; it is now populated from the
  madden-tools launch source — see §3c.)

### 3e. Signal quality (upstream)
- Per-season **gsis match rate ≥ 0.85** (the Plan-1 bridge gate; 2023 measured 0.9341,
  2024 measured 0.9469). Below threshold ⇒ degraded coverage downstream. 2009 (~0%) and
  pre-2017 SD/LAC team-code mismatches are known low-coverage years (outside the strong
  2015+ window).
- **2025 (madden-tools source) measured 0.8334**, just under the 0.85 gate. This is a
  *file-composition* effect, not a bridge failure: the madden-tools launch `players.json`
  carries the full ~3,067-player roster (camp/practice-squad/UDFA bodies that have no
  nflverse gsis by construction), vs the curated ~2,316-player theedgepredictor file the
  0.85 gate was set on. The unmatched players are overwhelmingly non-starters, so downstream
  `_ovr` coverage stays healthy (0.765, §3c) — the starters that drive the features match.

### 3f. What the VLM currently catches
1. **2025 coverage** (3c/3d) — **resolved in Phase 0** (0% → 76.5%): the depth-chart schema
   mapping and the madden-tools launch source landed on `madden-launch-ratings`. The gate
   now passes for 2025; a future season's schema change would re-trip §3c.
2. **Era drift** (3d) — informational; argues for leaning on diffs over raw level when
   training across eras.
3. **~25–35% baseline NaN** — expected (backups/unmatched/early-season diff windows), not
   a defect; tolerated by XGBoost.

> Status: today the pipeline relies on XGBoost's native NaN handling plus these documented
> checks. Promoting §3 into an executable gating module (run after `get_madden_data`,
> before train/predict) is a recommended follow-up so a future season's schema change
> fails loudly here instead of silently degrading the model.
