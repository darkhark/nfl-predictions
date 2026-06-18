# Madden-Ratings Features — Design

- **Date:** 2026-06-18
- **Branch context:** builds on `recency-ewma-features`
- **Status:** Design — pending user review

## 1. Goal

Test whether **player talent (Madden overall ratings), blended with weekly
starter/availability information, improves NFL game-outcome predictions.** This is
a genuinely *new* information source — orthogonal to the box-score / EPA / play-by-play
signal the model already uses — and is therefore one of the few additions that could
break the redundancy/over-selection ceiling documented in README Runs 21–24 (where
added feature families are mutually redundant and hold-out is data/regime-limited).

The headline experiment is a clean A/B via the existing `feature_groups/` ablation:
**base vs. base + `madden_ratings`** (and optionally `madden_ratings`-only).

## 2. Background / constraints

- **Model grain:** team-game. Each game is duplicated into two perspective rows keyed
  on `(target_team, season, week)`, with `target_*` / `opp_*` feature pairs
  (`src/data/collect_all.py`). Madden features must be produced at this grain.
- **Pipeline convention:** one module per source under `src/data/<source>/collect.py`
  (`play_by_play/`, `schedule/`, `weekly/`, `next_gen_stats/`), merged in
  `collect_all.py`. We add `src/data/madden/`.
- **Training window:** existing data spans 2003–2025. Madden raw data spans 2001–2025
  (see §3). We will use a window inside that overlap; start year decided by a
  per-season completeness check (default candidate: 2010 or 2015; see §8).
- **Overfit ceiling:** ~2,800 existing columns; the model over-selects and degrades on
  redundant additions. Madden features are grouped into a single ablation group so the
  whole block can be toggled and scored.

## 3. Data sources

### 3.1 Madden ratings — `theedgepredictor/nfl-madden-data`, `raw/` stage

- URL pattern: `https://raw.githubusercontent.com/theedgepredictor/nfl-madden-data/main/data/madden/raw/{season}.csv`
- **Coverage:** 2001–2025. **Completeness in `raw/`: ~0% null** on `Overall`,
  `Power Moves`, `Finesse Moves`, `Weight` (the 22–45% null rate seen earlier was a
  `processed/`-only artifact from appending un-rated roster players; `raw/` contains
  only rated players).
- **Why `raw/` and not `processed/`:** `processed/` is normalized and carries `gsis_id`
  but **drops the pass-rush move ratings** (`Power Moves` / `Finesse Moves` /
  `Block Shedding`) and collapses position to coarse `DL/LB/DB`. `raw/` retains the full
  ~50-attribute table **and** medium-grained positions — both required by the role
  classifier (§5).
- **Fields used:** `Position`, `Overall`, `Weight`, `Power Moves`, `Finesse Moves`
  (plus `Man Coverage`, `Zone Coverage`, `Pursuit`, `Hit Power`, `Strength`, `Speed`
  available as classifier inputs/cross-checks), player name, `Team`, `season`.

### 3.2 Season alignment (verified)

Madden N models NFL season `1999 + N`. The edgepredictor files are keyed by
**NFL season-start year**, so `season=Y` joins directly to that player's NFL season-Y
games — **no off-by-one**:

| File | Madden title | NFL season | Verified by |
|---|---|---|---|
| `2023.csv` | Madden 24 | 2023 | Bryce Young (2023 #1) present, `Panthers,QB,74` |
| `2024.csv` | Madden 25 | 2024 | Caleb Williams / Jayden Daniels / Bo Nix present |
| `2025.csv` | Madden 26 | 2025 | classic schema, rated roster present |

Launch/base ratings are set in **preseason (summer of year Y)**, so they are known
before any season-Y game — **leakage-safe** as a season-constant talent value.
**Verified** the `raw/` season value is the *launch* snapshot, not an end-of-season one,
via breakout players who'd look inflated if final: Puka Nacua 2023 = **67** (final ~84),
C.J. Stroud 2023 = **73** (final ~88), Brock Purdy 2022 = **59** (Mr. Irrelevant launch);
elite players are high at launch too (Mahomes 99 / 95). This matters for §7.3.

### 3.3 Schema drift (engineering caveat)

`raw/` mixes two source schemas and **drift is per-file, not per-era**:

- **Classic (weebly):** Title-Case columns (`Position`, `Overall`, `Power Moves`,
  `Weight`, …). Seen in 2018, 2022, 2023, 2025. Some within-era renames (e.g. 2015's
  overall column label differs).
- **Nested (EA API):** `stats/powerMoves/value`, `stats/overall/value`, lowercase
  `weight`, split `firstName`/`lastName`, plus `/diff` columns (weekly rating change).
  Seen in 2024.

**Normalizer detects schema by presence of `stats/` columns**, not by year, and emits a
single tidy table: `[season, player_name, team, position, overall, weight, power_moves,
finesse_moves]`. Known data-quality issue: **2025 `position_short_label` is `"False"`
for ~2,100/2,400 rows** — backfill 2025 position from the depth chart or `processed/`.

### 3.4 Joining Madden → NFL data

- `raw/` has **no `gsis_id`** (only name/team/position/season). Bridge to `gsis_id` via
  the `processed/` file (which carries both `gsis_id` and the same players), matched on
  name+team+season (or `madden_id`).
- **Fallback** for unmatched rows: name + position + team direct match to nflverse
  (the originally proposed key). Residual misses → `NaN`.

### 3.5 nflverse (already the project's data backbone, `nfl_data_py`)

- `import_depth_charts(years)` — pre-game slated starter per position slot, with
  `gsis_id`. Source of *who lines up*.
- `import_injuries(years)` — weekly `report_status` (Out/Doubtful/Questionable) +
  game status, with `gsis_id`. Source of *who's unavailable*.

## 4. Pipeline (5 stages, in `src/data/madden/`)

1. **Ingest + normalize** Madden `raw/` → tidy player-season table (§3.1, §3.3);
   attach `gsis_id` (§3.4).
2. **Role classification** (`roles.py`) → scheme/era-invariant role per player (§5).
3. **Starter identification** (leakage-safe) → actual starter per slot per team-week
   from depth charts minus injuries, keyed on `gsis_id` (§6).
4. **Feature assembly** → attach each starter's `overall` + role; build per-position +
   grouped + matchup columns, each with `overall`, `diff_vs_prev_game`,
   `diff_vs_4_games_ago`, for `target` and `opp` (§7).
5. **Merge** into `schedule_and_weekly` on `(target_team, season, week)`; tag all new
   columns into the `madden_ratings` feature group (§9).

## 5. Role classification (`src/data/madden/roles.py`)

Ports the **pre-EDGE guide-generator logic** (from
`madden-tools-guide-generator` history, commit prior to `9c04c76` "Test new defensive
positions", 2025-08-17), which classifies by **weight + pass-rush moves** rather than by
label:

```text
pass_rush_total      = power_moves + finesse_moves
is_edge_rusher       = (OLB and pass_rush_total >= 130) or (DE and weight <= 280)
is_interior_dl       = DT or (DE and weight >= 280)
is_off_ball_lb       = (OLB and pass_rush_total < 130) or MLB
```

**Vocabulary map (not hard-coded), to be era/source-robust:**

| Source label | Handling |
|---|---|
| `LEDGE`, `REDGE` | → Edge (unambiguous, modern game / companion data) |
| `SAM`, `MIKE`, `WILL` | → Off-ball LB (unambiguous, modern game / companion data) |
| `MLB` | → Off-ball LB |
| `DT` | → Interior DL |
| `LE`, `RE` (i.e. DE) | → Edge vs Interior via weight threshold (≤/≥ 280) |
| `LOLB`, `ROLB` (i.e. OLB) | → Edge vs Off-ball via `pass_rush_total` threshold (130) |
| `LT/LG/C/RG/RT`, `QB/HB/FB/WR/TE`, `CB/FS/SS` | → direct |

The public `raw/` data is uniformly classic (`LE/RE/DT/LOLB/ROLB/MLB`) across all years,
so the attribute classifier runs for every year; the modern labels are mapped directly
if/when encountered (e.g. companion data). **Side (L/R) is preserved** from
`LE/RE/LOLB/ROLB` for the edge matchup columns.

Output roles: `Edge`, `Interior DL`, `Off-ball LB`, `CB`, `Safety`, plus offense slots.

## 6. Starter identification (leakage-safe)

- Slated starter = depth-chart rank #1 at each slot (`import_depth_charts`, pre-game).
- Remove anyone `Out`/`Doubtful` on that week's report (`import_injuries`); promote the
  next available depth-chart player ("next man up").
- Keyed on `gsis_id`; attach Madden `overall` + role.
- **Only pre-kickoff information is used.** Final game status (set ~90 min pre-game) is
  permitted; nothing post-kickoff (e.g. realized snap counts for the game being
  predicted) is used.

## 7. Feature taxonomy (full set — no phasing)

Groups **A** and **B** below each expand to **3 columns** (`overall`,
`diff_vs_prev_game`, `diff_vs_4_games_ago`) × **2 perspectives** (`target`, `opp`) =
6 columns per entry. Group **C** (matchup deltas) is **one directional value per row**
each (no diff expansion), since each perspective row already carries its own target/opp
framing.

**Diff semantics (confirmed):** slot-based, comparing the overall of *whoever started
that slot* across the team's **games played** (previous game / 4 games back, skipping
byes). Same starter → ~0; backup replaces starter → negative by the talent drop; upgrade
(trade / return) → positive. First game of season → `NaN` for prev-game diff; games 1–4
→ `NaN` for 4-games-ago diff. Group diffs compare group-mean overall across games.

### A. Granular per-position (starter overall)

- **Offense:** QB, RB, WR1, WR2, WR3, TE, LT, LG, C, RG, RT
- **Defense (side-aware):** Left Edge, Right Edge

### B. Grouped talent (mean overall of the role's starters)

Backfield (RB+FB), WRs (top 3), Interior OL (LG/C/RG), Exterior OL (LT/RT), Edge (L+R),
Interior DL (DT), Off-ball LB (MIKE + off-ball OLB), Cornerbacks, Safeties. (QB's
granular column serves as its group — no separate QB group.)

### C. Matchup deltas (directional, one value per perspective)

- `target_ExteriorOL − opp_Edge`
- `target_InteriorOL − opp_InteriorDL`
- `target_WRs − opp_(CB+Safety)`
- `opp_ExteriorOL − target_Edge`

### Performance / momentum signal (§7.3)

Real-life performance and momentum *do* drive Madden ratings — EA bumps a hot player and
dings a cold one. But this enters the data at two different time scales, and only one is
available historically:

- **Within-season (weekly drift):** the live game updates ratings during the season as
  players perform. The public `raw/` source preserves **only the launch snapshot**, so a
  player's rating is **constant within a season** here. Therefore the within-season
  `diff_vs_prev_game` / `diff_vs_4_games_ago` columns are driven **entirely by starter
  changes** (injury / trade / benching), *not* by performance re-rating. Capturing true
  weekly drift would require polling the live EA API forward each week with leakage-safe
  as-of timestamps — **out of scope for v1** (§12). It is also the component **most
  redundant** with the model's existing EWMA/EPA recency features, so it is low-priority
  for the orthogonal-signal goal.
- **Season-over-season:** the launch rating for year Y reflects the player's *prior*
  season (Y−1), and is leakage-safe (set preseason). To capture this performance-driven
  momentum in a way the data actually supports, we add an **optional**
  `diff_vs_prev_season_launch` per granular slot (this season's starter launch overall −
  the *same player's* launch overall last season): positive = a player EA rewarded for a
  strong prior year (ascending), negative = declining. This sits in the `madden_ratings`
  group like the rest and is the leakage-safe realization of the momentum idea.

Approximate column count: ~140 base columns (+ ~13 if the optional season-over-season
diff is included). Acceptable because they are isolated into a single toggleable group
(§9) and scored by ablation.

## 8. Coverage window & missing-data handling

- Run a per-season completeness check on `raw/` Overall + attributes + the bridged
  `gsis_id` join rate among **actual depth-chart starters** (not raw file rows). Choose a
  start year where starter-level coverage is solid. Default candidate window: **2015–2025**
  (fallback 2010–2025 if 2010–2014 starter coverage validates).
- Missing values (unmatched player, missing depth chart, early-season diff windows) →
  **`NaN`** (XGBoost-native). No imputation in v1.

## 9. Experiment / validation

- Tag all new columns into a single feature group **`madden_ratings`** in
  `data_science_utilities/feature_groups/`, consistent with the existing group-ablation
  framework (`group_ablation.ipynb`, `run_group_ablation.py`, the
  `test_feature_group_*` suites).
- **Primary experiment:** group ablation of **base vs. base + `madden_ratings`**, scored
  identically to README Runs 21–24 (AUROC + the recorded hold-out metrics). Optional:
  **`madden_ratings`-only** to gauge standalone signal.
- Success = a real, non-degrading hold-out improvement when the group is added.

## 10. Module layout

```
src/data/madden/
  collect.py     # orchestrates stages 1–5, emits team-week features
  ingest.py      # download + per-file schema-normalize raw/ (stage 1)
  ids.py         # bridge raw -> gsis_id via processed/ + name/pos/team fallback (3.4)
  roles.py       # vocabulary map + attribute classifier (§5)
  starters.py    # depth-chart + injury -> leakage-safe starter per slot (§6)
  features.py    # per-position / grouped / matchup columns + diffs (§7)
tests/data_science_utilities/  # + feature-group registration test for madden_ratings
tests/data/madden/             # unit tests per module (TDD)
```

## 11. Risks / caveats

1. **`raw/` schema drift** — mitigated by per-file schema detection (§3.3).
2. **2025 position quality** (`"False"` rows) — backfill from depth chart / `processed/`.
3. **Join misses** — `processed/` bridge + name/pos/team fallback; residual → `NaN`.
4. **Depth-chart historical reliability** pre-2015 — validated in the §8 coverage check.
5. **Dimensionality vs. overfit ceiling** — contained by the single toggleable group +
   ablation scoring; expectation is that ablation, not raw addition, decides inclusion.
6. **Repo licensing** — `theedgepredictor/nfl-madden-data` and the guide-generator are
   unlicensed; use is personal/research only. Classifier logic is reimplemented, not
   copied verbatim.

## 12. Out of scope (v1)

- Companion-app / Postgres (SAM/MIKE/WILL) blending — public `raw/` suffices and reaches
  the full window.
- **Within-season weekly rating drift** (the performance/momentum signal of §7.3).
  Public data is launch-only; capturing weekly drift needs forward live-API polling with
  leakage-safe as-of timestamps, and it overlaps existing recency features. Later
  enhancement. (Season-over-season momentum *is* in scope as the optional
  `diff_vs_prev_season_launch`, §7.3.)
- Rating imputation for unmatched starters.
