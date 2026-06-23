# Madden Launch-Ratings Data Layer — Design (Phase 0)

- **Date:** 2026-06-22
- **Branch:** `madden-launch-ratings` (off `origin/master`, which carries the full Madden
  data layer, the feature-group/ablation infra, and the Bayesian-logistic package)
- **Status:** Design — pending user review
- **Program context:** Phase 0 of a four-phase program. North star: a Madden
  **launch-ratings** modeling pipeline plus a Bayesian **posterior** over the best feature
  set, to later seed a weekly-iteration transfer-learning model. Phases 1–3 (XGBoost RFE,
  BART + family pairing, Bayesian posterior) are out of scope for this spec.

## 1. Goal

Make **2025+ season-launch Madden ratings flow through the existing
`src/data/madden/` contract and actually contribute model signal** — i.e. take 2025
from the documented `_ovr` coverage of **0% → real coverage** — without disturbing the
2003–2024 path or the champion comparability contract.

This is a **data/plumbing phase**, not a modeling phase: it produces no hold-out AUROC.
Its output is a corrected feature table (2025 Madden columns populated) plus a coverage /
match-rate report. The first modeling result is Phase 1 (Run 28).

## 2. Background / constraints (verified against the code)

- The existing layer is clean and **source-agnostic downstream of ingest**: everything
  funnels through `ingest.OUTPUT_COLUMNS = [season, full_name, team, position, overall,
  weight, power_moves, finesse_moves]`. `roles.py`/`features.py` never see the source
  schema. The Run-26 within-season `_ovr` z-score is **already baked in**
  (`features.zscore_overalls_within_season`, called at the end of
  `collect.get_madden_data`). The target/opp double-merge + matchup deltas happen later in
  `src/data/collect_all.py` (lines 89–103), **after** the one-week shift.
- **The 2025 blocker is the starter→rating join, not the ratings.**
  `starters.get_weekly_starters` hard-requires `{game_type, club_code, depth_team, gsis_id,
  position}` and returns an **empty frame** when nflverse's 2025+ `import_depth_charts`
  schema drops the first three (`starters.py:44–50`; documented at
  `docs/madden-features-and-vlm.md:143`). With no starters, no team-week `_ovr` is built →
  2025 = 0% coverage.
- **The `madden-tools` brief is partly outdated** (verified by reading the repo at
  `~/ClaudeProjects/MaddenTools/madden-tools`):
  - `services/api` is **payments/entitlement only — not a ratings endpoint** (eliminated).
  - `companion-app-ingestor` is a **no-op stub**; the EA-API source inside `json-generator`
    is **commented out**. The live ingest is Companion-App → DigitalOcean managed Postgres.
  - The real artifact is **`players.json`, emitted per-iteration** by `json-generator`,
    alongside `teams.json` and `iterations.json` (11 files/iteration). A **real example is
    on disk** at
    `services/draft-genius/docs/examples/madden-26/json/iterations/23/players.json`
    (2019 players) — used as the offline test fixture.
  - **Launch is trivially identifiable**: `iterations.json` lists `{id:0, label:"Launch",
    release_date, active}`, then `Week 1`, `Week 2`, … So "launch iteration" = the entry
    with `label == "Launch"`. The future weekly phase points at a different iteration with
    zero new plumbing.

## 3. Decisions (confirmed with the user)

1. **Access path = CDN `players.json` file-contract.** Treat the launch-iteration
   `players.json` + `teams.json` as a versioned file input fetched from the DO Spaces CDN
   (the production artifact). nfl-predictions stays decoupled from the `madden-tools`
   runtime; tests run offline against the committed example fixture. (Rejected:
   `services/api` — not a ratings endpoint; running `json-generator` locally — pulls the
   pnpm/mise/node toolchain + live prod DB into Phase 0; direct Postgres query —
   reimplements the generator's normalization and couples to the DB schema.)
2. **Fix the 2025 starters join inside Phase 0** (quarantine-2025 is the documented
   fallback if the new depth-chart schema cannot reconstruct a depth rank). Rationale: the
   2025 NFL season is complete and is half the hold-out (≈544 rows); leaving it inert
   undercuts the 2025-forward north star.
3. **gsis bridge for the 2025+ path = nflverse seasonal rosters**
   (`nfl_data_py.import_seasonal_rosters`, which carries gsis `player_id`). Same source as
   starters → consistent team codes; no external-repo dependency; generalizes to weekly.
   The historical (≤2024) theedgepredictor `processed/` bridge is left untouched.
4. **Run-log:** Phase 0 gets a short data-quality note (2025 0% → X%, gsis match rates);
   the numbered **Run 28 is reserved for Phase 1** (first XGBoost-RFE modeling result).

## 4. Architecture — four additive workstreams

Everything below is **additive**: no existing function's behavior changes for ≤2024.

### 4.1 Launch `players.json` ingest adapter — new module `src/data/madden/launch.py`

Separation of concerns (Command-Query): a **pure parser** that takes already-loaded JSON,
and a **thin fetcher** that does I/O.

- `parse_launch_ratings(players: list[dict], teams: list[dict], season: int) -> DataFrame`
  Returns exactly `ingest.OUTPUT_COLUMNS`:
  - `full_name` = `first_name` + `" "` + `last_name` (reusing the same strip/space
    convention as `ingest._resolve_full_name`).
  - `team` = `teams[team_id].acronym` → `TEAM_ABBR_MAPPINGS` (a `team_id → acronym` lookup
    built from `teams.json`; verified shape `{id, name, acronym, conference, division,
    state}`, e.g. `26 → "BAL"`).
  - `position` = `position`; `overall` = `rating_overall`; `weight` = `weight`;
    `power_moves` = `rating_power_moves`; `finesse_moves` = `rating_finesse_moves`.
  - Empty/garbled input → empty frame with `OUTPUT_COLUMNS` (mirrors `ingest`'s resilience
    contract so the caller proceeds with NaN Madden features).
- `select_launch_iteration(iterations: list[dict]) -> dict` — the entry with
  `label == "Launch"` (validating exactly one such active entry; raise a specific error
  otherwise).
- `load_madden_launch(game_version: str) -> DataFrame` — fetcher: resolve the launch
  iteration's `players.json` + `teams.json` for `game_version` from the CDN, then delegate
  to `parse_launch_ratings`. I/O is isolated here so the parser is unit-tested without
  network. Mirrors the resilience of `ingest.load_madden_season` (network failure → empty
  frame + warning).

### 4.2 Source routing — `src/data/madden/collect.py`

`get_madden_data` / `_player_season` route by season via a small map:

```
SEASON_TO_GAME_VERSION = {2025: "madden-26"}   # extended forward per release
def _load_player_season_raw(season):
    if season >= 2025:
        return launch.load_madden_launch(SEASON_TO_GAME_VERSION[season])
    return ingest.load_madden_season(season)     # theedgepredictor raw, unchanged
```

The rest of `_player_season` (`ids.attach_gsis_id` → `roles.classify_roles`) is unchanged;
only the bridge **source** differs by season (4.3).

### 4.3 gsis bridge for 2025+ — `src/data/madden/ids.py`

Add a roster-backed lookup builder; keep `attach_gsis_id`'s matching logic:

- `attach_gsis_id_from_rosters(df, season, rosters=None)` — builds the same
  `name|team` (primary) and `name|coarse_position` (fallback, ambiguous→None) lookups as
  today, but from `nfl_data_py.import_seasonal_rosters([season])`
  (`player_name`, `player_id`=gsis, `team`, `position`), reusing `normalize_name`,
  `MADDEN_TO_COARSE_POSITION`, and `TEAM_ABBR_MAPPINGS`. Injectable `rosters` arg for tests.
- Routing: ≤2024 → existing `attach_gsis_id` (theedgepredictor `processed/`); ≥2025 →
  `attach_gsis_id_from_rosters`. (Decide in `collect`/`ids`; keep the two paths explicit.)

### 4.4 `starters.py` 2025 depth-chart fix

- Inspect the actual 2025 `nfl.import_depth_charts([2025])` schema (an early
  implementation task — its exact columns are unknown until run). Map the new columns to
  the rank / team / REG-filter semantics so `available_starters` (leakage-safe next-man-up,
  Out/Doubtful removed) still yields one row per position per team-week.
- Preserve every leakage convention: REG-only, pre-kickoff depth + injuries, depth rank 1
  minus unavailable. The old-schema path must keep working (≤2024).
- **Fallback:** if the 2025 schema genuinely lacks a reconstructable depth rank, fall back
  to quarantine-2025 (return empty for 2025, documented) rather than inventing a rank.

## 5. Data flow (unchanged except the two new source paths)

```
launch players.json + teams.json (iteration label="Launch")        [2025+]
theedgepredictor raw/{season}.csv                                  [≤2024]
        │  (adapter / ingest → ingest.OUTPUT_COLUMNS)
        ▼
ids.attach_gsis_id[_from_rosters]  →  roles.classify_roles
        ▼                                   ⟂  starters.get_weekly_starters
features.build_team_week_overalls  ◄───────────┘  (nflverse depth, 2025-fixed)
        ▼   add_within_season_diffs → add_prev_season_diff → zscore_overalls_within_season
collect_all.py double-merge (target_/opp_) + add_madden_matchup_columns   (after one-week shift)
        ▼
schedule_and_weekly.parquet  (188 madden_* columns; 2025 now populated)
```

## 6. Leakage & comparability contract (unchanged)

- Launch ratings are known at season start → valid for every in-season week; the one-week
  shift is already baked into the parquet (do **not** re-apply); REG-only.
- No new feature columns, no renames: the `madden_ratings` family (`partition.py`: any
  column containing `madden`) and its 188 members are unchanged — only their **2025 values**
  go from NaN to populated. This keeps Phases 1–3 comparable to the champion.
- Meta/leakage columns remain excluded as before.

## 7. Testing (TDD — write tests first)

Unit tests (pure, fixture-driven, no network/DB):

1. **Adapter parse** — example `players.json`+`teams.json` → expected `OUTPUT_COLUMNS`
   rows: full-name join, `team_id → acronym → abbr`, `rating_*` renames, dtype/NaN.
2. **Launch-iteration selection** — `iterations.json` → the `label=="Launch"` entry;
   error on zero/multiple.
3. **nflverse gsis bridge** — synthetic rosters → expected gsis; ambiguous name|pos → None;
   primary name|team wins over fallback.
4. **Starters 2025 schema** — synthetic new-schema depth frame → expected starters; the
   old schema still produces identical results (regression guard).
5. **Source routing** — season → correct source; `SEASON_TO_GAME_VERSION` lookup.
6. **Coverage assertion (integration-lite)** — after the fix, rebuilt 2025 `_ovr` columns
   have coverage **> 30%** (clears the VLM FAIL gate; target the documented 66–78% band).

## 8. Deliverables

- Importable modules (`launch.py`, the `ids`/`collect`/`starters` additions) — **not just
  notebooks** — with the unit tests above.
- A **data-quality report**: per-season gsis match rate + `_ovr` coverage %, showing 2025
  0% → X%; refresh the coverage table in `docs/madden-features-and-vlm.md` (§3c/§3e).
- A short README run-log **data-quality note** (Run 28 reserved for Phase 1).
- A PR off `madden-launch-ratings`.

## 9. Out of scope / deferred

- The theedgepredictor `raw/{season}.csv → dataset/{season}.parquet` upgrade (richer
  attributes + `player_id`/`pfr_id` join keys). It is an **optional quality lever** — pursue
  only if the gsis match rate proves to be the bottleneck. Not required to unblock 2025.
- All modeling (Phases 1–3) and weekly iterations (future project).
- Promoting the VLM §3 spec into an executable gating module (recommended follow-up,
  unchanged from PR #39).

## 10. Risks / open verification items

- **Real launch artifact availability.** The on-disk example is `madden-26` but dated 2024
  (likely placeholder/clone). Before relying on it, verify the **live** Madden-26 (2025
  season) launch `players.json`+`teams.json`+`iterations.json` are reachable on the CDN
  (DO Spaces bucket + creds are in `madden-tools`' committed `workers/json-generator/.env`).
  Build/test the adapter against the committed fixture in parallel; this verification gates
  only the real-data run, not the code.
- **Version ↔ season map.** Confirm Madden NFL 26 = the 2025 NFL season (so 2025 →
  `madden-26`), and define the forward mapping convention.
- **2025 depth-chart schema is unknown until run.** The `starters.py` fix's exact column
  mapping can't be finalized until `import_depth_charts([2025])` is inspected; the
  quarantine fallback bounds the risk.
- **gsis match rate for 2025.** nflverse rosters use current team codes; pre-2017
  SD/LAC-style mismatches don't apply to 2025, but spot-check the 2025 match rate against
  the ≥0.85 VLM gate.
