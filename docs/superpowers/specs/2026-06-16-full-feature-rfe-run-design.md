# Run 15 — Full (ranked + aggregated) feature pipeline: XGBoost + BART

**Date:** 2026-06-16
**Branch:** extend-rfe-and-bart-rfe (or a dedicated run-15 branch)
**Status:** Approved design — ready for implementation plan

## Objective

Test whether the full ~2,348-feature candidate set (league-wide **ranks** + **rank
changes** + aggregated **cumulative-average values** + their week-over-week **deltas** +
raw **cumulative sums**) beats the rank-only champion under identical seed, splits, and
CV folds — for **both** estimators:

- BART rank-only champion (Run 14): hold-out AUROC **0.705**
- XGBoost rank-only (Run 11/13): hold-out AUROC **~0.707**

Honest prior: rank and value are near-monotonic transforms within each (season, week)
slice, so collinearity makes **"no meaningful lift"** the likely outcome. A clean null
result is itself the deliverable — it justifies staying in rank-only mode and closes the
question.

## Background / why this is not redundant

Rank-only mode (`RANK_ONLY = True`, commit f2d20d9) was added **before** all the
play-by-play phases (Phase 1/2/3, commits ee0a300 → 2327ed6). Every recorded PBP run —
Runs 6, 9, 11, 13, and the BART champion (14) — used `RANK_ONLY = True`. **A full /
aggregated run with the current feature set has never been done.** The comparison is
clean because it reuses the same notebooks, seed (32), and train/test/hold-out splits
with only the candidate-feature set changed.

## Pipeline recap

1. **rfe.ipynb** — runs CV-RFE, writes `rfe_features_kfolds.csv` (one row per model size →
   that size's feature list), and `get_best_num_features(.005)` recommends the best size.
   RFE drops **10% of survivors per iteration**
   (`classifier_cross_validation.py:86`).
2. **grid_search.ipynb** — reads that CSV, pulls the row for `BEST_NUM_FEATS` (currently
   hardcoded **17**), runs a 100-iter random hyperparameter search, computes the 2024–25
   hold-out AUROC.
3. **bart_rfe.ipynb** — does **not** run its own pre-filter; it reads its start pool from
   the XGBoost RFE trace (`rfe_features_kfolds.csv` at `.loc[START_POOL_SIZE]`, currently
   91), then runs `BartBackwardElimination` (drop 20%/iter, 6 seed replicates, floor 10),
   re-measuring importance from BART's own `variable_inclusion` each iteration. The final
   cell does the single hold-out read with a full 1000-draw / 4-chain fit.

## Phase A — XGBoost (must run first; its trace feeds BART)

Changes to **rfe.ipynb**, parametrized by the existing `RANK_ONLY` flag so both runs stay
reproducible side by side:

- Cell 5: `RANK_ONLY = False` (full ~2,348-feature candidate set).
- Cell 9: `max_iter` 40 → **60**. At 10%-drop/round, 40 rounds from 2,348 only reaches
  ≈35 features (`0.9^40 · 2348 ≈ 35`); 60 rounds reaches the floor of 5 and crucially
  passes through the **5–17 regime** where the champion lives, making the collapse curve
  comparable. Marginal cost of 40→60 is small (the late small-model iterations are cheap;
  the early high-dimensional iterations dominate runtime and are unavoidable).
- Cell 11: write to **`rfe_features_kfolds_full.csv`** when `RANK_ONLY` is False, so the
  champion's `rfe_features_kfolds.csv` is never clobbered.

Changes to **grid_search.ipynb**:

- Read the source CSV conditioned on the same flag (`rfe_features_kfolds_full.csv` for
  this run).
- Set `BEST_NUM_FEATS` from this run's `get_best_num_features(.005)` rather than the
  hardcoded 17. (The chosen size must be an actual produced row in the full trace —
  `get_best_num_features` always returns a produced size, so this holds.)
- Otherwise unchanged — same splits, same search space → fair hold-out comparison.

## Phase B — BART (after Phase A)

Changes to **bart_rfe.ipynb**:

- `START_FEATURES` read from **`rfe_features_kfolds_full.csv`** (the full trace), at
  `START_POOL_SIZE` = the nearest produced row to **~90** (the geometric 0.9 decay from
  2,348 lands on rows near 98 and 88; use **88**, the closer one — confirm against the
  actual written CSV). This matches the proven Run-14 rationale (~3× the XGBoost CV peak,
  above the ~60 usable bound) while keeping fit times tractable (~30 min/fit at the high
  end × 6 replicates).
- Write the selected set to **`bart_rfe_features_full.csv`** (preserve the champion's
  `bart_rfe_features.csv`).
- Everything else unchanged: drop 0.2/iter, 6 replicates, floor 10, selection by
  validation-curve **peak** on the 2022–2023 slice, single hold-out read in the final
  cell with full 1000-draw / 4-chain sampling.

### Documented caveat (read the BART result accordingly)

BART's start pool is selected **by XGBoost importance** (2,348 → ~90). Run 12 already
showed XGBoost-selected features can hurt BART, so if an aggregated feature carries signal
**only BART** can exploit, the XGBoost pre-filter may discard it before BART sees it.
Accepted in exchange for tractable fit times. The BART result is therefore "best among
XGBoost's top ~90," **not** "best among all 2,348."

## Artifacts

Every output suffixed `_full` so champion artifacts are preserved for clean before/after
comparison:

| Artifact | Champion (rank-only) | This run (full) |
|---|---|---|
| XGBoost RFE feature trace | `rfe_features_kfolds.csv` | `rfe_features_kfolds_full.csv` |
| BART selected set | `bart_rfe_features.csv` | `bart_rfe_features_full.csv` |

## Execution & verification

Run headless via nbconvert; watch the existing sidecar logs
(`/tmp/xgb_rfe_progress.log`, `/tmp/bart_rfe_progress.log`) — nbconvert buffers cell
stdout until completion, so the sidecar files are the only live view. Expect **hours**,
dominated by Phase A's early high-dimensional iterations.

Per the "verify notebook executions" rule (nbconvert failures leave stale outputs), before
trusting any result confirm:

- Clean, nonzero-error exit code from each nbconvert run.
- XGBoost echoes `len(candidate_features) ≈ 2348` (proves `RANK_ONLY = False` took
  effect — a stale `True` would silently produce the 91-feature run again).
- BART echoes its ~90-feature start (`START run: N starting features`).

## Success criteria

Per estimator, report and compare against the 0.705 champion:

- RFE / backward-elimination curve (score vs #features).
- Recommended feature count (`get_best_num_features(.005)` for XGBoost; validation peak
  for BART).
- 2024–25 hold-out AUROC (+ accuracy, Brier, log loss for BART as in the current notebook).

Verdict: **beats / matches / underperforms** the champion. Do not over-interpret hold-out
AUROC deltas < ~0.005 — the 2024–25 hold-out is small.

## Out of scope (future cycles)

The other three feature ideas from this brainstorm, each its own spec → plan → run cycle:
matchup diff features (target − opp, and offense-vs-opposing-defense), situational
play-call PBP (pass/run rate by down & distance, 3rd-and-short), and recency / EWMA stats.
