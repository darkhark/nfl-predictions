# Run 15 — Full (ranked + aggregated) Feature Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing XGBoost and BART feature-selection pipelines on the full ~2,348-feature candidate set (`RANK_ONLY = False`) and report hold-out AUROC against the rank-only champion (0.705), without destroying any champion artifacts.

**Architecture:** Three existing notebooks are edited via config-only changes and re-run headless. XGBoost RFE runs first (its feature trace is BART's start pool). Every output is written to a `_full`-suffixed file so the champion's CSVs survive for clean before/after comparison.

**Tech Stack:** Jupyter notebooks executed via `jupyter nbconvert`, pandas, XGBoost, PyMC / pymc-bart. No source-module changes — this is an experiment-configuration plan.

---

## Conventions for this plan

- All paths are relative to the repo root `/Users/joshuaharkness/ClaudeProjects/nfl-predictions`. `cd` there first.
- Notebook directory (abbreviated **CV_DIR** below):
  `notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/`
- Edit notebook cells with the **NotebookEdit** tool (`cell_id` = the 0-based index shown by the inspection snippet in each task; pass the **full new cell source**). Do **not** hand-edit the JSON.
- These runs take **hours**. Execute with `run_in_background: true` and poll the sidecar logs; do not block.
- **Environment (REQUIRED for headless runs):** the notebooks run in the conda env `nfl-predictions`. Use its jupyter binary `/opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/jupyter`, and **prepend `PYTHONPATH=/Users/joshuaharkness/ClaudeProjects/nfl-predictions`** to every `nbconvert` invocation. Reason: `rfe.ipynb` cell 0 only adds `..` to `sys.path` (resolves to `schedule_and_weekly/`, not the repo root where `data_science_utilities` lives) — it works interactively because Jupyter is launched from the repo root, but headless nbconvert fails with `ModuleNotFoundError: No module named 'data_science_utilities'` unless `PYTHONPATH` supplies the repo root. Add `--ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1`. The data paths (`../../../../../data/...`) already resolve because nbconvert sets CWD to the notebook's directory. Canonical form:
  ```bash
  cd notebooks/model_training/predict_games/schedule_and_weekly/cross_validation
  PYTHONPATH=/Users/joshuaharkness/ClaudeProjects/nfl-predictions \
    /opt/homebrew/Caskroom/miniforge/base/envs/nfl-predictions/bin/jupyter nbconvert \
    --to notebook --execute --inplace <notebook>.ipynb \
    --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1 > /tmp/<log>.out 2>&1
  ```
- This is not classic unit-test TDD — the "test" before each long run is a **config echo**: run only the cheap setup cells and confirm the feature count printed matches expectations *before* committing the machine to a multi-hour fit. Per the project's "verify notebook executions" rule, a stale config silently produces the wrong run.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `CV_DIR/rfe.ipynb` | XGBoost CV-RFE; produces feature trace | Modify cells 5, 9, 11 |
| `CV_DIR/grid_search.ipynb` | XGBoost hyperparam search + hold-out eval | Modify cell 1; add overall-AUROC print |
| `CV_DIR/bart_rfe.ipynb` | BART backward-elimination + hold-out eval | Modify cells 1, 5 |
| `data/predict_games/model_features_in/rfe_features_kfolds_full.csv` | XGBoost full-run feature trace | Created by run |
| `data/predict_games/model_features_in/bart_rfe_features_full.csv` | BART full-run selected set | Created by run |
| `docs/superpowers/specs/2026-06-16-full-feature-rfe-run-design.md` | Design spec | Already committed |

---

## Task 1: Configure rfe.ipynb for the full run

**Files:**
- Modify: `CV_DIR/rfe.ipynb` cells 5, 9, 11

- [ ] **Step 1: Inspect current cell sources & indices**

Run:
```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
python -c "
import json
nb=json.load(open('notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/rfe.ipynb'))
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code':
        s=''.join(c['source'])
        if 'RANK_ONLY' in s or 'get_optimal_features' in s or 'to_csv' in s:
            print('=== cell',i,'==='); print(s); print()
"
```
Expected: cell 5 contains `RANK_ONLY = True`; cell 9 calls `get_optimal_features_no_grouped_records(max_iter=40, ...)`; cell 11 calls `features_df.to_csv('.../rfe_features_kfolds.csv')`. Confirm the printed indices match (5, 9, 11). If they differ, use the actual indices in the edits below.

- [ ] **Step 2: Edit cell 5 — flip the flag to the full set**

NotebookEdit `cell_id: 5` with this full new source (only the flag value changes; keep the explanatory comment and the rest of the cell intact):
```python
# Rank-only experiment (PR #28): drop every non-rank feature derived from a cumulative
# value -- the cumulative averages, their week-over-week changes, and the cumulative sums --
# keeping only the rank / rank-change features (plus raw per-game stats and context).
# This removes the highly-correlated source columns so RFE can surface the rank signal
# directly. Set RANK_ONLY = False to restore the full-feature pipeline.
# Run 15: RANK_ONLY = False -> full ~2,348-feature set (ranks + aggregated values + sums + deltas).
RANK_ONLY = False

candidate_features = list(MODEL_FEATURES_BEFORE_RFE['feature'])
if RANK_ONLY:
    candidate_features = [
        f for f in candidate_features
        if 'cumulative' not in f or f.endswith('_rank') or f.endswith('_rank_change')
    ]

# season is no longer a candidate feature (era drift; hold-out lies outside its
# training range) but is still needed for the time-based split below
model_inputs_df = INPUT_DATA[candidate_features + ['season']]
# randomize the data
model_inputs_df = model_inputs_df.sample(
    frac=1, random_state=RANDOM_SEED
).reset_index(drop=True)
model_inputs_df
```

- [ ] **Step 3: Edit cell 9 — raise max_iter to 60**

NotebookEdit `cell_id: 9` with this full new source (only `max_iter` changes, 40 → 60):
```python
rfe = classifier(
    X_train, y_train, RFE_XGB_PARAMS, model_score_metric='roc_auc'
)

def _log_progress(row):
    with open('/tmp/xgb_rfe_progress.log', 'a') as f:
        f.write(f"iter {row['iteration']}/{row['max_iter']}: "
                f"{row['num_features']} features, cv {row['score']:.4f}\n")

rfe.get_optimal_features_no_grouped_records(
    max_iter=60, min_features=5, verbose=1, on_iteration=_log_progress
)
```
Rationale: at 10%-drop/round, `0.9^40 · 2348 ≈ 35` (never reaches the small regime); `0.9^59 · 2348 ≈ 5` reaches the floor and passes through the 5–17 champion regime.

- [ ] **Step 4: Edit cell 11 — write to a non-clobbering filename**

NotebookEdit `cell_id: 11` with this full new source:
```python
features_df = rfe.get_features_in_dataframe()
_rfe_out = 'rfe_features_kfolds.csv' if RANK_ONLY else 'rfe_features_kfolds_full.csv'
features_df.to_csv(f'../../../../../data/predict_games/model_features_in/{_rfe_out}')
features_df
```

- [ ] **Step 5: Config echo — verify the full set is selected WITHOUT the long run**

Run only the setup cells (this is fast — it stops before the RFE call in cell 9):
```bash
cd notebooks/model_training/predict_games/schedule_and_weekly/cross_validation
jupyter nbconvert --to notebook --execute --stdout rfe.ipynb \
  --ExecutePreprocessor.timeout=600 2>/dev/null \
  | python -c "import sys,json; nb=json.load(sys.stdin);
import re
for c in nb['cells']:
    if c['cell_type']=='code':
        for o in c.get('outputs',[]):
            txt=str(o.get('data',{}).get('text/plain','')) + str(o.get('text',''))
            m=re.search(r'\[(\d+) rows x (\d+) columns\]', txt)
            if m: print('model_inputs_df shape rows/cols:', m.group(1), m.group(2))
" || echo "If this errors or times out on the RFE cell, instead add a temporary debug print of len(candidate_features) and re-run only cells 0-5."
```
Simpler robust alternative (recommended): replicate cell 5's filter in a standalone script (read `xgb_features_list.csv`, apply the filter), confirm it prints **≈ 2349** for the full set (vs **≈ 1529** for rank-only), then proceed.
Expected: candidate feature count ≈ **2349**. If it prints ≈ 1529, the flag edit did not take — fix cell 5 before proceeding. (Rank-only keeps 1,529 features, not the ~91 in earlier notes — 91 is BART's start-pool row, a different number.)

- [ ] **Step 6: Commit the config change**

```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
git add notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/rfe.ipynb
git commit -m "Configure rfe.ipynb for Run 15 full feature set (RANK_ONLY=False, max_iter=60, _full output)"
```

---

## Task 2: Execute the XGBoost full RFE (long run)

**Files:**
- Reads: `CV_DIR/rfe.ipynb`
- Produces: `data/predict_games/model_features_in/rfe_features_kfolds_full.csv`

- [ ] **Step 1: Clear the stale progress log**

```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
: > /tmp/xgb_rfe_progress.log
```

- [ ] **Step 2: Run the notebook headless, in the background**

Run with `run_in_background: true`:
```bash
cd notebooks/model_training/predict_games/schedule_and_weekly/cross_validation
jupyter nbconvert --to notebook --execute --inplace rfe.ipynb \
  --ExecutePreprocessor.timeout=-1 > /tmp/xgb_rfe_nbconvert.out 2>&1
```
This takes hours — the early iterations fit 5000-estimator CV models over 2,000+ features.

- [ ] **Step 3: Poll progress (repeat until the log shows it reaching ~5 features)**

```bash
tail -n 5 /tmp/xgb_rfe_progress.log
```
Expected: lines like `iter 1/60: 2348 features, cv 0.6xxx`, then steadily decreasing feature counts down toward 5.

- [ ] **Step 4: Verify clean completion (do NOT trust outputs without this)**

```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
tail -n 15 /tmp/xgb_rfe_nbconvert.out          # must NOT contain a Traceback / CellExecutionError
ls -la data/predict_games/model_features_in/rfe_features_kfolds_full.csv
python -c "
import pandas as pd
df = pd.read_csv('data/predict_games/model_features_in/rfe_features_kfolds_full.csv', index_col=0)
print('rows (model sizes):', list(df.index)[:3], '...', list(df.index)[-3:])
print('max size:', max(df.index), ' min size:', min(df.index))
"
```
Expected: no traceback; the CSV exists; index max ≈ **2349** (proves the full set ran) and min ≈ **5** (proves it reached the floor). If max ≈ 1529, the rank-only set ran — Task 1 Step 2 failed; fix and re-run.

- [ ] **Step 5: Capture the recommended feature count**

The notebook's cell 12 prints `get_best_num_features(.005)`. Read it from the executed notebook:
```bash
python -c "
import json
nb=json.load(open('notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/rfe.ipynb'))
for c in nb['cells']:
    if c['cell_type']=='code' and 'get_best_num_features' in ''.join(c['source']):
        for o in c.get('outputs',[]):
            print('best_num_feats =', o.get('data',{}).get('text/plain', o.get('text')))
"
```
Record this integer as **XGB_BEST_NUM_FEATS** — it feeds Task 3. (It is guaranteed to be one of the produced row indices.)

- [ ] **Step 6: Commit the executed notebook + trace**

```bash
git add notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/rfe.ipynb \
        data/predict_games/model_features_in/rfe_features_kfolds_full.csv
git commit -m "Run 15: XGBoost full-feature RFE trace (2348 -> 5), best_num_feats=<XGB_BEST_NUM_FEATS>"
```

---

## Task 3: Point grid_search.ipynb at the full trace

**Files:**
- Modify: `CV_DIR/grid_search.ipynb` cell 1 (source CSV + `BEST_NUM_FEATS`); add one overall-AUROC print cell

- [ ] **Step 1: Edit cell 1 — read the full trace and use the full-run best size**

NotebookEdit `cell_id: 1`. Change two things vs current: `BEST_NUM_FEATS` and the CSV path. Replace `<XGB_BEST_NUM_FEATS>` with the integer recorded in Task 2 Step 5. Full new source:
```python
RANDOM_SEED = 32
# Run 15: full (ranked + aggregated) feature set. BEST_NUM_FEATS comes from the FULL run's
# get_best_num_features(.005) in rfe.ipynb (NOT the rank-only 17); source CSV is the _full trace.
BEST_NUM_FEATS = <XGB_BEST_NUM_FEATS>
SELECTED_RFE_CSV = 'rfe_features_kfolds_full.csv'
TARGET = 'target_win'

RANDOM_XGB_PARAMS = {
    'learning_rate': [.03, .05, .1, .15, .2],
    'max_depth': [2, 3],
    'subsample': [.5, .7, .9],
    'min_child_weight': [10, 20, 50, 100],
    'gamma': [.5, 1, 5, 10, 100],
    'n_estimators': [10, 20,30, 40, 50],
    'early_stopping_rounds': [10, 20, 50],
    'importance_type': ['total_gain'],
    'eval_metric': ['auc']
}

RANDOM_SEARCH_PARAMS = dict(
    n_iter=100,
    cv=5,
    n_jobs=-1,
    random_state=RANDOM_SEED,
    scoring='roc_auc',
)

BEST_FEATS_FROM_RFE = list(pd.read_csv(
    f'../../../../../data/predict_games/model_features_in/{SELECTED_RFE_CSV}',
    index_col=0
).loc[BEST_NUM_FEATS, :].dropna().values)

MODEL_INPUTS_DF = pd.read_parquet(
    '../../../../../data/predict_games/input_data/schedule_and_weekly.parquet'
)

MODEL_INPUTS_DF = MODEL_INPUTS_DF.sample(
    frac=1, random_state=RANDOM_SEED
).reset_index(drop=True)

np.random.seed(RANDOM_SEED)
```
Note: keep the original `import` cell (cell 0) unchanged — `pd`/`np` are imported there.

- [ ] **Step 2: Add an explicit overall hold-out AUROC print (parity with BART)**

The notebook reports accuracy (`best_model.score`, cell 9) and per-week AUROC (cell 12) but no single overall hold-out AUROC. Insert a new cell **immediately after cell 10** (`holdout_df.loc[:, 'preds'] = ...`) using NotebookEdit with `edit_mode: insert` after `cell_id: 10`:
```python
# Run 15: single overall hold-out AUROC, computed exactly like bart.ipynb, for a clean
# apples-to-apples comparison against the 0.705 champion.
overall_holdout_auroc = roc_auc_score(holdout_df['target_win'], holdout_df['preds'])
print(f'2024+2025 hold-out ROC-AUC (overall): {overall_holdout_auroc:.4f}  '
      f'[{BEST_NUM_FEATS} features, full set]')
```

- [ ] **Step 3: Commit the config change**

```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
git add notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/grid_search.ipynb
git commit -m "Point grid_search.ipynb at Run 15 full trace; add overall hold-out AUROC print"
```

---

## Task 4: Execute the XGBoost grid search + hold-out eval

**Files:**
- Reads: `CV_DIR/grid_search.ipynb`, `rfe_features_kfolds_full.csv`

- [ ] **Step 1: Run headless (background)**

Run with `run_in_background: true`:
```bash
cd notebooks/model_training/predict_games/schedule_and_weekly/cross_validation
jupyter nbconvert --to notebook --execute --inplace grid_search.ipynb \
  --ExecutePreprocessor.timeout=-1 > /tmp/xgb_grid_nbconvert.out 2>&1
```
Faster than Task 2 (only `BEST_NUM_FEATS` features), but the 100-iter random search still takes a while.

- [ ] **Step 2: Verify clean completion and read the AUROC**

```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
tail -n 15 /tmp/xgb_grid_nbconvert.out      # must NOT contain a Traceback
python -c "
import json
nb=json.load(open('notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/grid_search.ipynb'))
for c in nb['cells']:
    if c['cell_type']=='code' and 'hold-out ROC-AUC (overall)' in ''.join(c['source']):
        for o in c.get('outputs',[]):
            print(''.join(o.get('text', '')))
"
```
Expected: a line `2024+2025 hold-out ROC-AUC (overall): 0.7xxx [N features, full set]`. Record as **XGB_FULL_AUROC**.

- [ ] **Step 3: Matched-baseline check (avoid a metric-definition mismatch)**

The recorded champion XGBoost figure (~0.707) predates the overall-AUROC cell, so confirm the comparison is apples-to-apples: the champion number must be the *overall* `roc_auc_score(y_holdout, predict_proba[:,1])`, computed the same way as Step 2. If there is any doubt about how the 0.707 was derived, re-derive the rank-only baseline identically — temporarily set `SELECTED_RFE_CSV = 'rfe_features_kfolds.csv'` and `BEST_NUM_FEATS = 17` in a scratch copy, run, and read the overall-AUROC line. Record that as **XGB_RANKONLY_AUROC_MATCHED**. (Skip only if you are certain the existing 0.707 is already the overall metric.)

- [ ] **Step 4: Commit the executed notebook**

```bash
git add notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/grid_search.ipynb
git commit -m "Run 15: XGBoost full-set grid search, hold-out AUROC <XGB_FULL_AUROC>"
```

---

## Task 5: Configure bart_rfe.ipynb for the full run

**Files:**
- Modify: `CV_DIR/bart_rfe.ipynb` cell 1 (start-pool source + size), cell 5 (output filename)

- [ ] **Step 1: Pick the start-pool row nearest ~90 from the FULL trace**

```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
python -c "
import pandas as pd
idx = pd.read_csv('data/predict_games/model_features_in/rfe_features_kfolds_full.csv', index_col=0).index
nearest = min(idx, key=lambda n: abs(n-90))
print('produced sizes near 90:', sorted([n for n in idx if 60 <= n <= 130]))
print('START_POOL_SIZE to use:', nearest)
"
```
Record the printed integer as **BART_START_POOL** (expected ≈ 88).

- [ ] **Step 2: Edit cell 1 — read the full trace at BART_START_POOL**

NotebookEdit `cell_id: 1`. Two changes vs current: `START_POOL_SIZE` value and the CSV filename in the `pd.read_csv(...)`. Replace `<BART_START_POOL>` with the integer from Step 1. Full new source:
```python
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb
import os, sys, time, threading
module_path = os.path.abspath(os.path.join('../../../../../'))
if module_path not in sys.path:
    sys.path.append(module_path)
from data_science_utilities.models.bart.feature_selection.backward_elimination import (
    BartBackwardElimination,
)

RANDOM_SEED = 32
TARGET = 'target_win'
BART_NAN_SENTINEL = -100.0
# Run 15: start pool drawn from the FULL XGBoost RFE trace (rfe_features_kfolds_full.csv).
# Kept near ~90 (Run-14 rationale: ~3x the XGBoost CV peak, above the ~60 usable bound)
# so BART starts MUCH lower than XGBoost's 2,348 and fit times stay tractable. Caveat: the
# pool is XGBoost-importance-pruned, so a BART-only aggregated signal could be lost first.
START_POOL_SIZE = <BART_START_POOL>

# Per-fit / per-iteration progress to a sidecar file: headless nbconvert buffers a
# cell's stdout until the cell finishes, so this is the only way to watch live.
_LOG_PATH = '/tmp/bart_rfe_progress.log'
_log_lock = threading.Lock()


def log_progress(message):
    line = f"{time.strftime('%H:%M:%S')} {message}\n"
    with _log_lock:
        with open(_LOG_PATH, 'a') as handle:
            handle.write(line)


START_FEATURES = list(pd.read_csv(
    '../../../../../data/predict_games/model_features_in/rfe_features_kfolds_full.csv',
    index_col=0,
).loc[START_POOL_SIZE].dropna().values)

MODEL_INPUTS_DF = pd.read_parquet(
    '../../../../../data/predict_games/input_data/schedule_and_weekly.parquet'
).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

train_df = MODEL_INPUTS_DF[MODEL_INPUTS_DF['season'] < 2022]
valid_df = MODEL_INPUTS_DF[
    (MODEL_INPUTS_DF['season'] >= 2022) & (MODEL_INPUTS_DF['season'] < 2024)
]
y_train = train_df[TARGET].to_numpy(dtype=int)
y_valid = valid_df[TARGET].to_numpy(dtype=int)
log_progress(f'START run: {len(START_FEATURES)} starting features')
print(len(START_FEATURES), 'starting features')
```

- [ ] **Step 3: Edit cell 5 — write the selected set to a non-clobbering filename**

NotebookEdit `cell_id: 5` with this full new source:
```python
pd.DataFrame({'feature': best_features}).to_csv(
    '../../../../../data/predict_games/model_features_in/bart_rfe_features_full.csv',
    index=False,
)
sorted(best_features)
```

- [ ] **Step 4: Config echo — confirm the start count without the long run**

```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
python -c "
import pandas as pd
START=<BART_START_POOL>
feats=pd.read_csv('data/predict_games/model_features_in/rfe_features_kfolds_full.csv', index_col=0).loc[START].dropna()
print('BART start features:', len(feats))
"
```
Expected: prints **BART_START_POOL** (≈ 88). If it errors with a KeyError, that row is not in the full trace — re-pick via Task 5 Step 1.

- [ ] **Step 5: Commit the config change**

```bash
git add notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/bart_rfe.ipynb
git commit -m "Configure bart_rfe.ipynb for Run 15: start pool from full trace (~90), _full output"
```

---

## Task 6: Execute the BART full RFE + hold-out eval (long run)

**Files:**
- Reads: `CV_DIR/bart_rfe.ipynb`
- Produces: `data/predict_games/model_features_in/bart_rfe_features_full.csv`

- [ ] **Step 1: Clear the stale progress log**

```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
: > /tmp/bart_rfe_progress.log
```

- [ ] **Step 2: Run headless (background)**

Run with `run_in_background: true`:
```bash
cd notebooks/model_training/predict_games/schedule_and_weekly/cross_validation
jupyter nbconvert --to notebook --execute --inplace bart_rfe.ipynb \
  --ExecutePreprocessor.timeout=-1 > /tmp/bart_rfe_nbconvert.out 2>&1
```
Hours: 6 replicate fits per iteration, ~30 min/fit at the high end, eliminating 20%/iter from ~88 down to 10, then a final 4-chain hold-out fit.

- [ ] **Step 3: Poll progress**

```bash
tail -n 6 /tmp/bart_rfe_progress.log
```
Expected: `START run: 88 starting features`, then `fit done: ...` and `=== ITER complete: N features, mean validation 0.6xxx ===` lines with decreasing N.

- [ ] **Step 4: Verify clean completion and read the metrics**

```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
tail -n 20 /tmp/bart_rfe_nbconvert.out        # must NOT contain a Traceback
ls -la data/predict_games/model_features_in/bart_rfe_features_full.csv
python -c "
import json
nb=json.load(open('notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/bart_rfe.ipynb'))
for c in nb['cells']:
    if c['cell_type']=='code':
        for o in c.get('outputs',[]):
            t=''.join(o.get('text',''))
            if 'hold-out' in t or 'validation peak' in t or 'Brier' in t or 'Log loss' in t:
                print(t)
"
```
Expected: `validation peak at N features ...`, then `2024+2025 hold-out ROC-AUC: 0.7xxx`, accuracy, Brier, log loss. Record the AUROC as **BART_FULL_AUROC**. If `bart_rfe_features_full.csv` is missing or a traceback appears, the run failed — do not record results.

- [ ] **Step 5: Commit the executed notebook + selected set**

```bash
git add notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/bart_rfe.ipynb \
        data/predict_games/model_features_in/bart_rfe_features_full.csv
git commit -m "Run 15: BART full-feature backward elimination, hold-out AUROC <BART_FULL_AUROC>"
```

---

## Task 7: Record Run 15 verdict

**Files:**
- Modify: `CV_DIR/bart_rfe.ipynb` (add a results markdown cell) OR the project's experiment log / README where prior runs are recorded.

- [ ] **Step 1: Locate where prior runs are recorded**

```bash
cd /Users/joshuaharkness/ClaudeProjects/nfl-predictions
grep -rln "Run 14\|Run 13\|experiment-log\|hold-out AUROC" --include=*.md --include=*.ipynb . | grep -v ".git/" | head
```
Use the same location/format prior runs use (commit messages already capture per-run numbers; add a markdown summary cell to `bart_rfe.ipynb` if that matches the existing pattern).

- [ ] **Step 2: Write the verdict**

Record, using the values captured above:
- XGBoost full set: **XGB_BEST_NUM_FEATS** features → hold-out AUROC **XGB_FULL_AUROC** (vs rank-only **XGB_RANKONLY_AUROC_MATCHED** / ~0.707).
- BART full set (start pool **BART_START_POOL**): peak at N features → hold-out AUROC **BART_FULL_AUROC** (vs champion **0.705**).
- Verdict per estimator: **beats / matches / underperforms**. Do not call a delta < ~0.005 meaningful (small 2024–25 hold-out).
- Restate the caveat: the BART result is "best among XGBoost's top ~90," not "best among all 2,348."

- [ ] **Step 3: Commit the recorded result**

```bash
git add -A
git commit -m "Record Run 15: full ranked+aggregated set vs rank-only champion (XGBoost <XGB_FULL_AUROC>, BART <BART_FULL_AUROC>)"
```

- [ ] **Step 4: (Optional) Open a PR**

Only if the user asks — follow the project's per-run PR cadence.

---

## Self-Review notes

- **Spec coverage:** Phase A (rfe + grid_search) → Tasks 1–4; Phase B (bart_rfe) → Tasks 5–6; `_full` artifact preservation → cell-11/cell-5 edits + verify steps; `max_iter`→60 → Task 1 Step 3; BART start pool ~90 from full trace → Task 5; verification rule (exit code + config echo) → Task 1 Step 5, Task 2 Step 4, Task 5 Step 4; success criteria/verdict → Task 7. All covered.
- **Matched-baseline gap:** the champion's 0.707 may use a different AUROC aggregation than the added overall print; Task 4 Step 3 closes this by re-deriving the rank-only baseline identically when in doubt.
- **Placeholders:** `<XGB_BEST_NUM_FEATS>`, `<BART_START_POOL>`, `<XGB_FULL_AUROC>`, `<BART_FULL_AUROC>` are **runtime-captured values**, not unspecified design decisions — each has an explicit capture step (Task 2 Step 5, Task 5 Step 1, Task 4 Step 2, Task 6 Step 4). This is expected for an experiment plan.
