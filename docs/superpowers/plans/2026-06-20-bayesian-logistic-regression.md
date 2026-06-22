# Bayesian Logistic Regression vs. BART Champion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PyMC Bayesian logistic regression that predicts `target_win` on the exact same protocol as BART Run-6 (fixed 2024+2025 season hold-out, frozen 54-feature set), report it side-by-side with BART (0.705) + the Bayesian extras (calibration, posterior uncertainty, sampler diagnostics), and expose a reusable posterior.

**Architecture:** Small single-purpose modules under `data_science_utilities/models/bayes_logistic/` (`data_prep`, `model`, `evaluate`, `posterior`), driven by a thin orchestrator at `scripts/experiments/bayes_logistic.py`. Leakage-safe preprocessing (z-score + median-impute + missingness flag, all fit on the train fold only). Two prior variants (weakly-informative Normal, regularized horseshoe). Posterior saved as NetCDF + a per-coefficient JSON summary that also carries the standardizer moments, so a later Madden-update model can seed its prior.

**Tech Stack:** Python 3.11, PyMC 5.23.0, ArviZ 0.23.4, pytensor 2.31.7, scikit-learn 1.4.2, pandas 2.2.3, numpy 1.26.4. Tests: `unittest`-style `TestCase` classes run under pytest 9.1.0.

## Global Constraints

- **Conda env:** all commands run via `conda run --no-capture-output -n nfl-predictions <cmd>`.
- **Run from repo root.** No `__init__.py` anywhere (PEP-420 namespace packages); first-party imports (`from data_science_utilities...`, `from src...`) only resolve when the repo root is on `sys.path`. Always run tests with `python -m pytest` (the `-m` puts cwd on the path) **from the repo root**: `/Users/joshuaharkness/ClaudeProjects/nfl-predictions`.
- **Reproducibility seed:** `RANDOM_SEED = 32` everywhere (shuffle, `np.random.seed`, sampler).
- **Protocol is fixed and must not change** (this is the comparability contract):
  - shuffle: `df.sample(frac=1, random_state=32).reset_index(drop=True)` then `np.random.seed(32)`
  - split: `train season<2022`, `valid 2022–2023`, **`holdout season>=2024`** (headline)
  - target: `target_win` (int64)
  - per-row eval on all holdout rows (two perspective rows per game — do **not** dedup).
- **Never** feed leakage/meta cols: `game_id, season, season_type, opp_team, opp_score, target_team, target_score, h_win`. (The frozen feature list contains none of these by construction.)
- **No `-100` sentinel** in this linear model. Missing values get median-impute + a `*_was_missing` flag, fit on train only. (Inert on the chosen NaN-free set, but required for correctness/generality.)
- **Paths (absolute-from-root):**
  - parquet: `data/predict_games/input_data/schedule_and_weekly.parquet`
  - frozen features: `data/predict_games/model_features_in/bayes_logistic_features.csv`
  - outputs: `data/predict_games/bayes_logistic/`
  - module: `data_science_utilities/models/bayes_logistic/`
  - tests: `tests/data_science_utilities/models/bayes_logistic/`

---

## File Structure

| File | Responsibility |
|---|---|
| `data/predict_games/model_features_in/bayes_logistic_features.csv` | Frozen 54-feature Run-6 set (data artifact) |
| `data_science_utilities/models/bayes_logistic/data_prep.py` | Load features, season-split, leakage-safe preprocessing |
| `data_science_utilities/models/bayes_logistic/evaluate.py` | Metric functions matching `bart.ipynb` |
| `data_science_utilities/models/bayes_logistic/model.py` | PyMC model build / sample / predict / diagnostics |
| `data_science_utilities/models/bayes_logistic/posterior.py` | NetCDF trace + JSON coefficient summary export |
| `scripts/experiments/bayes_logistic.py` | Orchestrator: wire it all, print table, write artifacts, emit README line |
| `tests/data_science_utilities/models/bayes_logistic/test_*.py` | Unit tests (one per module) |
| `README.md` | Run-27 entry + verdict (Task 8) |

---

## Task 1: Freeze the 54-feature list + loader

**Files:**
- Create: `data/predict_games/model_features_in/bayes_logistic_features.csv`
- Create: `data_science_utilities/models/bayes_logistic/data_prep.py`
- Create: `tests/data_science_utilities/models/bayes_logistic/test_data_prep.py`

**Interfaces:**
- Produces: `data_prep.load_feature_list(path: str) -> list[str]`; module constants `FEATURE_LIST_PATH`, `PARQUET_PATH`, `TARGET = "target_win"`, `RANDOM_SEED = 32`.

- [ ] **Step 1: Create the frozen feature CSV from git provenance**

Run (from repo root):
```bash
conda run --no-capture-output -n nfl-predictions python - <<'PY'
import pandas as pd, io, subprocess
raw = subprocess.run(
    ["git","show","b1410bd:data/predict_games/model_features_in/rfe_features_kfolds.csv"],
    capture_output=True, text=True).stdout
feats = list(pd.read_csv(io.StringIO(raw), index_col=0).loc[54, :].dropna().values)
assert len(feats) == 54, len(feats)
out = "data/predict_games/model_features_in/bayes_logistic_features.csv"
with open(out, "w") as f:
    f.write("# Frozen 54-feature set from BART Run 6 (README run 6, hold-out AUROC 0.705).\n")
    f.write("# Provenance: b1410bd:data/predict_games/model_features_in/rfe_features_kfolds.csv .loc[54]\n")
    f.write("# (current rfe_features_kfolds.csv no longer has a 54-count row after Phase 2/3 RFE re-runs)\n")
    f.write("feature\n")
    for c in feats:
        f.write(c + "\n")
print("wrote", out, len(feats), "features")
PY
```
Expected: `wrote data/predict_games/model_features_in/bayes_logistic_features.csv 54 features`

- [ ] **Step 2: Write the failing test**

Create `tests/data_science_utilities/models/bayes_logistic/test_data_prep.py`:
```python
import unittest

import pandas as pd

from data_science_utilities.models.bayes_logistic import data_prep

# The 54 features of the BART Run-6 set (b1410bd rfe_features_kfolds.csv .loc[54]).
EXPECTED_54 = [
    "is_home_target",
    "target_def_cumulative_avg_points_allowed_rank",
    "opp_def_cumulative_avg_points_allowed_rank",
    "off_opp_passing_epa_cumulative_average_rank",
    "target_off_cumulative_avg_score_rank",
    "opp_off_cumulative_avg_score_rank",
    "off_target_passing_epa_cumulative_average_rank",
    "off_target_pacr",
    "def_opp_rushing_yards_cumulative_average_rank",
    "off_target_interceptions_cumulative_average_rank",
    "def_target_carries_cumulative_average_rank",
    "def_target_rushing_fumbles_lost_cumulative_average_rank",
    "def_opp_attempts",
    "def_target_attempts",
    "def_target_rushing_yards_cumulative_average_rank",
    "def_opp_rushing_tds_cumulative_average_rank",
    "def_opp_racr",
    "def_target_rushing_tds_cumulative_average_rank",
    "off_target_sack_yards_cumulative_average_rank",
    "opp_game_count",
    "def_opp_rushing_first_downs_cumulative_average_rank",
    "off_opp_passing_epa",
    "off_target_passing_epa",
    "off_opp_rushing_epa",
    "def_opp_pacr_cumulative_average_rank_change",
    "off_target_passing_yards_after_catch",
    "off_target_passing_tds_cumulative_average_rank",
    "off_opp_sacks_cumulative_average_rank",
    "off_opp_carries_cumulative_average_rank",
    "off_target_sacks_cumulative_average_rank",
    "off_opp_receiving_fumbles_cumulative_average_rank",
    "def_opp_passing_tds_cumulative_average_rank",
    "def_target_receiving_fumbles_cumulative_average_rank_change",
    "off_opp_interceptions_cumulative_average_rank",
    "off_opp_passing_yards_after_catch",
    "off_opp_sack_yards_cumulative_average_rank",
    "def_opp_passing_yards",
    "def_target_sack_yards",
    "def_target_interceptions_cumulative_average_rank",
    "off_target_attempts_cumulative_average_rank_change",
    "def_opp_receiving_fumbles_cumulative_average_rank",
    "def_target_sacks_cumulative_average_rank",
    "def_opp_sack_yards",
    "def_target_rushing_fumbles_cumulative_average_rank",
    "off_opp_rushing_first_downs_cumulative_average_rank",
    "off_target_sack_fumbles_cumulative_average_rank",
    "off_target_rushing_tds_cumulative_average_rank",
    "off_target_passing_yards",
    "def_target_passing_tds_cumulative_average_rank",
    "def_opp_sack_fumbles_cumulative_average_rank",
    "off_target_carries_cumulative_average_rank_change",
    "def_target_sack_yards_cumulative_average_rank",
    "def_opp_passing_epa",
    "def_target_passing_epa",
]

LEAKAGE_COLS = {
    "game_id", "season", "season_type", "opp_team", "opp_score",
    "target_team", "target_score", "h_win",
}


class TestLoadFeatureList(unittest.TestCase):
    def test_returns_exact_54_run6_features_in_order(self):
        feats = data_prep.load_feature_list(data_prep.FEATURE_LIST_PATH)
        self.assertEqual(feats, EXPECTED_54)

    def test_no_leakage_columns_present(self):
        feats = data_prep.load_feature_list(data_prep.FEATURE_LIST_PATH)
        self.assertEqual(set(feats) & LEAKAGE_COLS, set())

    def test_all_features_exist_in_parquet(self):
        feats = data_prep.load_feature_list(data_prep.FEATURE_LIST_PATH)
        cols = pd.read_parquet(data_prep.PARQUET_PATH, columns=None).columns
        missing = [f for f in feats if f not in cols]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data_science_utilities/models/bayes_logistic/test_data_prep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_science_utilities.models.bayes_logistic'`

- [ ] **Step 4: Create the module with the loader**

Create `data_science_utilities/models/bayes_logistic/data_prep.py`:
```python
"""Leakage-safe data preparation for the Bayesian logistic regression model.

Matches the BART champion protocol (bart.ipynb): a fixed season split with the
frozen 54-feature Run-6 set. No PyMC import here so this stays fast and pure.
"""
import os

import numpy as np
import pandas as pd

RANDOM_SEED = 32
TARGET = "target_win"

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
FEATURE_LIST_PATH = os.path.join(
    _REPO_ROOT, "data", "predict_games", "model_features_in",
    "bayes_logistic_features.csv",
)
PARQUET_PATH = os.path.join(
    _REPO_ROOT, "data", "predict_games", "input_data", "schedule_and_weekly.parquet",
)


def load_feature_list(path=FEATURE_LIST_PATH):
    """Read the frozen feature list (one feature per row, '#'-comment lines skipped)."""
    df = pd.read_csv(path, comment="#")
    return df["feature"].tolist()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data_science_utilities/models/bayes_logistic/test_data_prep.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add data/predict_games/model_features_in/bayes_logistic_features.csv \
        data_science_utilities/models/bayes_logistic/data_prep.py \
        tests/data_science_utilities/models/bayes_logistic/test_data_prep.py
git commit -m "feat(bayes): freeze 54-feature Run-6 set + feature loader"
```

---

## Task 2: Season split + target extraction

**Files:**
- Modify: `data_science_utilities/models/bayes_logistic/data_prep.py`
- Modify: `tests/data_science_utilities/models/bayes_logistic/test_data_prep.py`

**Interfaces:**
- Consumes: `RANDOM_SEED`, `TARGET` from Task 1.
- Produces: `data_prep.Split` dataclass (`.train`, `.valid`, `.holdout` DataFrames); `data_prep.load_and_split(parquet_path=PARQUET_PATH, seed=RANDOM_SEED) -> Split`; `data_prep.get_target(df) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

Append to `tests/data_science_utilities/models/bayes_logistic/test_data_prep.py`:
```python
import numpy as np


def _synthetic_seasons():
    # two rows per season 2019..2025 so splits are non-empty
    rows = []
    for season in range(2019, 2026):
        for win in (0, 1):
            rows.append({"season": season, "target_win": win, "week": 1, "feat": season + win})
    return pd.DataFrame(rows)


class TestLoadAndSplit(unittest.TestCase):
    def test_split_boundaries(self):
        df = _synthetic_seasons()
        split = data_prep.split_frames(df)  # split a given (already shuffled) frame
        self.assertTrue((split.train["season"] < 2022).all())
        self.assertTrue(((split.valid["season"] >= 2022) & (split.valid["season"] < 2024)).all())
        self.assertTrue((split.holdout["season"] >= 2024).all())
        # every row lands in exactly one fold
        self.assertEqual(len(split.train) + len(split.valid) + len(split.holdout), len(df))

    def test_seed_is_reproducible(self):
        df = _synthetic_seasons()
        a = data_prep.shuffle(df, seed=32)
        b = data_prep.shuffle(df, seed=32)
        pd.testing.assert_frame_equal(a, b)

    def test_get_target_is_int_array(self):
        df = _synthetic_seasons()
        y = data_prep.get_target(df)
        self.assertEqual(y.dtype, np.dtype(int))
        self.assertEqual(set(np.unique(y)), {0, 1})


class TestLoadAndSplitRealData(unittest.TestCase):
    def test_holdout_is_2024_plus_2025(self):
        split = data_prep.load_and_split()
        self.assertEqual(sorted(split.holdout["season"].unique().tolist()), [2024, 2025])
        self.assertTrue((split.train["season"] < 2022).all())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data_science_utilities/models/bayes_logistic/test_data_prep.py -v -k "Split"`
Expected: FAIL — `AttributeError: module ... has no attribute 'split_frames'`

- [ ] **Step 3: Implement split helpers**

Append to `data_science_utilities/models/bayes_logistic/data_prep.py`:
```python
from dataclasses import dataclass


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    holdout: pd.DataFrame


def shuffle(df, seed=RANDOM_SEED):
    """Champion-protocol shuffle: same call as bart.ipynb."""
    out = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    np.random.seed(seed)
    return out


def split_frames(df):
    """Fixed season split: train<2022, valid 2022-2023, holdout>=2024."""
    train = df[df["season"] < 2022]
    valid = df[(df["season"] >= 2022) & (df["season"] < 2024)]
    holdout = df[df["season"] >= 2024].copy()
    return Split(train=train, valid=valid, holdout=holdout)


def load_and_split(parquet_path=PARQUET_PATH, seed=RANDOM_SEED):
    df = pd.read_parquet(parquet_path)
    return split_frames(shuffle(df, seed=seed))


def get_target(df):
    return df[TARGET].to_numpy(dtype=int)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data_science_utilities/models/bayes_logistic/test_data_prep.py -v`
Expected: all passed (Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add data_science_utilities/models/bayes_logistic/data_prep.py \
        tests/data_science_utilities/models/bayes_logistic/test_data_prep.py
git commit -m "feat(bayes): season split + reproducible shuffle + target extraction"
```

---

## Task 3: Leakage-safe preprocessor (impute + flag + standardize)

**Files:**
- Modify: `data_science_utilities/models/bayes_logistic/data_prep.py`
- Modify: `tests/data_science_utilities/models/bayes_logistic/test_data_prep.py`

**Interfaces:**
- Consumes: feature list + `Split` from Tasks 1–2.
- Produces: `data_prep.TrainFitPreprocessor(feature_names: list[str])` with `.fit(train_df) -> self`, `.transform(df) -> np.ndarray`, and attributes `.feature_names_out_ : list[str]`, `.means_ : np.ndarray`, `.stds_ : np.ndarray` (aligned to `feature_names_out_`). Missingness flags are appended for any base feature with NaN in train; flags are raw 0/1 (recorded as mean 0, std 1). Zero-variance base features get `std=1` (column centered to 0).

- [ ] **Step 1: Write the failing test**

Append to `tests/data_science_utilities/models/bayes_logistic/test_data_prep.py`:
```python
class TestTrainFitPreprocessor(unittest.TestCase):
    def _frame(self, vals, extra=None):
        d = {"a": vals}
        if extra is not None:
            d["b"] = extra
        return pd.DataFrame(d)

    def test_standardizes_using_train_moments_only(self):
        train = self._frame([0.0, 2.0, 4.0])           # mean 2, std 2 (ddof=0)
        pre = data_prep.TrainFitPreprocessor(["a"]).fit(train)
        # a *different* frame must be standardized with TRAIN mean/std, not its own
        other = self._frame([2.0, 2.0, 2.0])
        out = pre.transform(other)
        np.testing.assert_allclose(out[:, 0], [0.0, 0.0, 0.0])  # (2-2)/2 == 0
        np.testing.assert_allclose(pre.means_, [2.0])
        np.testing.assert_allclose(pre.stds_, [2.0])

    def test_median_impute_and_missing_flag_on_train_nan(self):
        train = self._frame([1.0, np.nan, 3.0])         # train median 2.0
        pre = data_prep.TrainFitPreprocessor(["a"]).fit(train)
        self.assertIn("a_was_missing", pre.feature_names_out_)
        test = self._frame([np.nan, 5.0, 5.0])
        out = pre.transform(test)
        flag_idx = pre.feature_names_out_.index("a_was_missing")
        np.testing.assert_allclose(out[:, flag_idx], [1.0, 0.0, 0.0])  # flag from raw NaN
        # imputed value standardized: first row used train median 2.0 before scaling
        base_idx = pre.feature_names_out_.index("a")
        mean_a, std_a = pre.means_[base_idx], pre.stds_[base_idx]
        np.testing.assert_allclose(out[0, base_idx], (2.0 - mean_a) / std_a)

    def test_no_flag_when_train_has_no_nan(self):
        train = self._frame([1.0, 2.0, 3.0])
        pre = data_prep.TrainFitPreprocessor(["a"]).fit(train)
        self.assertEqual(pre.feature_names_out_, ["a"])  # no flag column

    def test_zero_variance_guard(self):
        train = self._frame([5.0, 5.0, 5.0])            # std 0
        pre = data_prep.TrainFitPreprocessor(["a"]).fit(train)
        out = pre.transform(train)
        self.assertFalse(np.isnan(out).any())
        np.testing.assert_allclose(out[:, 0], [0.0, 0.0, 0.0])

    def test_transform_output_has_no_nan(self):
        train = self._frame([1.0, np.nan, 3.0], extra=[1.0, 2.0, 3.0])
        pre = data_prep.TrainFitPreprocessor(["a", "b"]).fit(train)
        out = pre.transform(self._frame([np.nan, 1.0, 2.0], extra=[9.0, 8.0, 7.0]))
        self.assertFalse(np.isnan(out).any())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data_science_utilities/models/bayes_logistic/test_data_prep.py -v -k "Preprocessor"`
Expected: FAIL — `AttributeError: module ... has no attribute 'TrainFitPreprocessor'`

- [ ] **Step 3: Implement the preprocessor**

Append to `data_science_utilities/models/bayes_logistic/data_prep.py`:
```python
class TrainFitPreprocessor:
    """Median-impute (+ missingness flag) then z-score, all fit on the train fold.

    Output column order: the base features (in the given order), then a
    ``<feat>_was_missing`` flag for each base feature that had any NaN in train.
    Base features are standardized; flags are left as raw 0/1.
    """

    def __init__(self, feature_names):
        self.feature_names = list(feature_names)

    def fit(self, train_df):
        X = train_df[self.feature_names]
        self.medians_ = X.median(numeric_only=False)
        self.nan_cols_ = [c for c in self.feature_names if X[c].isna().any()]
        imputed = X.fillna(self.medians_)
        base_mean = imputed.mean().to_numpy(dtype=float)
        base_std = imputed.std(ddof=0).to_numpy(dtype=float)
        base_std = np.where(base_std == 0.0, 1.0, base_std)  # zero-variance guard
        self.feature_names_out_ = list(self.feature_names) + [
            f"{c}_was_missing" for c in self.nan_cols_
        ]
        flag_mean = np.zeros(len(self.nan_cols_))
        flag_std = np.ones(len(self.nan_cols_))
        self.means_ = np.concatenate([base_mean, flag_mean])
        self.stds_ = np.concatenate([base_std, flag_std])
        return self

    def transform(self, df):
        X = df[self.feature_names]
        flags = np.column_stack(
            [X[c].isna().to_numpy(dtype=float) for c in self.nan_cols_]
        ) if self.nan_cols_ else np.empty((len(X), 0))
        imputed = X.fillna(self.medians_).to_numpy(dtype=float)
        base = (imputed - self.means_[: len(self.feature_names)]) / self.stds_[
            : len(self.feature_names)
        ]
        out = np.column_stack([base, flags]) if self.nan_cols_ else base
        assert not np.isnan(out).any(), "NaN survived preprocessing"
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data_science_utilities/models/bayes_logistic/test_data_prep.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add data_science_utilities/models/bayes_logistic/data_prep.py \
        tests/data_science_utilities/models/bayes_logistic/test_data_prep.py
git commit -m "feat(bayes): leakage-safe impute+flag+standardize preprocessor"
```

---

## Task 4: Evaluation metrics (match bart.ipynb)

**Files:**
- Create: `data_science_utilities/models/bayes_logistic/evaluate.py`
- Create: `tests/data_science_utilities/models/bayes_logistic/test_evaluate.py`

**Interfaces:**
- Produces: `evaluate.auroc`, `evaluate.accuracy`, `evaluate.brier`, `evaluate.logloss`, `evaluate.per_week_auroc(y, p, weeks) -> (pd.Series, float)`, `evaluate.reliability_curve(y, p, n_bins=10) -> (np.ndarray, np.ndarray)`, `evaluate.width_stratified_brier(y, p, p_std, n_quartiles=4) -> pd.Series`, `evaluate.evaluate(y, p, p_std=None, weeks=None) -> dict`.

- [ ] **Step 1: Write the failing test**

Create `tests/data_science_utilities/models/bayes_logistic/test_evaluate.py`:
```python
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from data_science_utilities.models.bayes_logistic import evaluate


class TestMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = (rng.uniform(size=200) < 0.5).astype(int)
        self.p = np.clip(self.y * 0.6 + rng.uniform(size=200) * 0.4, 0.01, 0.99)

    def test_auroc_matches_sklearn(self):
        self.assertAlmostEqual(evaluate.auroc(self.y, self.p), roc_auc_score(self.y, self.p))

    def test_accuracy_uses_half_threshold(self):
        expected = ((self.p > 0.5).astype(int) == self.y).mean()
        self.assertAlmostEqual(evaluate.accuracy(self.y, self.p), expected)

    def test_brier_matches_sklearn(self):
        self.assertAlmostEqual(evaluate.brier(self.y, self.p), brier_score_loss(self.y, self.p))

    def test_logloss_matches_sklearn(self):
        self.assertAlmostEqual(evaluate.logloss(self.y, self.p), log_loss(self.y, self.p))

    def test_per_week_auroc_mean(self):
        weeks = np.repeat([1, 2], 100)
        series, mean = evaluate.per_week_auroc(self.y, self.p, weeks)
        self.assertEqual(sorted(series.index.tolist()), [1, 2])
        self.assertAlmostEqual(mean, series.mean())

    def test_reliability_curve_quantile_shape(self):
        prob_true, prob_pred = evaluate.reliability_curve(self.y, self.p, n_bins=10)
        self.assertEqual(len(prob_true), len(prob_pred))
        self.assertLessEqual(len(prob_true), 10)

    def test_width_stratified_brier_returns_quartiles(self):
        rng = np.random.default_rng(1)
        p_std = rng.uniform(0.02, 0.2, size=200)
        s = evaluate.width_stratified_brier(self.y, self.p, p_std, n_quartiles=4)
        self.assertEqual(len(s), 4)

    def test_evaluate_dict_keys(self):
        weeks = np.repeat([1, 2], 100)
        p_std = np.full(200, 0.1)
        res = evaluate.evaluate(self.y, self.p, p_std=p_std, weeks=weeks)
        for k in ["auroc", "accuracy", "brier", "logloss", "per_week_auroc_mean"]:
            self.assertIn(k, res)
        self.assertIsInstance(res["auroc"], float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data_science_utilities/models/bayes_logistic/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: ... 'evaluate'`

- [ ] **Step 3: Implement the metrics**

Create `data_science_utilities/models/bayes_logistic/evaluate.py`:
```python
"""Holdout metrics, mirroring the exact calls in bart.ipynb for comparability."""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss


def auroc(y_true, preds):
    return float(roc_auc_score(y_true, preds))


def accuracy(y_true, preds):
    return float(((np.asarray(preds) > 0.5).astype(int) == np.asarray(y_true)).mean())


def brier(y_true, preds):
    return float(brier_score_loss(y_true, preds))


def logloss(y_true, preds):
    return float(log_loss(y_true, preds))


def per_week_auroc(y_true, preds, weeks):
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(preds), "week": np.asarray(weeks)})
    series = df.groupby("week").apply(
        lambda x: roc_auc_score(x["y"], x["p"]), include_groups=False
    )
    return series, float(series.mean())


def reliability_curve(y_true, preds, n_bins=10):
    prob_true, prob_pred = calibration_curve(
        y_true, preds, n_bins=n_bins, strategy="quantile"
    )
    return prob_true, prob_pred


def width_stratified_brier(y_true, preds, preds_std, n_quartiles=4):
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(preds), "std": np.asarray(preds_std)})
    labels = ["narrowest", "q2", "q3", "widest"][:n_quartiles]
    df["q"] = pd.qcut(df["std"], n_quartiles, labels=labels)
    return df.groupby("q", observed=True).apply(
        lambda x: brier_score_loss(x["y"], x["p"]), include_groups=False
    )


def evaluate(y_true, preds, p_std=None, weeks=None):
    res = {
        "auroc": auroc(y_true, preds),
        "accuracy": accuracy(y_true, preds),
        "brier": brier(y_true, preds),
        "logloss": logloss(y_true, preds),
    }
    if weeks is not None:
        _, res["per_week_auroc_mean"] = per_week_auroc(y_true, preds, weeks)
    if p_std is not None:
        res["width_stratified_brier"] = {
            str(k): float(v)
            for k, v in width_stratified_brier(y_true, preds, p_std).items()
        }
    return res
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data_science_utilities/models/bayes_logistic/test_evaluate.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add data_science_utilities/models/bayes_logistic/evaluate.py \
        tests/data_science_utilities/models/bayes_logistic/test_evaluate.py
git commit -m "feat(bayes): holdout metrics matching bart.ipynb (auroc/brier/calibration/width-brier)"
```

---

## Task 5: PyMC model — build / sample / predict / diagnostics

**Files:**
- Create: `data_science_utilities/models/bayes_logistic/model.py`
- Create: `tests/data_science_utilities/models/bayes_logistic/test_model.py`

**Interfaces:**
- Produces:
  - `model.build_model(X: np.ndarray, y: np.ndarray, prior="normal", *, slab_scale=2.0, slab_df=4.0) -> pm.Model` (named vars `alpha`, `beta`, `p`, `y`).
  - `model.sample(pm_model, *, draws=2000, tune=2000, chains=4, seed=32, target_accept=0.9) -> az.InferenceData`
  - `model.predict(pm_model, idata, X_new, seed=32) -> (mean_p: np.ndarray, std_p: np.ndarray)`
  - `model.diagnostics(idata, var_names=("alpha", "beta")) -> dict` with keys `max_rhat, min_ess_bulk, n_divergences, passed`.
- Note: this pattern is verified working on PyMC 5.23 (build + `set_data` predict + diagnostics).

- [ ] **Step 1: Write the failing test** (tiny sampler for speed)

Create `tests/data_science_utilities/models/bayes_logistic/test_model.py`:
```python
import unittest

import numpy as np

from data_science_utilities.models.bayes_logistic import model

SAMPLE_KW = dict(draws=60, tune=60, chains=2, seed=32)


def _toy(n=150, d=4, seed=32):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    beta = np.array([1.5, -1.0, 0.5, 0.0])
    p = 1.0 / (1.0 + np.exp(-(0.2 + X @ beta)))
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y


class TestBuildModel(unittest.TestCase):
    def test_normal_model_has_named_vars(self):
        X, y = _toy()
        m = model.build_model(X, y, prior="normal")
        names = {v.name for v in m.free_RVs} | {d.name for d in m.deterministics}
        self.assertIn("alpha", names)
        self.assertIn("beta", names)
        self.assertIn("p", names)

    def test_horseshoe_model_builds(self):
        X, y = _toy()
        m = model.build_model(X, y, prior="horseshoe")
        self.assertIsNotNone(m)

    def test_unknown_prior_raises(self):
        X, y = _toy()
        with self.assertRaises(ValueError):
            model.build_model(X, y, prior="banana")


class TestSamplePredictDiagnostics(unittest.TestCase):
    def test_predict_shapes_and_range(self):
        X, y = _toy()
        m = model.build_model(X, y, prior="normal")
        idata = model.sample(m, **SAMPLE_KW)
        mean_p, std_p = model.predict(m, idata, X[:20])
        self.assertEqual(mean_p.shape, (20,))
        self.assertEqual(std_p.shape, (20,))
        self.assertTrue(((mean_p >= 0) & (mean_p <= 1)).all())
        self.assertTrue((std_p >= 0).all())

    def test_predict_is_deterministic_with_seed(self):
        X, y = _toy()
        m = model.build_model(X, y, prior="normal")
        idata = model.sample(m, **SAMPLE_KW)
        a, _ = model.predict(m, idata, X[:10], seed=32)
        b, _ = model.predict(m, idata, X[:10], seed=32)
        np.testing.assert_allclose(a, b)

    def test_diagnostics_keys(self):
        X, y = _toy()
        m = model.build_model(X, y, prior="normal")
        idata = model.sample(m, **SAMPLE_KW)
        diag = model.diagnostics(idata)
        for k in ["max_rhat", "min_ess_bulk", "n_divergences", "passed"]:
            self.assertIn(k, diag)
        self.assertIsInstance(diag["n_divergences"], int)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data_science_utilities/models/bayes_logistic/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: ... 'model'`

- [ ] **Step 3: Implement the model**

Create `data_science_utilities/models/bayes_logistic/model.py`:
```python
"""PyMC Bayesian logistic regression: two prior variants, mirroring the BART
predict pattern (pm.Data + set_data + posterior-predictive of a Deterministic p).
"""
import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

RANDOM_SEED = 32


def build_model(X, y, prior="normal", *, slab_scale=2.0, slab_df=4.0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n_features = X.shape[1]
    with pm.Model() as m:
        X_data = pm.Data("X", X)
        alpha = pm.Normal("alpha", 0.0, 1.5)
        if prior == "normal":
            beta = pm.Normal("beta", 0.0, 1.0, shape=n_features)
        elif prior == "horseshoe":
            # Regularized (Finnish) horseshoe, non-centered (Piironen & Vehtari 2017).
            tau = pm.HalfCauchy("tau", beta=1.0)
            lam = pm.HalfCauchy("lam", beta=1.0, shape=n_features)
            c2 = pm.InverseGamma("c2", alpha=slab_df / 2, beta=slab_df / 2 * slab_scale**2)
            lam_tilde = pt.sqrt(c2 * lam**2 / (c2 + tau**2 * lam**2))
            z = pm.Normal("z", 0.0, 1.0, shape=n_features)
            beta = pm.Deterministic("beta", z * lam_tilde * tau)
        else:
            raise ValueError(f"unknown prior {prior!r}; use 'normal' or 'horseshoe'")
        logit_p = alpha + pm.math.dot(X_data, beta)
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        pm.Bernoulli("y", p=p, observed=y, shape=X_data.shape[0])
    return m


def sample(pm_model, *, draws=2000, tune=2000, chains=4, seed=RANDOM_SEED, target_accept=0.9):
    with pm_model:
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=chains,
            random_seed=seed, target_accept=target_accept, progressbar=False,
        )
    return idata


def predict(pm_model, idata, X_new, seed=RANDOM_SEED):
    X_new = np.asarray(X_new, dtype=float)
    with pm_model:
        pm.set_data({"X": X_new})
        ppc = pm.sample_posterior_predictive(
            idata, var_names=["p"], random_seed=seed, progressbar=False
        )
    post_p = ppc.posterior_predictive["p"]
    return (
        post_p.mean(dim=["chain", "draw"]).to_numpy(),
        post_p.std(dim=["chain", "draw"]).to_numpy(),
    )


def diagnostics(idata, var_names=("alpha", "beta")):
    summ = az.summary(idata, var_names=list(var_names))
    n_div = int(idata.sample_stats["diverging"].sum())
    max_rhat = float(summ["r_hat"].max())
    min_ess_bulk = float(summ["ess_bulk"].min())
    return {
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "n_divergences": n_div,
        "passed": bool(max_rhat < 1.01 and n_div == 0 and min_ess_bulk > 400),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data_science_utilities/models/bayes_logistic/test_model.py -v`
Expected: 6 passed (takes ~30–60s for the tiny sampler runs).

- [ ] **Step 5: Commit**

```bash
git add data_science_utilities/models/bayes_logistic/model.py \
        tests/data_science_utilities/models/bayes_logistic/test_model.py
git commit -m "feat(bayes): PyMC logistic model (normal + regularized horseshoe), sample/predict/diagnostics"
```

---

## Task 6: Posterior export (NetCDF trace + JSON coef summary)

**Files:**
- Create: `data_science_utilities/models/bayes_logistic/posterior.py`
- Create: `tests/data_science_utilities/models/bayes_logistic/test_posterior.py`

**Interfaces:**
- Consumes: an `idata` with `posterior["alpha"]`/`posterior["beta"]` (Task 5); a fitted `TrainFitPreprocessor` (Task 3).
- Produces:
  - `posterior.coefficient_summary(idata, feature_names_out, preprocessor, prior) -> dict`
  - `posterior.save_summary(summary, path) -> None`
  - `posterior.save_trace(idata, path) -> None`
  - `posterior.load_trace(path) -> az.InferenceData`
- Summary schema: `{"prior": str, "scale": "standardized", "n_features": int, "intercept": {"mean": float, "sd": float}, "coefficients": [{"feature": str, "mean": float, "sd": float, "standardizer_mean": float, "standardizer_std": float}, ...]}`.

- [ ] **Step 1: Write the failing test**

Create `tests/data_science_utilities/models/bayes_logistic/test_posterior.py`:
```python
import json
import os
import tempfile
import unittest

import numpy as np

from data_science_utilities.models.bayes_logistic import data_prep, model, posterior


def _fitted():
    import pandas as pd
    rng = np.random.default_rng(32)
    df = pd.DataFrame({"f0": rng.normal(size=120), "f1": rng.normal(size=120)})
    pre = data_prep.TrainFitPreprocessor(["f0", "f1"]).fit(df)
    X = pre.transform(df)
    y = (rng.uniform(size=120) < 0.5).astype(int)
    m = model.build_model(X, y, prior="normal")
    idata = model.sample(m, draws=60, tune=60, chains=2, seed=32)
    return idata, pre


class TestCoefficientSummary(unittest.TestCase):
    def test_summary_structure(self):
        idata, pre = _fitted()
        summary = posterior.coefficient_summary(
            idata, pre.feature_names_out_, pre, prior="normal"
        )
        self.assertEqual(summary["prior"], "normal")
        self.assertEqual(summary["n_features"], len(pre.feature_names_out_))
        self.assertEqual(len(summary["coefficients"]), len(pre.feature_names_out_))
        self.assertIn("mean", summary["intercept"])
        c0 = summary["coefficients"][0]
        for k in ["feature", "mean", "sd", "standardizer_mean", "standardizer_std"]:
            self.assertIn(k, c0)
        self.assertEqual(c0["feature"], pre.feature_names_out_[0])

    def test_summary_json_roundtrips(self):
        idata, pre = _fitted()
        summary = posterior.coefficient_summary(idata, pre.feature_names_out_, pre, prior="normal")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "coef.json")
            posterior.save_summary(summary, path)
            loaded = json.load(open(path))
        self.assertEqual(loaded["n_features"], summary["n_features"])


class TestTraceRoundtrip(unittest.TestCase):
    def test_netcdf_roundtrip(self):
        idata, _ = _fitted()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.nc")
            posterior.save_trace(idata, path)
            self.assertTrue(os.path.exists(path))
            reloaded = posterior.load_trace(path)
            self.assertIn("beta", reloaded.posterior)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data_science_utilities/models/bayes_logistic/test_posterior.py -v`
Expected: FAIL — `ModuleNotFoundError: ... 'posterior'`

- [ ] **Step 3: Implement posterior export**

Create `data_science_utilities/models/bayes_logistic/posterior.py`:
```python
"""Expose the fitted posterior for diagnostics and for seeding a later model's prior.

The JSON summary carries both the posterior moments (on the standardized scale) and
the standardizer moments per feature, so a downstream model can rebuild a prior on
the standardized *or* raw scale.
"""
import json

import arviz as az
import numpy as np


def coefficient_summary(idata, feature_names_out, preprocessor, prior):
    post = idata.posterior
    beta_mean = post["beta"].mean(dim=["chain", "draw"]).to_numpy()
    beta_sd = post["beta"].std(dim=["chain", "draw"]).to_numpy()
    alpha_mean = float(post["alpha"].mean(dim=["chain", "draw"]).to_numpy())
    alpha_sd = float(post["alpha"].std(dim=["chain", "draw"]).to_numpy())
    coefficients = []
    for i, name in enumerate(feature_names_out):
        coefficients.append({
            "feature": name,
            "mean": float(beta_mean[i]),
            "sd": float(beta_sd[i]),
            "standardizer_mean": float(preprocessor.means_[i]),
            "standardizer_std": float(preprocessor.stds_[i]),
        })
    return {
        "prior": prior,
        "scale": "standardized",
        "n_features": len(feature_names_out),
        "intercept": {"mean": alpha_mean, "sd": alpha_sd},
        "coefficients": coefficients,
    }


def save_summary(summary, path):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def save_trace(idata, path):
    idata.to_netcdf(path)


def load_trace(path):
    return az.from_netcdf(path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data_science_utilities/models/bayes_logistic/test_posterior.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add data_science_utilities/models/bayes_logistic/posterior.py \
        tests/data_science_utilities/models/bayes_logistic/test_posterior.py
git commit -m "feat(bayes): posterior export (NetCDF trace + coef-summary JSON with standardizer moments)"
```

---

## Task 7: Orchestrator script

**Files:**
- Create: `scripts/experiments/bayes_logistic.py`
- Create: `tests/data_science_utilities/models/bayes_logistic/test_orchestrator.py`

**Interfaces:**
- Consumes: all four modules.
- Produces: `bayes_logistic.run(prior, split, results_dir, *, draws, tune, chains, seed, target_accept) -> dict` (one variant's results row) and `bayes_logistic.main(argv=None) -> dict` (full run, writes artifacts, returns the results dict). CLI flags: `--draws --tune --chains --seed --outdir --smoke --priors`.
- `--smoke` shrinks sampling (`draws=60, tune=60, chains=2`) and subsamples train to 800 rows for a fast end-to-end check.

- [ ] **Step 1: Write the failing test** (smoke end-to-end)

Create `tests/data_science_utilities/models/bayes_logistic/test_orchestrator.py`:
```python
import json
import os
import tempfile
import unittest

import scripts.experiments.bayes_logistic as orch


class TestOrchestratorSmoke(unittest.TestCase):
    def test_smoke_run_writes_results(self):
        with tempfile.TemporaryDirectory() as d:
            res = orch.main([
                "--smoke", "--priors", "normal", "--outdir", d,
            ])
            self.assertIn("normal", res["variants"])
            row = res["variants"]["normal"]
            for k in ["auroc", "brier", "diagnostics"]:
                self.assertIn(k, row)
            self.assertTrue(os.path.exists(os.path.join(d, "results.json")))
            saved = json.load(open(os.path.join(d, "results.json")))
            self.assertIn("variants", saved)
            self.assertTrue(os.path.exists(os.path.join(d, "posterior_normal.nc")))
            self.assertTrue(os.path.exists(os.path.join(d, "coef_summary_normal.json")))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data_science_utilities/models/bayes_logistic/test_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.experiments.bayes_logistic'`

- [ ] **Step 3: Implement the orchestrator**

Create `scripts/experiments/bayes_logistic.py`:
```python
"""Train + evaluate the Bayesian logistic regression against the BART champion.

Run from repo root:
    conda run --no-capture-output -n nfl-predictions \
        python scripts/experiments/bayes_logistic.py
Fast check:  ... python scripts/experiments/bayes_logistic.py --smoke --priors normal
"""
import argparse
import json
import os
import sys
import time

import numpy as np

# Repo root on sys.path so first-party namespace packages import when run by path.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_science_utilities.models.bayes_logistic import (  # noqa: E402
    data_prep, evaluate, model, posterior,
)

# Reference rows for the side-by-side table (README run log).
REFERENCE = {
    "BART Run-6 (same 54 feats)": {"auroc": 0.705, "brier": 0.2194},
    "BART Run-10 (champion)": {"auroc": 0.708, "brier": 0.2185},
    "XGBoost Run-11 (best)": {"auroc": 0.707, "brier": 0.2206},
}
DEFAULT_OUTDIR = os.path.join(REPO_ROOT, "data", "predict_games", "bayes_logistic")
TARGET_ACCEPT = {"normal": 0.9, "horseshoe": 0.95}


def run(prior, split, feature_names, results_dir, *, draws, tune, chains, seed):
    pre = data_prep.TrainFitPreprocessor(feature_names).fit(split.train)
    X_train = pre.transform(split.train)
    y_train = data_prep.get_target(split.train)
    X_holdout = pre.transform(split.holdout)
    y_holdout = data_prep.get_target(split.holdout)

    t0 = time.time()
    pm_model = model.build_model(X_train, y_train, prior=prior)
    idata = model.sample(
        pm_model, draws=draws, tune=tune, chains=chains, seed=seed,
        target_accept=TARGET_ACCEPT[prior],
    )
    mean_p, std_p = model.predict(pm_model, idata, X_holdout, seed=seed)
    elapsed = time.time() - t0

    metrics = evaluate.evaluate(
        y_holdout, mean_p, p_std=std_p, weeks=split.holdout["week"].to_numpy()
    )
    diag = model.diagnostics(idata)

    posterior.save_trace(idata, os.path.join(results_dir, f"posterior_{prior}.nc"))
    summary = posterior.coefficient_summary(idata, pre.feature_names_out_, pre, prior=prior)
    posterior.save_summary(summary, os.path.join(results_dir, f"coef_summary_{prior}.json"))

    row = dict(metrics)
    row["diagnostics"] = diag
    row["elapsed_sec"] = round(elapsed, 1)
    print(f"[{prior}] AUROC {row['auroc']:.4f} | Brier {row['brier']:.4f} | "
          f"acc {row['accuracy']:.4f} | rhat {diag['max_rhat']:.3f} | "
          f"div {diag['n_divergences']} | ess {diag['min_ess_bulk']:.0f} | "
          f"{row['elapsed_sec']}s | passed={diag['passed']}")
    return row


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=data_prep.RANDOM_SEED)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--priors", nargs="+", default=["normal", "horseshoe"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    feature_names = data_prep.load_feature_list()
    split = data_prep.load_and_split(seed=args.seed)

    draws, tune, chains = args.draws, args.tune, args.chains
    if args.smoke:
        draws, tune, chains = 60, 60, 2
        split = data_prep.Split(
            train=split.train.head(800), valid=split.valid, holdout=split.holdout.head(200),
        )

    variants = {}
    for prior in args.priors:
        variants[prior] = run(
            prior, split, feature_names, args.outdir,
            draws=draws, tune=tune, chains=chains, seed=args.seed,
        )

    results = {"variants": variants, "reference": REFERENCE,
               "n_features": len(feature_names), "smoke": args.smoke}
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Side-by-side table + a ready-to-paste README run-log line for the best variant.
    print("\n=== Side-by-side (holdout 2024+2025) ===")
    for name, m in {**{f"Bayes-{k}": v for k, v in variants.items()}, **REFERENCE}.items():
        print(f"  {name:32s} AUROC {m['auroc']:.4f}  Brier {m['brier']:.4f}")
    best = max(variants, key=lambda k: variants[k]["auroc"])
    b = variants[best]
    print(f"\nREADME Run-27 line (best = {best}):")
    print(f"| 27 | 2026-06-20 | **Bayesian logistic regression** ({best} prior), "
          f"frozen 54-feature Run-6 set | {len(feature_names)} | "
          f"{b['auroc']:.3f} | {b['accuracy']:.3f} | {b['brier']:.4f} |")
    return results


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run --no-capture-output -n nfl-predictions python -m pytest tests/data_science_utilities/models/bayes_logistic/test_orchestrator.py -v`
Expected: 1 passed (the smoke run takes ~30–90s).

- [ ] **Step 5: Run the whole suite + commit**

```bash
conda run --no-capture-output -n nfl-predictions python -m pytest tests/data_science_utilities/models/bayes_logistic/ -v
git add scripts/experiments/bayes_logistic.py \
        tests/data_science_utilities/models/bayes_logistic/test_orchestrator.py
git commit -m "feat(bayes): orchestrator script (both priors, artifacts, README line)"
```

---

## Task 8: Real run + README Run-27 entry + honest verdict

**Files:**
- Create: `data/predict_games/bayes_logistic/` artifacts (generated)
- Modify: `README.md` (run-log table + commentary)

**Interfaces:** consumes the orchestrator from Task 7.

- [ ] **Step 1: Execute the full run (both priors, full sampler)**

Run (from repo root; expect a few minutes):
```bash
conda run --no-capture-output -n nfl-predictions \
  python scripts/experiments/bayes_logistic.py 2>&1 | tee /tmp/bayes_run.log
```
Expected: both variants print AUROC/Brier/diagnostics; a `=== Side-by-side ===` block; a `README Run-27 line`. Artifacts appear under `data/predict_games/bayes_logistic/` (`posterior_normal.nc`, `posterior_horseshoe.nc`, `coef_summary_*.json`, `results.json`).

- [ ] **Step 2: Verify sampler diagnostics passed**

Check `/tmp/bayes_run.log`: for each variant confirm `passed=True` (max_rhat<1.01, 0 divergences, min_ess_bulk>400). If horseshoe shows divergences, re-run it with a higher target_accept:
```bash
conda run --no-capture-output -n nfl-predictions \
  python -c "import scripts.experiments.bayes_logistic as o; o.main(['--priors','horseshoe'])"
```
(If divergences persist after target_accept=0.95, record it honestly in the verdict rather than hiding it — see Step 4.)

- [ ] **Step 3: Add the README Run-27 row**

In `README.md`, in the run-log table (after the Run-26 row, ~line 192), paste the `README Run-27 line` printed by the script (the best variant). Keep the exact column alignment of neighboring rows.

- [ ] **Step 4: Add README commentary + honest verdict**

In `README.md`, in the per-run commentary section (after the Run-26 note), add a short block stating:
- the two variants' holdout AUROC/Brier and which one was reported;
- the sampler diagnostics (max R-hat, divergences, min ESS);
- the calibration result (reliability curve direction) and whether the posterior-width-stratified Brier is monotone;
- **the verdict:** whether Bayesian-logistic is competitive with BART 0.705 — and if it underperforms, attribute it to the linear-additivity assumption vs. the trees' interaction capture (do not overstate). Note the posterior artifacts now exist to seed a later Madden transfer-learning model.

- [ ] **Step 5: Commit**

```bash
git add README.md data/predict_games/bayes_logistic/results.json \
        data/predict_games/bayes_logistic/coef_summary_normal.json \
        data/predict_games/bayes_logistic/coef_summary_horseshoe.json
# Note: large NetCDF traces may be git-ignored; check `git status` and decide
# (see project .gitignore conventions) before adding *.nc.
git commit -m "experiment(bayes): Run 27 — Bayesian logistic regression vs BART champion"
```

---

## Self-Review (against the spec)

**Spec coverage:**
- §1/§3.1 protocol → Tasks 2 (split/shuffle/seed) + 7 (per-row holdout eval). ✓
- §2 scope (flat, two priors, deferred hierarchy/CV) → Task 5 (both priors), no CV task. ✓
- §3.2 metrics (auroc/acc/brier/logloss/per-week/reliability/width-brier) → Task 4. ✓
- §3.3 frozen 54-feature set + provenance + test → Task 1. ✓
- §3.4 library availability → no new deps; uses installed pymc/arviz. ✓
- §4.1 data_prep → Tasks 1–3. §4.2 model → Task 5. §4.3 evaluate → Task 4. §4.4 posterior → Task 6. §4.5 orchestrator → Task 7. ✓
- §5 model spec (α~N(0,1.5), β~N(0,1), horseshoe, NUTS draws/tune/chains/target_accept) → Task 5 + Task 7 (`TARGET_ACCEPT`). ✓
- §6 caveats → Task 8 verdict commentary. ✓
- §7 tests (split boundaries, seed, leakage standardizer, impute+flag, zero-variance, metric correctness, posterior JSON, NetCDF roundtrip, MCMC smoke) → Tasks 2–7. ✓
- §8 deliverables (modules + script + frozen CSV + tests + artifacts + README) → Tasks 1–8. ✓

**Placeholder scan:** every code/test/command step contains concrete content. No TBD/TODO. ✓

**Type consistency:** `Split` fields (`train/valid/holdout`), `TrainFitPreprocessor` API (`.fit/.transform/.feature_names_out_/.means_/.stds_`), `model.{build_model,sample,predict,diagnostics}`, `posterior.{coefficient_summary,save_summary,save_trace,load_trace}`, and `bayes_logistic.{run,main}` are referenced consistently across tasks. `coefficient_summary(idata, feature_names_out, preprocessor, prior)` signature matches its call in Task 7. ✓

**One open execution note (not a plan gap):** whether to commit the `.nc` traces depends on repo `.gitignore`/size conventions — flagged in Task 8 Step 5 for the executor to decide; the coef-summary JSONs are the lightweight transfer-learning surface and are always committed.
