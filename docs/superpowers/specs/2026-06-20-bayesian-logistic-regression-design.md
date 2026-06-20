# Bayesian Logistic Regression vs. the BART Champion — Design

- **Date:** 2026-06-20
- **Branch context:** builds on `madden-ratings-features`
- **Status:** Design — pending user review

## 1. Goal

Build a **classic parametric Bayesian model** — Bayesian logistic regression in PyMC —
that predicts `target_win`, trained and evaluated on the **same dataset and protocol as
the current BART champion** (Run 6, hold-out ROC-AUC **0.705**), so the results are
directly comparable. This is the first parametric-Bayes baseline in the repo.

Two secondary objectives shape the design:

1. **Transfer-learning foundation.** The fitted posterior must be exposed in a reusable
   form so a later model can seed its prior from this posterior and update on weekly
   Madden data. Concretely: a saved trace **and** a compact per-coefficient
   posterior summary, both leakage-aware about the standardization used.
2. **Bayesian extras BART can't cheaply give.** A calibration/reliability check and
   posterior-predictive uncertainty (per-row posterior std), plus sampler diagnostics.

**Honest-outcome clause:** a non-competitive Bayesian model is a valid, informative
result and will be reported plainly (it would say the signal in these 54 features is
not linearly separable in a way a GLM captures, which is itself useful).

## 2. Scope (YAGNI)

- **In scope:** flat (non-hierarchical) Bayesian logistic regression, two prior
  variants (weakly-informative Normal and regularized horseshoe), the fixed-season-split
  eval that matches BART, the Bayesian diagnostics, posterior export, a reproducible
  script, unit tests, and a README Run-27 entry.
- **Deferred (not built now):** hierarchical / partial-pooling extension (e.g. by
  season or team); the actual Madden transfer-learning update model; rolling-origin CV
  as a headline metric.
- **Optional stretch (flag, build only if requested):** rolling-origin CV reported as a
  *robustness check only* — never the headline, which must be the fixed split for
  comparability.

## 3. Background / verified facts

All of the following were verified directly against source, not assumed.

### 3.1 The champion eval protocol — a fixed season split, **not** k-fold

Source of truth: `notebooks/model_training/predict_games/schedule_and_weekly/cross_validation/bart.ipynb`.
(The folder is named `cross_validation/`, but the **headline metric uses a single fixed
season split**; StratifiedKFold appears only in the historical RFE *feature search*,
`rfe.ipynb`, not in the champion eval.)

```python
RANDOM_SEED = 32
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
np.random.seed(RANDOM_SEED)
train_df   = df[df.season < 2022]                         # ~7.8k rows
valid_df   = df[(df.season >= 2022) & (df.season < 2024)] # 2022-2023 check
holdout_df = df[df.season >= 2024]                        # 2024+2025 = HEADLINE (1,088 rows / 544 games)
```

- **Target:** `target_win` (int64, perfectly balanced 50/50).
- **`season` is deliberately excluded as a feature** (era drift; the hold-out is a novel
  era). The one-week leakage shift is already baked into the parquet — the new model
  must **not** re-apply any shift.
- **Grain:** team-game, two perspective rows per game (`target_*` / `opp_*`). BART
  evaluates **per-row** on all 1,088 holdout rows; we match that exactly.

### 3.2 Metrics (same calls as `bart.ipynb`)

- `roc_auc_score(y_holdout, preds)`
- accuracy: `((preds > 0.5).astype(int) == y_holdout).mean()`
- `brier_score_loss(holdout['target_win'], holdout['preds'])`
- `log_loss(...)`
- per-week AUROC: `groupby('week').apply(roc_auc_score)` then `.mean()`
- 10-bin **quantile** reliability curve: `calibration_curve(..., n_bins=10, strategy='quantile')`
- **posterior-width-stratified Brier:** `pd.qcut(preds_std, 4)` → Brier per quartile
  (the Bayesian uncertainty-trustworthiness diagnostic; the GLM analogue uses the
  per-row posterior std of P(win)).

### 3.3 The 54-feature Run-6 set — provenance, recovery, NaN profile

- README Run 5 (XGBoost) = "rank-only, **`dakota` dropped**, 2024+2025 hold-out, 54
  feats, AUROC 0.697". README Run 6 = "**BART** on the **same 54 features / split as run
  5**, AUROC **0.705**, Brier 0.2194". This is the comparability anchor.
- The current `rfe_features_kfolds.csv` **no longer contains a 54-count row** (Phase 2/3
  feature additions re-ran RFE; counts now jump 57 → 51). The exact Run-6 set is
  recovered from git: **`b1410bd:data/predict_games/model_features_in/rfe_features_kfolds.csv` → `.loc[54]`**.
- **Verified:** all 54 columns still exist in the current parquet; **zero of the 54 have
  any NaN**; none are `madden_*` or `dakota`. The set is ranks (1–32), rank-changes
  (±31), and a handful of complete raw stats (`off_target_pacr`, `def_opp_racr`,
  `*_passing_epa`, `*_yards_after_catch`, `def_opp_attempts`, `opp_game_count`, …).
- **Action:** freeze this list into a committed file
  `data/predict_games/model_features_in/bayes_logistic_features.csv` (one feature per
  row, plus a header comment citing `b1410bd .loc[54]` provenance) so the script is
  reproducible without git archaeology. A unit test asserts the frozen list matches the
  git-recovered 54 exactly.

**Consequence for the NaN strategy:** because the set is NaN-free, the imputation path is
**inert here**. It is still implemented (leakage-safe) so a future feature swap is
covered, but on this set it produces no imputed values and no missingness flags. This is
a clean complete-case comparison — no imputation distortion of linear coefficients.

### 3.4 BART champion numbers + library availability

- Comparison row: **BART Run 6 — AUROC 0.705 / Brier 0.2194 / 54 feats** (the exact same
  feature set + split). Context: BART Run 10 = 0.708 (AUROC champion, 57 feats); best
  XGBoost Run 11 = 0.707 / 0.2206.
- BART's posterior is **not persisted** — there is nothing to align to; we expose our own.
- **`pymc` + `arviz` are already installed** (transitively via `pymc-bart==0.9.2`,
  `environment.yml:23`). **No new dependency.** `bambi` / `numpyro` are absent and not
  needed. Env: `nfl-predictions` (Python 3.11, scikit-learn 1.4.2, pandas 2.2, numpy 1.26).

## 4. Architecture

Reusable modeling code lives under `data_science_utilities/models/bayes_logistic/`
(mirroring `models/bart/`, `models/xgb/`); a thin orchestrator script lives under
`scripts/experiments/`. Each unit is small and independently testable.

### 4.1 `data_science_utilities/models/bayes_logistic/data_prep.py`

Pure, deterministic, leakage-safe data preparation. No PyMC import.

- `load_feature_list(path) -> list[str]` — read the frozen 54-feature CSV.
- `load_and_split(parquet_path, seed=32) -> Split` — load, shuffle (`sample(frac=1,
  random_state=seed)`, `reset_index`), `np.random.seed(seed)`, return a `Split`
  dataclass with `train/valid/holdout` DataFrames (boundaries from §3.1).
- `class TrainFitPreprocessor` — fit on **train only**, transform any fold:
  - median impute (train medians) + `*_was_missing` indicator for any column with NaN in
    train (inert on this set);
  - z-score standardize using **train** mean/sd; **zero-variance guard** (std==0 →
    leave column centered-only / drop the dead missingness flag);
  - `fit(train_X)`, `transform(X) -> np.ndarray`, exposes `.feature_names_out_`,
    `.means_`, `.stds_` (the last two are needed by the posterior export for raw-scale
    reconstruction).
- `prepare(...) -> (X_train, y_train, X_valid, y_valid, X_holdout, y_holdout, preprocessor)`
  convenience that wires the above. Asserts no NaN reaches the model (mirrors the BART
  `assert not np.isnan(...)` guard).

### 4.2 `data_science_utilities/models/bayes_logistic/model.py`

PyMC model construction + sampling + prediction. One function per prior variant via a
`prior: Literal["normal", "horseshoe"]` argument.

- `build_model(X, y, prior="normal", **hyperparams) -> pm.Model`:
  - `α ~ Normal(0, 1.5)`; `logit_p = α + X@β`; `y ~ Bernoulli(p=sigmoid(logit_p))`;
    `pm.Data('X', X)` so new X can be swapped for prediction (mirrors BART `set_data`).
  - `prior="normal"`: `β ~ Normal(0, 1)` (standardized features).
  - `prior="horseshoe"`: regularized horseshoe (Piironen–Vehtari) — global scale `τ`,
    local scales `λ_j`, slab `c`; non-centered parameterization for sampling stability.
- `sample(model, draws=2000, tune=2000, chains=4, seed=32, target_accept=0.9) -> idata`
  (horseshoe uses `target_accept=0.95`).
- `predict(model, idata, X_new) -> (mean_p, std_p)` — posterior-predictive P(win) mean
  and per-row std (mirrors `bart_predict`).
- `diagnostics(idata) -> dict` — max R-hat, divergence count, min ESS-bulk/tail;
  acceptance gate documented (R-hat<1.01, 0 divergences, ESS-bulk>400). The script warns
  loudly if the gate fails rather than silently reporting numbers.

### 4.3 `data_science_utilities/models/bayes_logistic/evaluate.py`

Pure metric functions on `(y_true, preds[, preds_std, weeks])`, each a thin, tested
wrapper over the exact sklearn calls in §3.2, returning a results dict. Includes
`reliability_curve(...)` and `width_stratified_brier(...)`. Reuses sklearn directly
(the existing `utils/calibration/factors.py` is for probability *re-calibration*, not
reliability diagrams, so it is not used here).

### 4.4 `data_science_utilities/models/bayes_logistic/posterior.py`

The transfer-learning surface.

- `save_trace(idata, path)` — `idata.to_netcdf(path)` (full `.nc`).
- `coefficient_summary(idata, feature_names, preprocessor) -> dict` — per-coefficient
  posterior **mean** and **sd** (intercept + each β), **plus** the preprocessor's
  per-feature `mean`/`std`, so a downstream model can build a prior on the standardized
  *or* raw scale. Serialized to JSON.
- `save_summary(summary, path)` — write JSON. Artifacts land under
  `data/predict_games/bayes_logistic/` (e.g. `posterior_normal.nc`,
  `coef_summary_normal.json`, and the horseshoe equivalents).

### 4.5 `scripts/experiments/bayes_logistic.py`

Thin orchestrator following `scripts/experiments/madden_ablation.py` conventions
(`REPO_ROOT` + `sys.path`, config via env vars / argparse with sane defaults, docstring,
timing, progress logging). It:

1. prepares data; 2. for each prior variant: builds, samples, predicts, evaluates, saves
trace + coef summary; 3. prints sampler diagnostics; 4. prints a side-by-side metrics
table (Variant A / Variant B / BART Run-6 0.705 / best XGBoost 0.707); 5. writes a JSON
results file under `data/predict_games/bayes_logistic/`; 6. prints a ready-to-paste
README Run-27 line.

Run via `conda run -n nfl-predictions python scripts/experiments/bayes_logistic.py`.

## 5. Model specification (statistics)

- Likelihood: `y_i ~ Bernoulli(sigmoid(α + xᵢ·β))`, features standardized.
- **Variant A (weakly-informative):** `α ~ Normal(0, 1.5)`, `β_j ~ Normal(0, 1)`. On
  standardized predictors, `Normal(0,1)` keeps per-feature log-odds contributions
  weakly regularized; `Normal(0,1.5)` on the intercept is near-flat for a balanced target.
- **Variant B (regularized horseshoe):** aggressive shrinkage of irrelevant coefficients
  toward zero with heavy tails for genuinely strong features — appropriate given the 54
  predictors are correlated ranks. Reported alongside A to show whether sparsity helps.
- **Sampler:** NUTS, `chains=4, tune=2000, draws=2000, seed=32` (cheap for a 54-dim GLM;
  more than BART's 1000 for tighter ESS). Diagnostics reported every run.

## 6. Known caveats (honest, shared with BART)

- **Two non-independent rows per game** (complementary perspectives) violate strict
  Bernoulli independence. BART shares this; we keep per-row eval for comparability and
  document it rather than "fixing" it (a fix would break the apples-to-apples).
- **No CV in the headline.** The fixed split is a single realization; we report it
  because BART does. Rolling-origin CV is available as an optional robustness check only.
- **Linear-additivity assumption.** If the Bayesian GLM underperforms BART, the most
  likely cause is non-linear/interaction structure the trees capture and the GLM cannot;
  this will be stated explicitly in the verdict.

## 7. Testing (TDD, `unittest`)

Under `tests/data_science_utilities/models/bayes_logistic/`, mirroring repo conventions
(`test_<module>.py`, `Test<Thing>`, `test_<scenario>`):

- **data_prep:** season-split boundaries (a row with `season==2021/2022/2023/2024` lands
  in the right fold); shuffle/seed reproducibility; **standardizer fit-on-train-only**
  (transform of valid/holdout uses train means/stds — explicit leakage test); impute+flag
  behavior on **synthetic** NaN (since the real set has none); zero-variance guard; frozen
  feature list equals the git-recovered 54; no-NaN-reaches-model assertion.
- **evaluate:** each metric matches a direct sklearn computation on a toy array; per-week
  AUROC averaging; `width_stratified_brier` returns 4 quartiles; reliability-curve shape.
- **posterior:** `coefficient_summary` JSON has intercept + 54 β entries each with
  mean/sd plus per-feature standardizer mean/std; NetCDF round-trips (save/load) on a
  tiny idata.
- **model (smoke):** build + sample a tiny synthetic problem (2 chains, ~50 draws) →
  `predict` returns probabilities in [0,1] of correct shape; deterministic under fixed
  seed. Kept fast; the full-data fit is exercised by the script, not the unit suite.

## 8. Deliverables

1. `data_science_utilities/models/bayes_logistic/{data_prep,model,evaluate,posterior}.py`
   + the orchestrator `scripts/experiments/bayes_logistic.py`.
2. Frozen feature file `data/predict_games/model_features_in/bayes_logistic_features.csv`.
3. Unit tests (§7), all passing under `python -m unittest discover -s tests`.
4. Posterior artifacts under `data/predict_games/bayes_logistic/` (`.nc` + coef-summary
   JSON for both variants) + a JSON results file.
5. **README Run 27** entry in the exact run-log table format, with a short commentary
   note on whether Bayesian-logistic is competitive with BART 0.705 (honest verdict).

## 9. Open questions

None blocking. Resolved during brainstorming: feature set = frozen 54-feature Run-6 set;
priors = Normal(0,1) **and** regularized horseshoe (both reported); NaN = median-impute +
missingness flag (inert on this set); posterior export = NetCDF trace + JSON coef summary;
rolling-origin CV deferred/optional.
