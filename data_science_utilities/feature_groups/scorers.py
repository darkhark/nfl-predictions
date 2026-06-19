"""Model-specific fold scorers for the group-ablation harness.

A *fold scorer* is ``fit_score_fn(feature_subset: list[str]) -> float``: it trains one
model on a fixed train block and returns the validation metric on a fixed test block.
Feeding a fold scorer to ``GroupAblation`` scores every group subset on that fold;
running one ablation per rolling-origin season fold (``season_rolling_origin_folds``)
and aggregating with ``cross_validated_group_ablation`` yields per-group effects with
honest error bars.

Design choice: the sweep uses a FIXED, regularised model per subset -- NO per-subset
RFE or hyperparameter search. That deliberately isolates the *group* effect from the
*selection* effect; re-running RFE inside each subset would re-import the crowding-out
confound this harness exists to measure cleanly.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Mirrors ClassifierCrossValidationRecursiveFeatureSelection._get_test_scores.
HIGHER_IS_BETTER_METRICS = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy']

# Fixed regularised XGBoost for the sweep. Shallow + heavily subsampled so a
# 2,800-column subset cannot simply memorise; early stopping adapts tree count per
# subset. Matches the spirit of the project's tuned space (max_depth 2-3, strong
# min_child_weight / gamma) without per-subset tuning.
DEFAULT_XGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.5,
    min_child_weight=20,
    gamma=1.0,
    reg_lambda=1.0,
    early_stopping_rounds=30,
    eval_metric='logloss',
    importance_type='total_gain',
    n_jobs=-1,
    verbosity=0,
    random_state=32,
)

BART_NAN_SENTINEL = -100.0


def score_predictions(y_true, proba, metric='roc_auc'):
    """Score positive-class probabilities. Mirrors the RFE class's metric dispatch."""
    if metric == 'roc_auc':
        return roc_auc_score(y_true, proba)
    if metric == 'log_loss':
        return log_loss(y_true, proba)
    if metric == 'brier':
        return brier_score_loss(y_true, proba)
    if metric == 'f1':
        return f1_score(y_true, proba.round())
    if metric == 'precision':
        return precision_score(y_true, proba.round())
    if metric == 'recall':
        return recall_score(y_true, proba.round())
    if metric == 'accuracy':
        return accuracy_score(y_true, proba.round())
    raise ValueError(
        'metric must be one of roc_auc, log_loss, brier, f1, precision, recall, '
        f'accuracy; got {metric!r}')


def to_model_matrix(df, features, sentinel=BART_NAN_SENTINEL):
    """Select ``features`` and fill NaN with ``sentinel`` for estimators (BART) that
    cannot ingest NaN. The sentinel must sit OUTSIDE the data range so the trees can
    split the missing region off; if a genuine value collides with it (possible once
    RANK_ONLY=False brings continuous aggregated values into the pool) this raises
    rather than silently merging real data into the missing-marker. Also asserts no
    NaN survives, so a stale/failed run fails loudly."""
    selected = df[list(features)]
    raw = selected.to_numpy(dtype=float)
    if np.any(raw == sentinel):
        raise ValueError(
            f'sentinel {sentinel} collides with a real feature value; pick a sentinel '
            f'outside the data range (e.g. below the global minimum)')
    matrix = selected.fillna(sentinel).to_numpy(dtype=float)
    if np.isnan(matrix).any():
        raise ValueError('NaN remains after sentinel fill (is the sentinel NaN?)')
    return matrix


def season_rolling_origin_folds(df, season_col='season', first_test_season=2018,
                                last_test_season=2025, valid_years=2,
                                min_train_seasons=3):
    """Expanding-window, season-blocked rolling-origin folds.

    For each test season ``s``: test = season == s; validation = the ``valid_years``
    seasons immediately before s; train = everything earlier. Folds are defined purely
    on the season column, so the two perspective rows of a game (which always share a
    season) never split across blocks. A fold is skipped unless at least
    ``min_train_seasons`` seasons precede its validation window.

    :returns: list of {'test_season', 'train', 'valid', 'test'} where the index values
        are row labels into ``df``.
    """
    seasons = sorted(df[season_col].unique())
    folds = []
    for test_season in range(first_test_season, last_test_season + 1):
        valid_low = test_season - valid_years
        if sum(1 for y in seasons if y < valid_low) < min_train_seasons:
            continue
        train = df.index[df[season_col] < valid_low]
        valid = df.index[(df[season_col] >= valid_low) & (df[season_col] < test_season)]
        test = df.index[df[season_col] == test_season]
        if len(test) == 0 or len(valid) == 0 or len(train) == 0:
            continue
        folds.append({'test_season': test_season,
                      'train': train, 'valid': valid, 'test': test})
    return folds


def make_xgb_fold_scorer(train_df, valid_df, test_df, target='target_win',
                         metric='roc_auc', xgb_params=None):
    """A fixed-XGBoost fold scorer. Early-stops on ``valid_df``, scores on ``test_df``.
    XGBoost routes NaN natively, so no sentinel fill is needed here."""
    from xgboost import XGBClassifier

    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
    y_train = train_df[target]
    y_valid = valid_df[target]
    y_test = test_df[target].to_numpy()

    def fit_score(features):
        features = list(features)
        if not features:
            raise ValueError('cannot fit on an empty feature subset')
        model = XGBClassifier(**params)
        model.fit(train_df[features], y_train,
                  eval_set=[(valid_df[features], y_valid)], verbose=False)
        proba = model.predict_proba(test_df[features])[:, 1]
        return score_predictions(y_test, proba, metric)

    return fit_score


def make_bart_fold_scorer(train_df, test_df, target='target_win', metric='brier',
                          sentinel=BART_NAN_SENTINEL, m=50, draws=1000, tune=1000,
                          chains=4, cores=4, seed=32):
    """A probit-link BART fold scorer for CONFIRMING a handful of top subsets (slow:
    ~1-3 min per fit). Fills NaN with the sentinel and asserts NaN-free before
    sampling, exactly as bart.ipynb does. Note: BART trains on ALL of ``train_df``
    with no early-stopping slice (it needs none); when comparing interaction signs
    against the XGBoost scorer, give XGBoost the same train rows or caveat that BART
    sees the extra validation seasons. PyMC is imported lazily (inside the fit) so this
    module -- and the empty-subset guard -- work without it installed."""
    y_train = train_df[target].to_numpy(dtype=int)
    y_test = test_df[target].to_numpy(dtype=int)

    def fit_score(features):
        features = list(features)
        if not features:
            raise ValueError('cannot fit on an empty feature subset')
        import pymc as pm
        import pymc_bart as pmb

        x_train = to_model_matrix(train_df, features, sentinel)
        x_test = to_model_matrix(test_df, features, sentinel)
        with pm.Model():
            x_data = pm.Data('X', x_train)
            mu = pmb.BART('mu', x_data, y_train, m=m)
            p = pm.Deterministic('p', pm.math.invprobit(mu))
            pm.Bernoulli('y', p=p, observed=y_train, shape=mu.shape)
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                              random_seed=seed, progressbar=False)
            pm.set_data({'X': x_test})
            ppc = pm.sample_posterior_predictive(
                idata, var_names=['p'], random_seed=seed, progressbar=False)
        preds = ppc.posterior_predictive['p'].mean(dim=['chain', 'draw']).to_numpy()
        # guard against a future pymc/pymc-bart change that fails to honour set_data
        # and returns train-length predictions instead of test-length ones
        assert len(preds) == len(y_test), (len(preds), len(y_test))
        return score_predictions(y_test, preds, metric)

    return fit_score
