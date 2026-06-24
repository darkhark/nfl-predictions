"""Phase 2 — BART with the full Madden launch-rating family.

BART backward-elimination over the champion-start features + all 188 madden_* columns,
then a final 4-chain BART, comparable to the BART champion (Run 10 = 0.708). Reuses the
in-repo BartBackwardElimination + bayes_logistic.evaluate. NaN -> -100 sentinel."""
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb
from sklearn.metrics import brier_score_loss
from data_science_utilities.models.bart.feature_selection.backward_elimination import (
    BartBackwardElimination,
)

from scripts.experiments.xgb_launch_ratings import madden_columns

RANDOM_SEED = 32
TARGET = 'target_win'
BART_NAN_SENTINEL = -100.0
META = {'game_id', 'season', 'season_type', 'opp_team', 'opp_score', 'target_team',
        'target_score', 'h_win', TARGET}
CHAMPIONS = {'bart_run6': {'auroc': 0.705, 'brier': 0.2194},
             'bart_run10': {'auroc': 0.708, 'brier': 0.2185}}


def build_start_pool(champion_start, madden_cols):
    """90 non-madden champion-start features + all madden columns; meta/target dropped,
    de-duplicated, order preserved (champion-start first, then madden)."""
    out, seen = [], set()
    for f in list(champion_start) + list(madden_cols):
        if f in META or f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def to_bart_matrix(df, features):
    """df[features] with NaN -> -100 sentinel as a float matrix (BART cannot ingest NaN)."""
    X = df[features].fillna(BART_NAN_SENTINEL).to_numpy(dtype=float)
    assert not np.isnan(X).any(), 'NaN remains after sentinel fill'
    return X


def split_seasons(df):
    """Seed-32 shuffle, then (train <2022, valid 2022-2023, holdout >=2024)."""
    data = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    train = data[data['season'] < 2022]
    valid = data[(data['season'] >= 2022) & (data['season'] < 2024)]
    holdout = data[data['season'] >= 2024].copy()
    return train, valid, holdout


def build_results(metrics, selected_features, history, best_num_feats):
    mads = madden_columns(selected_features)
    deltas = {}
    for name, champ in CHAMPIONS.items():
        deltas[f'{name}_auroc_delta'] = round(metrics['auroc'] - champ['auroc'], 4)
        deltas[f'{name}_brier_delta'] = round(metrics['brier'] - champ['brier'], 4)
    return {
        'config': {'seed': RANDOM_SEED, 'selection_metric': 'brier',
                   'nan_sentinel': BART_NAN_SENTINEL, 'best_num_feats': int(best_num_feats)},
        'n_selected': len(selected_features),
        'madden_selected': mads,
        'n_madden': len(mads),
        'metrics': metrics,
        'champion_deltas': deltas,
        'validation_curve': [{'n': int(r['num_features']),
                              'val_brier': float(r['validation_score'])}
                             for _, r in history.iterrows()],
    }


_RFE_PROGRESS_LOG = '/tmp/bart_launch_rfe_progress.log'


def _log_iter(row):
    with open(_RFE_PROGRESS_LOG, 'a') as fh:
        fh.write(f"iter: {row['num_features']} features, "
                 f"val_brier {row['validation_score']:.4f}\n")


def make_bart_fit(train_df, valid_df, y_train, y_valid, *,
                  m=50, draws=500, tune=1000, chains=2, cores=2):
    """Build a selection-grade BART fit_fn: train probit BART on <2022, score the
    2022-2023 validation slice with Brier, return chain-averaged variable_inclusion."""
    def bart_fit(features, seed):
        t0 = time.perf_counter()
        X_train = to_bart_matrix(train_df, features)
        X_valid = to_bart_matrix(valid_df, features)
        with pm.Model():
            X_data = pm.Data('X', X_train)
            mu = pmb.BART('mu', X_data, y_train, m=m)
            p = pm.Deterministic('p', pm.math.invprobit(mu))
            pm.Bernoulli('y', p=p, observed=y_train, shape=mu.shape)
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                              random_seed=seed, progressbar=False)
            pm.set_data({'X': X_valid})
            ppc = pm.sample_posterior_predictive(
                idata, var_names=['p'], random_seed=seed, progressbar=False)
        valid_preds = ppc.posterior_predictive['p'].mean(dim=['chain', 'draw']).to_numpy()
        # `variable_inclusion` is the PGBART sampler's per-feature usage count (pymc-bart
        # 0.9.2). The key is sampler-internal, not PyMC public API — re-verify on env upgrades.
        inclusion = pd.Series(
            idata.sample_stats['variable_inclusion'].mean(dim=['chain', 'draw']).to_numpy(),
            index=features)
        with open(_RFE_PROGRESS_LOG, 'a') as fh:
            fh.write(f"  fit: {len(features)} feat seed {seed} "
                     f"{time.perf_counter() - t0:.0f}s\n")
        return {'validation_score': float(brier_score_loss(y_valid, valid_preds)),
                'variable_inclusion': inclusion}
    return bart_fit


def run_bart_rfe(df, start_features, out_csv, *, fit_fn=None, replicates=6, max_workers=6,
                 drop_rate=0.2, min_features=10, fit_kwargs=None):
    """BART backward-elimination from start_features; writes the selected set (brier-1SE)
    and returns (best_features, history). Inject fit_fn for tests; else build a real one."""
    train, valid, _ = split_seasons(df)
    if fit_fn is None:
        fit_fn = make_bart_fit(train, valid, train[TARGET].to_numpy(dtype=int),
                               valid[TARGET].to_numpy(dtype=int), **(fit_kwargs or {}))
    rfe = BartBackwardElimination(fit_fn, drop_rate=drop_rate, min_features=min_features,
                                  replicates=replicates, max_workers=max_workers,
                                  base_seed=RANDOM_SEED, on_iteration=_log_iter)
    history = rfe.run(start_features)
    best = rfe.get_best_features_1se(higher_is_better=False)
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    pd.DataFrame({'feature': best}).to_csv(out_csv, index=False)
    return best, history


def run_final_bart(df, selected_features, *, m=50, draws=1000, tune=1000, chains=4, cores=4):
    """Final probit BART on the selected set; returns (holdout posterior-mean preds,
    posterior std, holdout_df). The std feeds the width-stratified Brier."""
    train, _, holdout = split_seasons(df)
    y_train = train[TARGET].to_numpy(dtype=int)
    X_train = to_bart_matrix(train, selected_features)
    X_holdout = to_bart_matrix(holdout, selected_features)
    with pm.Model():
        X_data = pm.Data('X', X_train)
        mu = pmb.BART('mu', X_data, y_train, m=m)
        p = pm.Deterministic('p', pm.math.invprobit(mu))
        pm.Bernoulli('y', p=p, observed=y_train, shape=mu.shape)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                          random_seed=RANDOM_SEED, progressbar=False)
        pm.set_data({'X': X_holdout})
        ppc = pm.sample_posterior_predictive(
            idata, var_names=['p'], random_seed=RANDOM_SEED, progressbar=False)
    post = ppc.posterior_predictive['p']
    preds = post.mean(dim=['chain', 'draw']).to_numpy()
    p_std = post.std(dim=['chain', 'draw']).to_numpy()
    return preds, p_std, holdout


from scripts.experiments.xgb_launch_ratings import assert_madden_2025_coverage
from data_science_utilities.models.bayes_logistic.evaluate import evaluate

PARQUET = 'data/predict_games/input_data/schedule_and_weekly.parquet'
CHAMPION_START_CSV = 'data/predict_games/model_features_in/rfe_features_kfolds_brier.csv'
CHAMPION_START_COUNT = 90
RFE_OUT = 'data/predict_games/model_features_in/bart_rfe_features_brier_madden.csv'
RESULTS_DIR = 'data/predict_games/bart_launch_ratings'


def load_champion_start(path, count):
    """The non-madden champion start features (row `count` of the XGBoost RFE table)."""
    table = pd.read_csv(path, index_col=0)
    return list(table.loc[count, :].dropna().values)


def main(stage='all', *, parquet=PARQUET, champion_start_csv=CHAMPION_START_CSV,
         champion_start_count=CHAMPION_START_COUNT, rfe_out=RFE_OUT,
         results_dir=RESULTS_DIR, selected=None):
    df = pd.read_parquet(parquet)
    assert_madden_2025_coverage(df)
    best = selected
    history = None

    if stage in ('rfe', 'all'):
        champ = load_champion_start(champion_start_csv, champion_start_count)
        madden = madden_columns(list(df.columns))
        start = build_start_pool(champ, madden)
        print(f'BART RFE start pool: {len(start)} features ({len(madden)} madden)')
        best, history = run_bart_rfe(df, start, rfe_out)
        print(f'1-SE selected {len(best)} features ({len(madden_columns(best))} madden)')
    if stage in ('final', 'all'):
        if best is None:
            best = list(pd.read_csv(rfe_out)['feature'])
        if history is None:
            # validation curve is an rfe-only artifact; standalone --stage final has none
            history = pd.DataFrame({'num_features': [len(best)], 'validation_score': [float('nan')]})
        preds, p_std, holdout = run_final_bart(df, best)
        metrics = evaluate(holdout[TARGET].to_numpy(dtype=int), preds,
                           p_std=p_std, weeks=holdout['week'])
        os.makedirs(results_dir, exist_ok=True)
        results = build_results(metrics, best, history, len(best))
        with open(os.path.join(results_dir, 'results.json'), 'w') as fh:
            json.dump(results, fh, indent=2)
            fh.write('\n')
        print(json.dumps(results['metrics'], indent=2))
        print('champion deltas:', results['champion_deltas'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['rfe', 'final', 'all'], default='all')
    args = ap.parse_args()
    main(stage=args.stage)
