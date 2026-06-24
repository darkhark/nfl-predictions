"""Phase 2 — BART with the full Madden launch-rating family.

BART backward-elimination over the champion-start features + all 188 madden_* columns,
then a final 4-chain BART, comparable to the BART champion (Run 10 = 0.708). Reuses the
in-repo BartBackwardElimination + bayes_logistic.evaluate. NaN -> -100 sentinel."""
import argparse
import json
import os
import numpy as np
import pandas as pd

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
