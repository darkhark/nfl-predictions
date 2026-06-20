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
