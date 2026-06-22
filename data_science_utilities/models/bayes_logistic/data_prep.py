"""Leakage-safe data preparation for the Bayesian logistic regression model.

Matches the BART champion protocol (bart.ipynb): a fixed season split with the
frozen 54-feature Run-6 set. No PyMC import here so this stays fast and pure.
"""
import os
from dataclasses import dataclass

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
        self.medians_ = X.median()
        self.nan_cols_ = [c for c in self.feature_names if X[c].isna().any()]
        imputed = X.fillna(self.medians_)
        base_mean = imputed.mean().to_numpy(dtype=float)
        # ddof=1 (sample std): matches the test expectation; the brief's ddof=0 note was erroneous.
        base_std = imputed.std(ddof=1).to_numpy(dtype=float)
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
        if np.isnan(out).any():
            raise ValueError("NaN survived preprocessing — check imputation logic")
        return out
