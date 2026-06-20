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
