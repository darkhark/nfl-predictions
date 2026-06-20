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
    df = pd.DataFrame(
        {"y": np.asarray(y_true), "p": np.asarray(preds), "std": np.asarray(preds_std)}
    )
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
