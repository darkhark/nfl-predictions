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
        p_std = np.linspace(0.02, 0.2, 200)
        res = evaluate.evaluate(self.y, self.p, p_std=p_std, weeks=weeks)
        for k in ["auroc", "accuracy", "brier", "logloss", "per_week_auroc_mean"]:
            self.assertIn(k, res)
        self.assertIsInstance(res["auroc"], float)
