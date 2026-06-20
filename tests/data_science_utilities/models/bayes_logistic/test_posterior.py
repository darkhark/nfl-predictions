import json
import os
import tempfile
import unittest

import numpy as np

from data_science_utilities.models.bayes_logistic import data_prep, model, posterior


_FITTED_CACHE = None


def _fitted():
    global _FITTED_CACHE
    if _FITTED_CACHE is None:
        import pandas as pd
        rng = np.random.default_rng(32)
        df = pd.DataFrame({"f0": rng.normal(size=120), "f1": rng.normal(size=120)})
        pre = data_prep.TrainFitPreprocessor(["f0", "f1"]).fit(df)
        X = pre.transform(df)
        y = (rng.uniform(size=120) < 0.5).astype(int)
        m = model.build_model(X, y, prior="normal")
        idata = model.sample(m, draws=60, tune=60, chains=2, seed=32)
        _FITTED_CACHE = (idata, pre)
    return _FITTED_CACHE


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
        self.assertIn("sd", summary["intercept"])
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
            with open(path) as f:
                loaded = json.load(f)
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
