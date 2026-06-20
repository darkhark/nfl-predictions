import unittest

import numpy as np

from data_science_utilities.models.bayes_logistic import model

SAMPLE_KW = dict(draws=60, tune=60, chains=2, seed=32)


def _toy(n=150, d=4, seed=32):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    beta = np.array([1.5, -1.0, 0.5, 0.0])
    p = 1.0 / (1.0 + np.exp(-(0.2 + X @ beta)))
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y


class TestBuildModel(unittest.TestCase):
    def test_normal_model_has_named_vars(self):
        X, y = _toy()
        m = model.build_model(X, y, prior="normal")
        names = {v.name for v in m.free_RVs} | {d.name for d in m.deterministics}
        self.assertIn("alpha", names)
        self.assertIn("beta", names)
        self.assertIn("p", names)

    def test_horseshoe_model_builds(self):
        X, y = _toy()
        m = model.build_model(X, y, prior="horseshoe")
        self.assertIsNotNone(m)

    def test_unknown_prior_raises(self):
        X, y = _toy()
        with self.assertRaises(ValueError):
            model.build_model(X, y, prior="banana")


class TestSamplePredictDiagnostics(unittest.TestCase):
    def test_predict_shapes_and_range(self):
        X, y = _toy()
        m = model.build_model(X, y, prior="normal")
        idata = model.sample(m, **SAMPLE_KW)
        mean_p, std_p = model.predict(m, idata, X[:20])
        self.assertEqual(mean_p.shape, (20,))
        self.assertEqual(std_p.shape, (20,))
        self.assertTrue(((mean_p >= 0) & (mean_p <= 1)).all())
        self.assertTrue((std_p >= 0).all())

    def test_predict_is_deterministic_with_seed(self):
        X, y = _toy()
        m = model.build_model(X, y, prior="normal")
        idata = model.sample(m, **SAMPLE_KW)
        a, _ = model.predict(m, idata, X[:10], seed=32)
        b, _ = model.predict(m, idata, X[:10], seed=32)
        np.testing.assert_allclose(a, b)

    def test_diagnostics_keys(self):
        X, y = _toy()
        m = model.build_model(X, y, prior="normal")
        idata = model.sample(m, **SAMPLE_KW)
        diag = model.diagnostics(idata)
        for k in ["max_rhat", "min_ess_bulk", "n_divergences", "passed"]:
            self.assertIn(k, diag)
        self.assertIsInstance(diag["n_divergences"], int)
