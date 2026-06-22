import json
import os
import tempfile
import unittest

import scripts.experiments.bayes_logistic as orch


class TestOrchestratorSmoke(unittest.TestCase):
    def test_smoke_run_writes_results(self):
        with tempfile.TemporaryDirectory() as d:
            res = orch.main([
                "--smoke", "--priors", "normal", "--outdir", d,
            ])
            self.assertIn("normal", res["variants"])
            row = res["variants"]["normal"]
            for k in ["auroc", "brier", "diagnostics"]:
                self.assertIn(k, row)
            self.assertTrue(os.path.exists(os.path.join(d, "results.json")))
            with open(os.path.join(d, "results.json")) as f:
                saved = json.load(f)
            self.assertIn("variants", saved)
            self.assertTrue(os.path.exists(os.path.join(d, "posterior_normal.nc")))
            self.assertTrue(os.path.exists(os.path.join(d, "coef_summary_normal.json")))
