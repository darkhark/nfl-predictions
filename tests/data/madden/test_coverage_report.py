import unittest
import numpy as np
import pandas as pd
from scripts.experiments import madden_coverage_report as rpt


class TestCoverageAgg(unittest.TestCase):
    def test_ovr_coverage_and_match_rate(self):
        tw = pd.DataFrame({
            'season': [2025, 2025],
            'madden_qb_ovr': [1.2, np.nan],
            'madden_rb_ovr': [0.3, 0.4],
            'team': ['BAL', 'SF'], 'week': [1, 1],
        })
        cov = rpt.season_ovr_coverage(tw)
        self.assertAlmostEqual(cov, 0.75)  # 3 of 4 _ovr cells non-null
