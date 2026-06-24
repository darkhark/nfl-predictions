import unittest
import numpy as np
import pandas as pd
from scripts.experiments import bart_launch_ratings as blr


class TestPureHelpers(unittest.TestCase):
    def test_build_start_pool(self):
        champ = ['off_target_epa_per_play_rank', 'week', 'game_id']  # game_id is meta -> dropped
        madden = ['target_madden_qb_ovr', 'target_madden_qb_ovr_diff_prev', 'target_madden_qb_ovr']  # dup
        pool = blr.build_start_pool(champ, madden)
        self.assertEqual(pool, ['off_target_epa_per_play_rank', 'week',
                                'target_madden_qb_ovr', 'target_madden_qb_ovr_diff_prev'])
        self.assertNotIn('game_id', pool)        # meta dropped
        self.assertEqual(len(pool), len(set(pool)))  # deduped

    def test_to_bart_matrix_fills_sentinel(self):
        df = pd.DataFrame({'a': [1.0, np.nan], 'b': [np.nan, 2.0]})
        X = blr.to_bart_matrix(df, ['a', 'b'])
        self.assertFalse(np.isnan(X).any())
        self.assertEqual(X[1, 0], -100.0)
        self.assertEqual(X.dtype, float)

    def test_split_seasons(self):
        df = pd.DataFrame({'season': [2019, 2021, 2022, 2023, 2024, 2025],
                           'target_win': [0, 1, 0, 1, 0, 1], 'x': range(6)})
        tr, va, ho = blr.split_seasons(df)
        self.assertEqual(set(tr['season']), {2019, 2021})
        self.assertEqual(set(va['season']), {2022, 2023})
        self.assertEqual(set(ho['season']), {2024, 2025})
        self.assertEqual(len(tr) + len(va) + len(ho), 6)  # no row loss/duplication

    def test_build_results(self):
        metrics = {'auroc': 0.71, 'brier': 0.218}
        hist = pd.DataFrame({'num_features': [20, 16], 'validation_score': [0.222, 0.220]})
        res = blr.build_results(metrics, ['target_madden_qb_ovr', 'week'], hist, best_num_feats=16)
        self.assertEqual(res['n_selected'], 2)
        self.assertEqual(res['madden_selected'], ['target_madden_qb_ovr'])
        self.assertEqual(res['n_madden'], 1)
        self.assertIn('bart_run10_auroc_delta', res['champion_deltas'])
        self.assertAlmostEqual(res['champion_deltas']['bart_run10_auroc_delta'], 0.71 - 0.708, places=4)
        self.assertAlmostEqual(res['champion_deltas']['bart_run6_auroc_delta'], 0.71 - 0.705, places=4)
        self.assertEqual(len(res['validation_curve']), 2)
