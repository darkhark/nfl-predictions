# tests/experiments/test_xgb_launch_ratings.py
import os
import tempfile
import unittest
import numpy as np
import pandas as pd
from scripts.experiments import xgb_launch_ratings as xlr


class TestPureHelpers(unittest.TestCase):
    def test_build_rfe_pool_keeps_madden_drops_cumulative_and_target(self):
        cols = ['target_madden_qb_ovr', 'opp_madden_edge_ovr',
                'off_target_epa_per_play_cumulative_average',  # dropped (rank-only)
                'off_target_epa_per_play_rank',                # kept
                'week', 'target_win']                          # target_win dropped
        pool = xlr.build_rfe_pool(cols)
        self.assertIn('target_madden_qb_ovr', pool)
        self.assertIn('opp_madden_edge_ovr', pool)
        self.assertIn('off_target_epa_per_play_rank', pool)
        self.assertIn('week', pool)
        self.assertNotIn('off_target_epa_per_play_cumulative_average', pool)
        self.assertNotIn('target_win', pool)

    def test_madden_columns(self):
        feats = ['target_madden_qb_ovr', 'week', 'opp_madden_edge_ovr']
        self.assertEqual(xlr.madden_columns(feats),
                         ['target_madden_qb_ovr', 'opp_madden_edge_ovr'])

    def test_2025_coverage_and_guard(self):
        df = pd.DataFrame({
            'season': [2024, 2025, 2025],
            'target_madden_qb_ovr': [1.0, 2.0, np.nan],
            'opp_madden_qb_ovr': [0.5, np.nan, np.nan],
            'week': [1, 1, 2],
        })
        # 2025 rows: 4 _ovr cells, 1 non-null -> 0.25
        self.assertAlmostEqual(xlr.season_2025_ovr_coverage(df), 0.25)
        with self.assertRaises(RuntimeError):
            xlr.assert_madden_2025_coverage(df, min_cov=0.30)
        xlr.assert_madden_2025_coverage(df, min_cov=0.10)  # passes

    def test_selected_features_from_rfe_csv(self):
        # mimic get_features_in_dataframe().to_csv(): index = num_features, cols 0..N
        fdf = pd.DataFrame.from_dict(
            {3: ['a', 'b', 'c'], 2: ['a', 'b', None]}, orient='index')
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'rfe.csv')
            fdf.to_csv(p)
            self.assertEqual(xlr.selected_features_from_rfe_csv(p, 2), ['a', 'b'])
            self.assertEqual(xlr.selected_features_from_rfe_csv(p, 3), ['a', 'b', 'c'])


def _synthetic(n_per_season=30):
    rng = np.random.default_rng(0)
    seasons = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    rows = []
    for s in seasons:
        for i in range(n_per_season):
            signal = rng.normal()
            rows.append({
                'season': s, 'week': (i % 5) + 1,  # Fewer weeks for better balance per week
                'target_madden_qb_ovr': signal + rng.normal(0, 0.1),
                'opp_madden_edge_ovr': rng.normal(),
                'off_target_epa_per_play_rank': rng.normal(),
                'off_target_epa_per_play_cumulative_average': rng.normal(),  # rank-only drop
                'target_win': int(signal + rng.normal(0, 0.2) > -0.1),
            })
    return pd.DataFrame(rows)


class TestRunRfe(unittest.TestCase):
    def test_run_rfe_writes_csv_and_returns_best(self):
        df = _synthetic()
        cands = ['target_madden_qb_ovr', 'opp_madden_edge_ovr',
                 'off_target_epa_per_play_rank',
                 'off_target_epa_per_play_cumulative_average', 'week', 'target_win']
        tiny = dict(xlr.RFE_XGB_PARAMS, n_estimators=40)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, 'rfe.csv')
            best, rfe = xlr.run_rfe(df, cands, out, rfe_params=tiny,
                                    max_iter=3, min_features=2, n_folds=2)
            self.assertTrue(os.path.exists(out))
            self.assertIsInstance(best, (int, np.integer))
            # the cumulative col must never be in any selected row (rank-only filtered out)
            written = pd.read_csv(out, index_col=0)
            flat = set(written.values.ravel().tolist())
            self.assertNotIn('off_target_epa_per_play_cumulative_average', flat)
            self.assertNotIn('target_win', flat)


class TestGridAndResults(unittest.TestCase):
    def test_grid_eval_and_results(self):
        df = _synthetic(n_per_season=40)
        selected = ['target_madden_qb_ovr', 'opp_madden_edge_ovr',
                    'off_target_epa_per_play_rank', 'week']
        tiny_params = dict(xlr.RANDOM_XGB_PARAMS, n_estimators=[20, 30])
        tiny_search = dict(xlr.RANDOM_SEARCH_PARAMS, n_iter=2, cv=2)
        best, metrics, holdout = xlr.run_grid_and_eval(
            df, selected, search_params=tiny_params, search_kwargs=tiny_search)
        for k in ('auroc', 'accuracy', 'brier', 'logloss',
                  'per_week_auroc_mean', 'reliability_curve'):
            self.assertIn(k, metrics)
        res = xlr.build_results(metrics, selected, best, best_num_feats=len(selected))
        self.assertEqual(res['n_selected'], 4)
        self.assertEqual(set(res['madden_selected']),
                         {'target_madden_qb_ovr', 'opp_madden_edge_ovr'})
        self.assertIn('champion_deltas', res)
        self.assertIn('xgb_run11_auroc_delta', res['champion_deltas'])


class TestMainGuard(unittest.TestCase):
    def test_main_raises_on_low_2025_coverage(self):
        df = _synthetic(n_per_season=5)
        # blank out 2025 madden so coverage is 0 -> guard must fire
        df.loc[df['season'] == 2025,
               ['target_madden_qb_ovr', 'opp_madden_edge_ovr']] = np.nan
        with tempfile.TemporaryDirectory() as d:
            pq = os.path.join(d, 'sw.parquet')
            df.to_parquet(pq, index=False)
            fl = os.path.join(d, 'feats.csv')
            pd.DataFrame({'feature': ['target_madden_qb_ovr', 'opp_madden_edge_ovr',
                                      'off_target_epa_per_play_rank', 'week',
                                      'target_win']}).to_csv(fl, index=False)
            with self.assertRaises(RuntimeError):
                xlr.main(stage='rfe', parquet=pq, features_list=fl)


class TestRestrictToFamilies(unittest.TestCase):
    def test_keeps_only_requested_families(self):
        feats = ['target_madden_qb_ovr',            # madden_ratings
                 'off_target_epa_per_play_rank',     # pbp_phase1
                 'off_target_passing_yards',         # box_score
                 'week',                             # context_rest
                 'target_win']                       # unclassified -> dropped
        kept = xlr.restrict_to_families(feats, ['madden_ratings', 'context_rest'])
        self.assertIn('target_madden_qb_ovr', kept)
        self.assertIn('week', kept)
        self.assertNotIn('off_target_epa_per_play_rank', kept)
        self.assertNotIn('off_target_passing_yards', kept)
        self.assertNotIn('target_win', kept)


class TestDropMaddenDiffs(unittest.TestCase):
    def test_keeps_levels_drops_diffs(self):
        feats = ['target_madden_qb_ovr',                 # level -> kept
                 'target_madden_qb_ovr_diff_prev',        # diff -> dropped
                 'opp_madden_edge_ovr_diff_4g',           # diff -> dropped
                 'madden_matchup_pass_pro',               # matchup level -> kept
                 'off_target_epa_per_play_rank']          # non-madden -> kept
        out = xlr.drop_madden_diffs(feats)
        self.assertIn('target_madden_qb_ovr', out)
        self.assertIn('madden_matchup_pass_pro', out)
        self.assertIn('off_target_epa_per_play_rank', out)
        self.assertNotIn('target_madden_qb_ovr_diff_prev', out)
        self.assertNotIn('opp_madden_edge_ovr_diff_4g', out)
