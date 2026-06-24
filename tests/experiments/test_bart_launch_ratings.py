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


def _synthetic(n_per_season=40):
    rng = np.random.default_rng(0)
    rows = []
    for s in (2019, 2020, 2021, 2022, 2023, 2024, 2025):
        for i in range(n_per_season):
            sig = rng.normal()
            rows.append({'season': s, 'week': (i % 5) + 1,
                         'f_qb': sig + rng.normal(0, 0.2),
                         'f_b': rng.normal(), 'f_c': rng.normal(),
                         'target_win': int(sig + rng.normal(0, 0.5) > 0)})
    return pd.DataFrame(rows)


class TestRunBartRfeWiring(unittest.TestCase):
    def test_rfe_uses_fit_fn_and_writes_csv(self):
        # Stub fit_fn: instant, deterministic; importance favors f_qb so it survives.
        def stub_fit(features, seed):
            incl = pd.Series({f: (3.0 if f == 'f_qb' else 1.0) for f in features})
            return {'validation_score': 0.22 - 0.001 * (3 - len(features)),
                    'variable_inclusion': incl}
        df = _synthetic()
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, 'bart_rfe.csv')
            best, hist = blr.run_bart_rfe(
                df, ['f_qb', 'f_b', 'f_c'], out, fit_fn=stub_fit,
                replicates=1, max_workers=1, min_features=1)
            self.assertTrue(os.path.exists(out))
            self.assertIn('f_qb', best)                 # highest importance survives
            self.assertIn('feature', pd.read_csv(out).columns)
            self.assertGreaterEqual(len(hist), 1)


class TestBartFitContract(unittest.TestCase):
    def test_tiny_real_fit_returns_contract(self):
        # ONE tiny real PyMC BART fit (slow ~20-40s) to prove the model + return shape.
        df = _synthetic(n_per_season=20)
        tr, va, _ = blr.split_seasons(df)
        fit = blr.make_bart_fit(tr, va, tr['target_win'].to_numpy(int),
                                va['target_win'].to_numpy(int),
                                draws=20, tune=20, chains=1, cores=1)
        out = fit(['f_qb', 'f_b', 'f_c'], seed=32)
        self.assertIn('validation_score', out)
        self.assertIsInstance(out['validation_score'], float)
        self.assertIsInstance(out['variable_inclusion'], pd.Series)
        self.assertEqual(set(out['variable_inclusion'].index), {'f_qb', 'f_b', 'f_c'})


class TestRunFinalBart(unittest.TestCase):
    def test_tiny_real_final_fit_and_eval(self):
        from data_science_utilities.models.bayes_logistic.evaluate import evaluate
        df = _synthetic(n_per_season=25)
        preds, p_std, holdout = blr.run_final_bart(
            df, ['f_qb', 'f_b', 'f_c'], draws=20, tune=20, chains=1, cores=1)
        self.assertEqual(len(preds), len(holdout))
        self.assertEqual(len(p_std), len(holdout))
        self.assertTrue(((preds >= 0) & (preds <= 1)).all())
        metrics = evaluate(holdout['target_win'].to_numpy(int), preds,
                           p_std=p_std, weeks=holdout['week'])
        for k in ('auroc', 'brier', 'per_week_auroc_mean',
                  'width_stratified_brier', 'reliability_curve'):
            self.assertIn(k, metrics)   # p_std present -> width_stratified_brier emitted
